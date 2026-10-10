"""Train or continue training Roger 0.1 Spark.

Fresh training example:
  python model/train.py --tiny --steps 100 --output models/roger-spark-tiny

Continue an existing checkpoint:
  python model/train.py --resume --steps 500 --output models/roger-spark-tiny

When --resume is set, --steps means *additional* steps. The existing tokenizer
and model configuration are reused so token IDs do not silently change.
"""
import argparse
import hashlib
import json
import random
import re
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

from roger_spark import RogerSpark, SparkConfig, tiny_config


class TokenBlocks(Dataset):
    def __init__(self, token_ids, sequence_length):
        self.ids = torch.tensor(token_ids, dtype=torch.long)
        self.sequence_length = sequence_length

    def __len__(self):
        return max(0, (len(self.ids) - 1) // self.sequence_length)

    def __getitem__(self, index):
        start = index * self.sequence_length
        chunk = self.ids[start:start + self.sequence_length + 1]
        return chunk[:-1], chunk[1:]


def train_tokenizer(text_path, tokenizer_path, vocab_size):
    tokenizer_path = Path(tokenizer_path)
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[USER]", "[ROGER]"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train([str(text_path)], trainer)
    tokenizer.save(str(tokenizer_path))
    return tokenizer


def main():
    parser = argparse.ArgumentParser(description="Train or continue training Roger Spark.")
    parser.add_argument("--data", default="data/roger_training.txt", help="Plain-text training corpus.")
    parser.add_argument("--steps", type=int, default=1000,
                        help="Training steps; additional steps when --resume is used.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--validation-fraction", type=float, default=0.1,
                        help="Fraction of paragraphs held out for validation.")
    parser.add_argument("--eval-every", type=int, default=100,
                        help="Evaluate validation loss every N training steps.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed used to shuffle paragraph-level train/validation split.")
    parser.add_argument("--config", default="model/config.json", help="Model configuration used for fresh training.")
    parser.add_argument("--tiny", action="store_true", help="Use a tiny model for pipeline tests.")
    parser.add_argument("--resume", action="store_true",
                        help="Continue from OUTPUT/checkpoint.pt and reuse its tokenizer/config.")
    parser.add_argument("--output", default="models/roger-0.1-spark")
    args = parser.parse_args()

    if args.steps < 1:
        raise SystemExit("--steps must be at least 1.")
    if args.batch_size < 1 or args.save_every < 1:
        raise SystemExit("--batch-size and --save-every must be at least 1.")

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"Training text not found: {data_path}\nAdd clean text you have permission to use.")
    text = data_path.read_text(encoding="utf-8")
    data_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if len(text.strip()) < 1000:
        raise SystemExit("Training corpus is too small. Add substantially more clean, permitted text first.")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "checkpoint.pt"
    tokenizer_path = output / "tokenizer.json"
    previous_checkpoint = None

    if args.resume:
        if not checkpoint_path.exists() or not tokenizer_path.exists():
            raise SystemExit(
                f"Cannot resume: expected both {checkpoint_path} and {tokenizer_path}. "
                "Train once first or import an existing model export."
            )
        previous_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config_data = previous_checkpoint.get("config")
        state = previous_checkpoint.get("model_state_dict")
        if not isinstance(config_data, dict) or not isinstance(state, dict):
            raise SystemExit("Checkpoint is missing config or model_state_dict.")
        base_config = SparkConfig(**config_data)
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        if tokenizer.get_vocab_size() != base_config.vocab_size:
            raise SystemExit(
                f"Tokenizer vocabulary ({tokenizer.get_vocab_size()}) does not match "
                f"checkpoint config ({base_config.vocab_size}); refusing to corrupt token IDs."
            )
        start_step = int(previous_checkpoint.get("step", 0))
        print(f"Resuming checkpoint at step {start_step}; keeping its tokenizer and model config.")
    else:
        base_config = tiny_config() if args.tiny else SparkConfig.from_file(args.config)
        tokenizer = train_tokenizer(data_path, tokenizer_path, base_config.vocab_size)
        base_config.vocab_size = tokenizer.get_vocab_size()
        base_config.save(output / "config.json")
        start_step = 0
        print("Training from random initialization; no pretrained weights are loaded.")

    # Validation scores are comparable only when the exact training corpus is unchanged.
    same_training_data = (
        previous_checkpoint is not None
        and previous_checkpoint.get("training_data_sha256") == data_sha256
    )
    best_checkpoint_path = output / "best_checkpoint.pt"
    if not same_training_data:
        best_checkpoint_path.unlink(missing_ok=True)
        print("Training corpus changed (or has no recorded fingerprint); resetting best-checkpoint selection.")

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")
    print(f"Model: {base_config.model_name}")

    if not 0.0 < args.validation_fraction < 0.5:
        raise SystemExit("--validation-fraction must be greater than 0 and less than 0.5.")
    if args.eval_every < 1:
        raise SystemExit("--eval-every must be at least 1.")

    # Split on paragraph boundaries so a Q/A example is not cut in half.
    paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
    if len(paragraphs) < 10:
        raise SystemExit("Need at least 10 paragraphs to make a useful validation split.")

    # Teach an explicit assistant-turn boundary. Without this, the model only
    # sees the next [USER] marker after an answer and often rambles into another
    # turn. [EOS] is already a registered special token in the tokenizer.
    def mark_answer_ends(paragraph):
        if "[USER]" not in paragraph or "[ROGER]" not in paragraph:
            return paragraph
        return re.sub(
            r"(\\[ROGER\\].*?)(?=\\n\\[USER\\]|$)",
            lambda match: match.group(1).rstrip() + "\\n[EOS]",
            paragraph,
            flags=re.DOTALL,
        )

    paragraphs = [mark_answer_ends(part) for part in paragraphs]
    random.Random(args.seed).shuffle(paragraphs)
    validation_count = max(1, int(len(paragraphs) * args.validation_fraction))
    validation_text = "\n\n".join(paragraphs[:validation_count])
    training_text = "\n\n".join(paragraphs[validation_count:])
    train_ids = tokenizer.encode(training_text).ids
    validation_ids = tokenizer.encode(validation_text).ids
    sequence_length = min(args.sequence_length, base_config.context_length)
    dataset = TokenBlocks(train_ids, sequence_length)
    validation_dataset = TokenBlocks(validation_ids, sequence_length)
    if len(dataset) < args.batch_size:
        raise SystemExit(
            f"Not enough tokenized training text for batch size {args.batch_size}; "
            "add more text or reduce --batch-size/--sequence-length."
        )
    if len(validation_dataset) < 1:
        raise SystemExit("Not enough held-out validation tokens; add more text or reduce sequence length.")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False)
    print(f"Paragraphs: {len(paragraphs)} total; {len(paragraphs)-validation_count} train; "
          f"{validation_count} validation")
    print(f"Token blocks: {len(dataset)} train; {len(validation_dataset)} validation")

    model = RogerSpark(base_config).to(device)
    if previous_checkpoint is not None:
        model.load_state_dict(previous_checkpoint["model_state_dict"])
    print(f"Parameters: {model.parameter_count:,} ({model.parameter_count / 1e6:.2f}M)")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.1
    )
    if previous_checkpoint is not None and previous_checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(previous_checkpoint["optimizer_state_dict"])
        # Loading optimizer state restores its old learning rate too. Reapply the
        # CLI value so a continuation run can intentionally use a gentler rate.
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = args.learning_rate

    step = start_step
    target_step = start_step + args.steps
    best_validation_loss = (
        float(previous_checkpoint.get("best_validation_loss", float("inf")))
        if same_training_data else float("inf")
    )

    @torch.no_grad()
    def evaluate():
        model.eval()
        losses = []
        for batch_index, (inputs, targets) in enumerate(validation_loader):
            if batch_index >= 32:
                break
            _, val_loss = model(inputs.to(device), targets.to(device))
            losses.append(float(val_loss.item()))
        model.train()
        return sum(losses) / max(1, len(losses))

    model.train()
    while step < target_step:
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(inputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            if step == start_step + 1 or step % 10 == 0:
                print(f"step {step}/{target_step} | train_loss={loss.item():.4f}")

            should_evaluate = step % args.eval_every == 0 or step == target_step
            if should_evaluate:
                validation_loss = evaluate()
                print(f"step {step}/{target_step} | validation_loss={validation_loss:.4f}")
                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    torch.save({
                        "step": step,
                        "model_state_dict": model.state_dict(),
                        "config": base_config.__dict__,
                        "validation_loss": validation_loss,
                        "training_data_sha256": data_sha256,
                        "training_note": "Best held-out validation checkpoint; RogerVGB custom model.",
                    }, output / "best_checkpoint.pt")
                    print(f"New best validation checkpoint: {validation_loss:.4f}")

            if step % args.save_every == 0 or step == target_step:
                checkpoint = {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "config": base_config.__dict__,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_validation_loss": best_validation_loss,
                    "validation_loss": validation_loss if should_evaluate else None,
                    "training_data_sha256": data_sha256,
                    "training_note": "RogerVGB custom model; trained only on the corpus supplied by the user.",
                }
                torch.save(checkpoint, checkpoint_path)
                (output / "training_state.json").write_text(
                    json.dumps({
                        "step": step,
                        "train_loss": float(loss.item()),
                        "validation_loss": validation_loss if should_evaluate else None,
                        "best_validation_loss": best_validation_loss,
                    }, indent=2), encoding="utf-8"
                )
                print(f"Saved checkpoint at step {step}")
            if step >= target_step:
                break

    print(f"Finished. Checkpoint: {checkpoint_path}")
    print(f"Best validation loss this run: {best_validation_loss:.4f}")


if __name__ == "__main__":
    main()
