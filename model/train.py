"""Train or continue training Roger Spark.

The default objective is ordinary next-token language modeling over every clean
paragraph, including explanatory prose and chat prompts. Use --loss-mode assistant
only when deliberately doing assistant-only fine-tuning. Windows are made per
paragraph with overlap so questions and answers are less likely to be split apart.
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
    """Overlapping fixed-length windows that never cross paragraph boundaries."""
    def __init__(self, examples, sequence_length, pad_id, loss_mode="all",
                 user_id=None, roger_id=None, eos_id=None):
        self.items = []
        stride = max(1, sequence_length // 2)
        for ids in examples:
            if len(ids) < 2:
                continue
            if loss_mode == "assistant":
                mask = [False] * len(ids)
                inside_answer = False
                for i, token_id in enumerate(ids):
                    if token_id == user_id:
                        inside_answer = False
                    elif token_id == roger_id:
                        inside_answer = True
                    elif token_id == eos_id:
                        mask[i] = inside_answer
                        inside_answer = False
                    elif inside_answer:
                        mask[i] = True
            else:
                mask = [True] * len(ids)

            for start in range(0, len(ids) - 1, stride):
                input_ids = ids[start:start + sequence_length]
                target_ids = ids[start + 1:start + sequence_length + 1]
                target_mask = mask[start + 1:start + sequence_length + 1]
                if not target_ids:
                    continue
                valid_length = len(target_ids)
                input_ids = input_ids[:valid_length]
                pad_count = sequence_length - valid_length
                input_ids += [pad_id] * pad_count
                target_ids += [-100] * pad_count
                target_ids = [
                    token if keep else -100
                    for token, keep in zip(target_ids, target_mask + [False] * pad_count)
                ]
                if any(token != -100 for token in target_ids):
                    self.items.append((
                        torch.tensor(input_ids, dtype=torch.long),
                        torch.tensor(target_ids, dtype=torch.long),
                    ))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


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
    parser.add_argument("--data", default="data/roger_training.txt")
    parser.add_argument("--steps", type=int, default=1000,
                        help="Training steps; additional steps when --resume is used.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--dropout", type=float, default=None,
                        help="Override model dropout, including when resuming.")
    parser.add_argument("--loss-mode", choices=("all", "assistant"), default="all",
                        help="Predict all text (recommended) or only Roger's answer tokens.")
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", default="model/config.json")
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", default="models/roger-0.1-spark")
    args = parser.parse_args()

    if args.steps < 1 or args.batch_size < 1 or args.save_every < 1:
        raise SystemExit("--steps, --batch-size, and --save-every must be positive.")
    if args.eval_every < 1 or not 0.0 < args.validation_fraction < 0.5:
        raise SystemExit("--eval-every must be positive and validation fraction between 0 and 0.5.")

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"Training text not found: {data_path}")
    text = data_path.read_text(encoding="utf-8")
    if len(text.strip()) < 1000:
        raise SystemExit("Training corpus is too small; add substantially more clean text.")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "checkpoint.pt"
    tokenizer_path = output / "tokenizer.json"
    previous_checkpoint = None

    if args.resume:
        if not checkpoint_path.exists() or not tokenizer_path.exists():
            raise SystemExit(f"Cannot resume: expected {checkpoint_path} and {tokenizer_path}.")
        previous_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config_data = previous_checkpoint.get("config")
        state = previous_checkpoint.get("model_state_dict")
        if not isinstance(config_data, dict) or not isinstance(state, dict):
            raise SystemExit("Checkpoint is missing config or model_state_dict.")
        base_config = SparkConfig(**config_data)
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        if tokenizer.get_vocab_size() != base_config.vocab_size:
            raise SystemExit("Tokenizer vocabulary does not match checkpoint config.")
        start_step = int(previous_checkpoint.get("step", 0))
        print(f"Resuming checkpoint at step {start_step}; preserving tokenizer and weights.")
    else:
        base_config = tiny_config() if args.tiny else SparkConfig.from_file(args.config)
        tokenizer = train_tokenizer(data_path, tokenizer_path, base_config.vocab_size)
        base_config.vocab_size = tokenizer.get_vocab_size()
        base_config.save(output / "config.json")
        start_step = 0
        print("Training from random initialization; no pretrained weights are loaded.")

    if args.dropout is not None:
        if not 0.0 <= args.dropout < 1.0:
            raise SystemExit("--dropout must be between 0 and 1.")
        base_config.dropout = args.dropout

    # Fingerprint includes preprocessing and objective. Old assistant-only scores
    # cannot be compared with scores from the new paragraph-windowed objective.
    data_sha256 = hashlib.sha256(
        f"paragraph-windows-v3:{args.loss_mode}:".encode("utf-8") + text.encode("utf-8")
    ).hexdigest()
    same_training_data = (
        previous_checkpoint is not None
        and previous_checkpoint.get("training_data_sha256") == data_sha256
    )
    best_checkpoint_path = output / "best_checkpoint.pt"
    if not same_training_data:
        best_checkpoint_path.unlink(missing_ok=True)
        print("Data/objective changed; resetting best-checkpoint selection.")

    paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
    if len(paragraphs) < 10:
        raise SystemExit("Need at least 10 paragraphs for a useful validation split.")

    def mark_answer_ends(paragraph):
        if "[USER]" not in paragraph or "[ROGER]" not in paragraph:
            return paragraph
        return re.sub(
            r"(\[ROGER\].*?)(?=\n\[USER\]|$)",
            lambda match: match.group(1).rstrip() + "\n[EOS]",
            paragraph,
            flags=re.DOTALL,
        )

    paragraphs = [mark_answer_ends(part) for part in paragraphs]
    random.Random(args.seed).shuffle(paragraphs)
    validation_count = max(1, int(len(paragraphs) * args.validation_fraction))
    validation_paragraphs = paragraphs[:validation_count]
    training_paragraphs = paragraphs[validation_count:]

    user_id = tokenizer.token_to_id("[USER]")
    roger_id = tokenizer.token_to_id("[ROGER]")
    eos_id = tokenizer.token_to_id("[EOS]")
    pad_id = tokenizer.token_to_id("[PAD]")
    if None in (user_id, roger_id, eos_id, pad_id):
        raise SystemExit("Tokenizer is missing a required dialogue marker.")

    train_examples = [tokenizer.encode(p).ids for p in training_paragraphs]
    validation_examples = [tokenizer.encode(p).ids for p in validation_paragraphs]
    sequence_length = min(args.sequence_length, base_config.context_length)
    dataset = TokenBlocks(train_examples, sequence_length, pad_id, args.loss_mode,
                          user_id, roger_id, eos_id)
    validation_dataset = TokenBlocks(validation_examples, sequence_length, pad_id, args.loss_mode,
                                     user_id, roger_id, eos_id)
    if len(dataset) < args.batch_size:
        raise SystemExit("Not enough training windows for the requested batch size.")
    if not len(validation_dataset):
        raise SystemExit("Not enough held-out validation windows.")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")
    print(f"Model: {base_config.model_name}")
    print(f"Paragraphs: {len(paragraphs)} total; {len(training_paragraphs)} train; "
          f"{len(validation_paragraphs)} validation")
    print(f"Loss mode: {args.loss_mode}; overlapping windows: "
          f"{len(dataset)} train, {len(validation_dataset)} validation; sequence length {sequence_length}")

    model = RogerSpark(base_config).to(device)
    if previous_checkpoint is not None:
        model.load_state_dict(previous_checkpoint["model_state_dict"])
    print(f"Parameters: {model.parameter_count:,} ({model.parameter_count / 1e6:.2f}M)")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.1
    )
    if previous_checkpoint is not None and previous_checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(previous_checkpoint["optimizer_state_dict"])
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate

    step = start_step
    target_step = start_step + args.steps
    best_validation_loss = (
        float(previous_checkpoint.get("best_validation_loss", float("inf")))
        if same_training_data else float("inf")
    )
    validation_loss = None

    @torch.no_grad()
    def evaluate():
        model.eval()
        losses = []
        for inputs, targets in validation_loader:
            _, val_loss = model(inputs.to(device), targets.to(device))
            if val_loss is not None and torch.isfinite(val_loss):
                losses.append(float(val_loss.item()))
        model.train()
        return sum(losses) / max(1, len(losses))

    model.train()
    while step < target_step:
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(inputs, targets)
            if not torch.isfinite(loss):
                raise SystemExit(f"Non-finite training loss at step {step + 1}.")
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
                    }, best_checkpoint_path)
                    print(f"New best validation checkpoint: {validation_loss:.4f}")

            if step % args.save_every == 0 or step == target_step:
                checkpoint = {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "config": base_config.__dict__,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_validation_loss": best_validation_loss,
                    "validation_loss": validation_loss,
                    "training_data_sha256": data_sha256,
                    "training_note": "RogerVGB custom model trained on the supplied corpus.",
                }
                torch.save(checkpoint, checkpoint_path)
                (output / "training_state.json").write_text(json.dumps({
                    "step": step,
                    "train_loss": float(loss.item()),
                    "validation_loss": validation_loss,
                    "best_validation_loss": best_validation_loss,
                    "loss_mode": args.loss_mode,
                    "training_windows": len(dataset),
                    "validation_windows": len(validation_dataset),
                }, indent=2), encoding="utf-8")
                print(f"Saved checkpoint at step {step}")
            if step >= target_step:
                break

    print(f"Finished. Checkpoint: {checkpoint_path}")
    print(f"Best validation loss this run: {best_validation_loss:.4f}")


if __name__ == "__main__":
    main()
