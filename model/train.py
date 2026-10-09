"""Train Roger 0.1 Spark from randomly initialized weights.

Example:
  python -m pip install -r requirements-model.txt
  python model/train.py --data data/roger_training.txt --steps 1000

This intentionally does not download or load any pretrained model.
"""
import argparse
import json
import math
import random
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
    parser = argparse.ArgumentParser(description="Train Roger 0.1 Spark from scratch.")
    parser.add_argument("--data", default="data/roger_training.txt", help="Plain-text training corpus.")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--tiny", action="store_true", help="Use a tiny model to test the pipeline.")
    parser.add_argument("--output", default="models/roger-0.1-spark")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"Training text not found: {data_path}\nCreate it from text you have permission to use.")
    text = data_path.read_text(encoding="utf-8")
    if len(text.strip()) < 1000:
        raise SystemExit("Training corpus is too small. Add substantially more clean, permitted text first.")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    base_config = tiny_config() if args.tiny else SparkConfig.from_file("model/config.json")
    tokenizer = train_tokenizer(data_path, output / "tokenizer.json", base_config.vocab_size)
    # Use the tokenizer's actual vocabulary size.
    base_config.vocab_size = tokenizer.get_vocab_size()
    base_config.save(output / "config.json")

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")
    print(f"Model: {base_config.model_name}")
    print("Training from random initialization; no pretrained weights are loaded.")

    token_ids = tokenizer.encode(text).ids
    sequence_length = min(args.sequence_length, base_config.context_length)
    dataset = TokenBlocks(token_ids, sequence_length)
    if len(dataset) == 0:
        raise SystemExit("Not enough tokenized text for one training sequence.")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    model = RogerSpark(base_config).to(device)
    print(f"Parameters: {model.parameter_count:,} ({model.parameter_count / 1e6:.1f}M)")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.1)
    model.train()
    step = 0
    while step < args.steps:
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(inputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            if step == 1 or step % 10 == 0:
                print(f"step {step}/{args.steps} | loss {loss.item():.4f}")
            if step % args.save_every == 0 or step == args.steps:
                checkpoint = {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "config": base_config.__dict__,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "training_note": "Random initialization; trained only on the corpus supplied by the user.",
                }
                torch.save(checkpoint, output / "checkpoint.pt")
                (output / "training_state.json").write_text(json.dumps({"step": step, "loss": loss.item()}, indent=2))
                print(f"Saved checkpoint at step {step}")
            if step >= args.steps:
                break
    print(f"Finished. Checkpoint: {output / 'checkpoint.pt'}")


if __name__ == "__main__":
    main()
