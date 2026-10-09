"""Run a trained Roger 0.1 Spark checkpoint locally. No Ollama required."""
import argparse
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from roger_spark import RogerSpark, SparkConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="models/roger-0.1-spark")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    checkpoint_path = model_dir / "checkpoint.pt"
    tokenizer_path = model_dir / "tokenizer.json"
    config_path = model_dir / "config.json"
    if not all(path.exists() for path in (checkpoint_path, tokenizer_path, config_path)):
        raise SystemExit(
            "No trained Roger Spark checkpoint found. Train it first with model/train.py. "
            "The repository does not include pre-trained weights."
        )

    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    config = SparkConfig.from_file(config_path)
    model = RogerSpark(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    print(f"{config.model_name} | {model.parameter_count / 1e6:.1f}M parameters | {device}")
    print("This is a raw language model, not instruction-tuned yet. Type /quit to exit.")
    while True:
        prompt = input("\nYou> ")
        if prompt.strip().lower() in {"/quit", "/exit"}:
            break
        encoded = tokenizer.encode(prompt).ids
        if not encoded:
            continue
        input_ids = torch.tensor([encoded], dtype=torch.long, device=device)
        generated = model.generate(input_ids, max_new_tokens=args.max_new_tokens,
                                   temperature=args.temperature)
        new_tokens = generated[0, len(encoded):].tolist()
        print("Roger> " + tokenizer.decode(new_tokens, skip_special_tokens=True))


if __name__ == "__main__":
    main()
