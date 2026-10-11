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
    print("Conversation context is enabled for recent turns.")
    history = []
    eos_id = tokenizer.token_to_id("[EOS]")
    prompt_budget = max(1, config.context_length - min(args.max_new_tokens, config.context_length - 1))

    def encode_conversation(current_prompt):
        first = 0
        while True:
            turns = [
                f"[USER] {turn['user']}\\n[ROGER] {turn['assistant']}\\n[EOS]"
                for turn in history[first:]
            ]
            turns.append(f"[USER] {current_prompt}\\n[ROGER] ")
            encoded_ids = tokenizer.encode("\\n\\n".join(turns)).ids
            if len(encoded_ids) <= prompt_budget:
                return encoded_ids
            if first < len(history):
                first += 1
                continue
            raise ValueError(
                f"That message needs {len(encoded_ids)} tokens, but only "
                f"{prompt_budget} prompt tokens fit beside the generation budget. Shorten it."
            )

    while True:
        prompt = input("\\nYou> ")
        if prompt.strip().lower() in {"/quit", "/exit"}:
            break
        if not prompt.strip():
            continue
        try:
            encoded = encode_conversation(prompt.strip())
            input_ids = torch.tensor([encoded], dtype=torch.long, device=device)
            generated = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                eos_token_id=eos_id,
            )
            new_tokens = generated[0, len(encoded):].tolist()
            answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            print("Roger> " + answer)
            if answer:
                history.append({"user": prompt.strip(), "assistant": answer})
        except ValueError as error:
            print(f"Spark> {error}")

if __name__ == "__main__":
    main()
