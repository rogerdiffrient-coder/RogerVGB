"""Fast correctness checks for a trained Roger Spark checkpoint."""
import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer

from roger_spark import RogerSpark, SparkConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="models/roger-0.1-spark")
    args = parser.parse_args()
    root = Path(args.model_dir)
    for name in ("checkpoint.pt", "config.json", "tokenizer.json"):
        if not (root / name).is_file():
            raise SystemExit(f"Missing {root / name}; train or restore a checkpoint first.")

    tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
    for marker in ("[USER]", "[ROGER]"):
        marker_id = tokenizer.token_to_id(marker)
        if marker_id is None or tokenizer.encode(marker).ids != [marker_id]:
            raise SystemExit(f"Tokenizer special-token test failed for {marker}.")
    sample = "Hi, world! 2+2 = 4. café 😀"
    if tokenizer.decode(tokenizer.encode(sample).ids) != sample:
        raise SystemExit("Tokenizer round-trip test failed for punctuation or Unicode.")
    print("PASS: special-token IDs and ordinary-text round trip")

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    config = SparkConfig.from_file(root / "config.json")
    model = RogerSpark(config).to(device)
    best_path = root / "best_checkpoint.pt"
    checkpoint_path = best_path if best_path.is_file() else root / "checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    print(f"Testing checkpoint: {checkpoint_path.name} (step {checkpoint.get('step', 'unknown')})")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    print(f"Parameter count: {model.parameter_count:,} ({model.parameter_count / 1e6:.2f}M)")
    if model.parameter_count < 1_000_000:
        raise SystemExit("Model quality test failed: expected the multi-million-parameter Spark configuration.")

    # Smoke-test conversational generation without requiring calculator behavior yet.
    prompts = [
        "[USER] hi\n[ROGER]",
        "[USER] yooooo hows it going\n[ROGER]",
        "[USER] [USER] yooooo hows it going\n[ROGER] Not much, how about you?\n[USER] im good, thx!\n[ROGER]",
        "[USER] What is debugging?\n[ROGER]",
        "[USER] Give me an idea for a tiny game.\n[ROGER]",
    ]
    print(f"PASS: finite logits on {device}")
    for prompt in prompts:
        ids = tokenizer.encode(prompt).ids
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            logits, _ = model(input_ids)
        if not torch.isfinite(logits).all():
            raise SystemExit(f"Model smoke test failed: non-finite logits for {prompt!r}.")
        generated = model.generate(input_ids, max_new_tokens=32, temperature=0.0, top_k=1,
                                      eos_token_id=tokenizer.token_to_id("[EOS]"))
        answer = tokenizer.decode(generated[0, len(ids):].tolist(), skip_special_tokens=True).strip()
        if not answer:
            raise SystemExit(f"Model smoke test failed: no visible text for {prompt!r}.")
        print(f"Sample for {prompt.splitlines()[-1]}: {answer!r}")
    print("PASS: deterministic generation on varied conversational prompts")


if __name__ == "__main__":
    main()
