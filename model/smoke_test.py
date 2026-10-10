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
    checkpoint = torch.load(root / "checkpoint.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    prompt = "[USER] hi\n[ROGER]"
    ids = tokenizer.encode(prompt).ids
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits, _ = model(input_ids)
    if not torch.isfinite(logits).all():
        raise SystemExit("Model smoke test failed: logits contain NaN or infinity.")
    generated = model.generate(input_ids, max_new_tokens=24, temperature=0, top_k=1)
    answer = tokenizer.decode(generated[0, len(ids):].tolist(), skip_special_tokens=True).strip()
    if not answer:
        raise SystemExit("Model smoke test failed: generation returned no visible text.")
    print(f"PASS: finite logits and non-empty generation on {device}")
    print(f"Sample output (not a quality guarantee): {answer!r}")


if __name__ == "__main__":
    main()
