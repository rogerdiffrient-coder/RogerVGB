"""Rebuild a resumable PyTorch checkpoint from Roger Spark's WebGPU export.

The export is quantized (INT8/FP16), so the reconstructed checkpoint is
approximate and optimizer history cannot be recovered. This is useful for
continuing experiments without keeping the original checkpoint in Git.
"""
import argparse
import json
from pathlib import Path
import shutil

import torch

from roger_spark import RogerSpark, SparkConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="webgpu/model", help="Directory with manifest.json, weights.bin and tokenizer.json.")
    parser.add_argument("--output", default="models/roger-0.1-spark", help="Directory for the reconstructed checkpoint.")
    args = parser.parse_args()

    source = Path(args.input)
    manifest_path = source / "manifest.json"
    weights_path = source / "weights.bin"
    tokenizer_path = source / "tokenizer.json"
    if not all(p.is_file() for p in (manifest_path, weights_path, tokenizer_path)):
        raise SystemExit(f"Expected manifest.json, weights.bin and tokenizer.json in {source}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "roger-spark-webgpu-v1":
        raise SystemExit("Unsupported export format.")
    config_data = manifest.get("config")
    tensors = manifest.get("tensors")
    if not isinstance(config_data, dict) or not isinstance(tensors, dict):
        raise SystemExit("Export manifest is missing config or tensor metadata.")

    config = SparkConfig(**config_data)
    model = RogerSpark(config)
    raw = weights_path.read_bytes()
    state = {}
    for name, entry in tensors.items():
        offset = int(entry["offset"])
        nbytes = int(entry["nbytes"])
        payload = raw[offset:offset + nbytes]
        if len(payload) != nbytes:
            raise SystemExit(f"Tensor {name} extends beyond weights.bin.")
        shape = tuple(int(d) for d in entry["shape"])
        dtype = entry["dtype"]
        if dtype == "int8":
            value = torch.frombuffer(bytearray(payload), dtype=torch.int8).to(torch.float32)
            value.mul_(float(entry["scale"]))
        elif dtype == "float16":
            value = torch.frombuffer(bytearray(payload), dtype=torch.float16).to(torch.float32)
        elif dtype == "float32":
            value = torch.frombuffer(bytearray(payload), dtype=torch.float32).clone()
        elif dtype == "int64":
            value = torch.frombuffer(bytearray(payload), dtype=torch.int64).clone()
        else:
            raise SystemExit(f"Unsupported tensor dtype {dtype!r} for {name}.")
        expected = 1
        for dim in shape:
            expected *= dim
        if value.numel() != expected:
            raise SystemExit(f"Tensor {name} has {value.numel()} values; expected {expected}.")
        state[name] = value.reshape(shape)

    try:
        model.load_state_dict(state, strict=True)
    except Exception as exc:
        raise SystemExit(f"Export tensors do not match the model architecture: {exc}") from exc

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(tokenizer_path, output / "tokenizer.json")
    config.save(output / "config.json")
    step = int(manifest.get("source_step") or 0)
    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "config": config.__dict__,
        "training_note": "Reconstructed from quantized WebGPU weights; optimizer state is not available.",
    }
    torch.save(checkpoint, output / "checkpoint.pt")
    print(f"Reconstructed {config.model_name} at step {step} into {output}")
    print("Note: INT8/FP16 quantization is lossy; optimizer state starts fresh on resume.")


if __name__ == "__main__":
    main()
