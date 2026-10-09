"""Export a trained Roger Spark checkpoint into a compact browser-runtime format.

Matrix weights use symmetric per-tensor INT8 quantization. Other floating-point
tensors (for example LayerNorm vectors) are stored as FP16. This is a simple,
portable export format for the planned WebGPU runtime; it is not itself an
inference engine.
"""
import argparse
import json
from pathlib import Path
import shutil

import torch


def quantize_int8(tensor):
    tensor = tensor.detach().cpu().float()
    max_abs = float(tensor.abs().max().item()) if tensor.numel() else 0.0
    scale = max_abs / 127.0 if max_abs > 0 else 1.0
    quantized = torch.clamp(torch.round(tensor / scale), -127, 127).to(torch.int8)
    return quantized, scale


def main():
    parser = argparse.ArgumentParser(description="Export Spark weights for a WebGPU browser runtime.")
    parser.add_argument("--checkpoint", default="models/roger-0.1-spark/checkpoint.pt")
    parser.add_argument("--output", default="webgpu/model")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}. Train Spark first.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict")
    config = checkpoint.get("config")
    if not isinstance(state, dict) or not isinstance(config, dict):
        raise SystemExit("Checkpoint is missing model_state_dict or config.")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer_source = checkpoint_path.parent / "tokenizer.json"
    if not tokenizer_source.exists():
        raise SystemExit(f"Tokenizer not found: {tokenizer_source}. Train Spark first.")
    shutil.copy2(tokenizer_source, output / "tokenizer.json")
    manifest = {
        "format": "roger-spark-webgpu-v1",
        "model_name": config.get("model_name", "Roger 0.1 Spark"),
        "quantization": "symmetric per-tensor INT8 for 2D+ tensors; FP16 for vectors and other tensors",
        "config": config,
        "tensors": {},
        "source_step": checkpoint.get("step"),
        "warning": "Experimental WebGPU runtime; validate on your browser and device.",
    }

    data_path = output / "weights.bin"
    offset = 0
    with data_path.open("wb") as blob:
        for name, value in state.items():
            tensor = value.detach().cpu().contiguous()
            if tensor.is_floating_point() and tensor.ndim >= 2:
                packed, scale = quantize_int8(tensor)
                raw = packed.numpy().tobytes(order="C")
                entry = {
                    "shape": list(tensor.shape),
                    "dtype": "int8",
                    "scale": scale,
                    "offset": offset,
                    "nbytes": len(raw),
                }
            elif tensor.is_floating_point():
                packed = tensor.to(torch.float16).contiguous()
                raw = packed.numpy().tobytes(order="C")
                entry = {
                    "shape": list(tensor.shape),
                    "dtype": "float16",
                    "offset": offset,
                    "nbytes": len(raw),
                }
            else:
                packed = tensor
                raw = packed.numpy().tobytes(order="C")
                entry = {
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype).replace("torch.", ""),
                    "offset": offset,
                    "nbytes": len(raw),
                }
            blob.write(raw)
            manifest["tensors"][name] = entry
            offset += len(raw)

    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Exported {len(state)} tensors to {output}")
    print(f"Packed weights: {data_path.stat().st_size / (1024 * 1024):.1f} MiB")
    print(f"Manifest: {manifest_path}")
    print(f"Tokenizer: {output / 'tokenizer.json'}")


if __name__ == "__main__":
    main()
