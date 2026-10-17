"""Export a trained Roger Spark checkpoint for the GitHub Pages browser app.

Run from the repository root after training:
  python training/export_browser_onnx.py --checkpoint training/roger_spark_final.pt

This writes:
  roger_spark.onnx
  roger_spark_config.json
  roger_spark_tokenizer.json (copied from the checkpoint run's tokenizer if present)
"""
import argparse
import json
import shutil
from pathlib import Path

import torch
import onnx
from onnxconverter_common import float16
from roger_spark import RogerSpark

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="training/roger_spark_final.pt")
    parser.add_argument("--output", default="roger_spark.onnx")
    parser.add_argument("--fp32", action="store_true", help="Keep float32 weights instead of converting to float16.")
    args = parser.parse_args()

    checkpoint_path = (ROOT / args.checkpoint).resolve()
    output_path = (ROOT / args.output).resolve()
    if not checkpoint_path.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config")
    state = checkpoint.get("model")
    if not isinstance(config, dict) or not isinstance(state, dict):
        raise SystemExit("Checkpoint is missing its model/config fields.")
    if int(checkpoint.get("parameter_count", 0)) != 123_509_504:
        raise SystemExit("This exporter expects the 123,509,504-parameter Roger Spark checkpoint.")

    model = RogerSpark(
        vocab_size=int(config["vocab_size"]),
        width=int(config["width"]),
        layers=int(config["layers"]),
        heads=int(config["heads"]),
        max_seq=int(config["max_seq"]),
        ffn_dims=list(config["ffn_dims"]),
        dropout=0.0,
    )
    model.load_state_dict(state, strict=True)
    model.eval()

    dummy = torch.ones((1, 8), dtype=torch.long)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (dummy,),
        str(output_path),
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {1: "sequence"}, "logits": {1: "sequence"}},
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    graph = onnx.load(str(output_path))
    if not args.fp32:
        graph = float16.convert_float_to_float16(graph, keep_io_types=True)
        onnx.save(graph, str(output_path))

    onnx.checker.check_model(onnx.load(str(output_path)))

    browser_config = {
        "parameter_count": int(checkpoint["parameter_count"]),
        "vocab_size": int(config["vocab_size"]),
        "max_seq": int(config["max_seq"]),
        "width": int(config["width"]),
        "layers": int(config["layers"]),
        "heads": int(config["heads"]),
        "max_prompt_tokens": 48,
        "max_generated_tokens": 80,
        "format": "onnx-fp16-webgpu" if not args.fp32 else "onnx-fp32-webgpu",
        "training_step": int(checkpoint.get("step", 0)),
    }
    (ROOT / "roger_spark_config.json").write_text(json.dumps(browser_config, indent=2) + "\n")

    tokenizer_source = ROOT / "roger_spark_tokenizer.json"
    if not tokenizer_source.exists():
        tokenizer_source = ROOT / "training" / "tokenizer_prompted_v4" / "tokenizer.json"
        if tokenizer_source.exists():
            shutil.copy2(tokenizer_source, ROOT / "roger_spark_tokenizer.json")
    if not tokenizer_source.exists():
        raise SystemExit("Tokenizer JSON missing. Run training first so the prompted-v4 tokenizer is available.")

    print(f"Exported {checkpoint['parameter_count']:,} parameters from step {checkpoint.get('step', '?')}.")
    print(f"ONNX: {output_path} ({output_path.stat().st_size / 1024 / 1024:.1f} MiB)")
    print("Next: upload roger_spark.onnx and the updated config/tokenizer to the repository's Pages branch.")
    print("Note: the GitHub Pages app's browser tokenizer must match this training tokenizer exactly.")


if __name__ == "__main__":
    main()
