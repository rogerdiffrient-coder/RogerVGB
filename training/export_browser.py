"""Validate and export Roger Spark to ONNX FP16 for browser WebGPU inference."""
import json
from pathlib import Path
import torch
import onnx
from onnxconverter_common import float16
from roger_spark import RogerSpark, TARGET_PARAMETERS

checkpoint_path = Path("training/roger_spark_final.pt")
tokenizer_path = Path("roger_spark_tokenizer.json")
out_path = Path("roger_spark.onnx")
config_out_path = Path("roger_spark_config.json")
if not checkpoint_path.exists():
    raise SystemExit(f"Missing {checkpoint_path}; finish training first.")
if not tokenizer_path.exists():
    raise SystemExit(f"Missing {tokenizer_path}; tokenizer must be saved during training.")

checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
config = checkpoint["config"]
model = RogerSpark(
    vocab_size=config["vocab_size"], width=config["width"], layers=config["layers"],
    heads=config["heads"], max_seq=config["max_seq"], ffn_dims=config["ffn_dims"],
    dropout=config.get("dropout", 0.0),
)
model.load_state_dict(checkpoint["model"])
model.eval()
actual = model.count_parameters()
if actual != TARGET_PARAMETERS or checkpoint.get("parameter_count", actual) != TARGET_PARAMETERS:
    raise ValueError(f"Expected exactly {TARGET_PARAMETERS:,} trainable parameters, got {actual:,}")

dummy = torch.ones((1, 8), dtype=torch.int32)
fp32_path = Path("training/roger_spark_fp32.onnx")
torch.onnx.export(
    model, dummy, str(fp32_path),
    input_names=["input_ids"], output_names=["logits"],
    dynamic_axes={"input_ids": {1: "sequence"}, "logits": {1: "sequence"}},
    opset_version=17, do_constant_folding=True,
)
graph = onnx.load(str(fp32_path))
onnx.checker.check_model(graph)
graph_fp16 = float16.convert_float_to_float16(graph, keep_io_types=True)
onnx.checker.check_model(graph_fp16)
onnx.save(graph_fp16, str(out_path))
fp32_path.unlink(missing_ok=True)

browser_config = {
    "parameter_count": actual,
    "max_seq": int(config["max_seq"]),
    "vocab_size": int(config["vocab_size"]),
    "width": int(config["width"]),
    "layers": int(config["layers"]),
    "heads": int(config["heads"]),
    "format": "ONNX FP16 weights / int32 token IDs",
}
config_out_path.write_text(json.dumps(browser_config, indent=2) + "\n", encoding="utf-8")
print(json.dumps({
    "model": str(out_path), "bytes": out_path.stat().st_size,
    "config": str(config_out_path), **browser_config,
}, indent=2))
