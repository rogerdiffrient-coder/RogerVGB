"""Compare a browser ONNX model against the original PyTorch checkpoint."""
import argparse
from pathlib import Path

import onnxruntime as ort
import torch
from tokenizers import Tokenizer
from roger_spark import RogerSpark

ROOT = Path(__file__).resolve().parents[1]
PROMPT = "Write me a story about Lily.\nStory:\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="training/roger_spark_final.pt")
    parser.add_argument("--onnx", default="roger_spark.onnx")
    args = parser.parse_args()

    checkpoint = torch.load(ROOT / args.checkpoint, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    model = RogerSpark(
        vocab_size=int(cfg["vocab_size"]), width=int(cfg["width"]),
        layers=int(cfg["layers"]), heads=int(cfg["heads"]),
        max_seq=int(cfg["max_seq"]), ffn_dims=list(cfg["ffn_dims"]), dropout=0.0,
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    tokenizer = Tokenizer.from_file(str(ROOT / "roger_spark_tokenizer.json"))
    ids = tokenizer.encode(PROMPT).ids[-int(cfg["max_seq"]):]
    input_ids = torch.tensor([ids], dtype=torch.int32)

    with torch.no_grad():
        torch_logits = model(input_ids)[0, -1].float().numpy()

    session = ort.InferenceSession(
        str(ROOT / args.onnx), providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    onnx_logits = session.run(None, {input_name: input_ids.numpy()})[0][0, -1].astype("float32")

    difference = abs(torch_logits - onnx_logits)
    top_torch = torch_logits.argsort()[-10:][::-1].tolist()
    top_onnx = onnx_logits.argsort()[-10:][::-1].tolist()
    print(f"Checkpoint step: {checkpoint.get('step')}")
    print(f"Prompt token count: {len(ids)}")
    print(f"Max absolute logit difference: {difference.max():.6f}")
    print(f"Mean absolute logit difference: {difference.mean():.6f}")
    print(f"PyTorch top-10 token IDs: {top_torch}")
    print(f"ONNX top-10 token IDs:    {top_onnx}")
    print(f"Top-1 token agrees: {top_torch[0] == top_onnx[0]}")
    print("PyTorch next token:", repr(tokenizer.decode([top_torch[0]])))
    print("ONNX next token:   ", repr(tokenizer.decode([top_onnx[0]])))
    if not torch.isfinite(torch.tensor(onnx_logits)).all():
        raise SystemExit("FAIL: ONNX logits contain non-finite values.")
    if difference.max() > 0.15:
        raise SystemExit("FAIL: ONNX logits differ from the checkpoint beyond FP16 tolerance.")
    print("PASS: ONNX logits match the original checkpoint within FP16 tolerance.")


if __name__ == "__main__":
    main()
