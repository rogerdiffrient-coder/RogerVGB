import json
from pathlib import Path
import torch

checkpoint = Path("training/roger_spark.pt")
out = Path("roger_spark_weights.json")
data = torch.load(checkpoint, map_location="cpu", weights_only=False)
state = data["state_dict"]
# The output head is tied to token.weight, so the browser reuses that matrix.
weights = {}
for name, tensor in state.items():
    if name == "lm_head.weight":
        continue
    weights[name] = tensor.detach().float().contiguous().view(-1).tolist()
bundle = {"format": "roger-spark-js-v1", "config": data["config"], "weights": weights}
out.write_text(json.dumps(bundle, separators=(",", ":")), encoding="utf-8")
print(f"Exported browser model: {out} ({out.stat().st_size:,} bytes)")
