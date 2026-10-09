# Roger Spark WebGPU export

This folder is the browser-runtime track for Roger 0.1 Spark. The target is to
run inference in-browser with WebGPU and compact quantized weights, without
Ollama, hosted model APIs, or pretrained model weights.

## Export trained weights

Install the model dependencies, train a checkpoint, then run:

```bash
python3 model/export_webgpu.py
```

Optional paths:

```bash
python3 model/export_webgpu.py --checkpoint models/roger-0.1-spark/checkpoint.pt --output webgpu/model
```

The exporter writes:

- `manifest.json`: model configuration, tensor shapes, byte offsets, dtypes, and quantization scales.
- `weights.bin`: packed tensor bytes.

Matrix weights are quantized to signed INT8 with symmetric per-tensor scaling.
Other floating-point tensors, including normalization vectors, are stored as
FP16. This substantially reduces weight storage compared with FP32, although
actual quality and browser memory use must be measured. The output is generated
locally and should not be committed to Git.

## Run the experimental browser runtime

1. Train a checkpoint using the commands in [../model/README.md](../model/README.md).
2. Export it with `python3 model/export_webgpu.py`. The exporter copies the matching tokenizer into this folder.
3. From the repository root, start a local static server: `python3 -m http.server 8000`.
4. Open `http://localhost:8000/webgpu/` in a WebGPU-compatible browser.

The page in `index.html` loads the manifest, tokenizer, and quantized weights, then runs the custom Transformer inference path using WebGPU compute shaders. It uses per-token KV caches and browser-side sampling.

This runtime is **experimental and not yet validated across browsers/devices**. Its first implementation prioritizes a complete custom path over speed; the simple quantized matrix-vector kernels may be slow for the 250M architecture. A trained checkpoint is required; the repository does not contain trained weights. WebGPU requires a compatible browser/device and a secure context (usually HTTPS or localhost).

## Quantization notes

INT8 quantization stores most matrix weights at roughly one byte per parameter,
plus scales and metadata. It is not a guarantee of a particular quality level;
evaluate generated text against the original FP32 checkpoint. The simple
per-tensor scheme here prioritizes a clear first export format. More advanced
per-channel or group-wise quantization can be added after correctness tests.
