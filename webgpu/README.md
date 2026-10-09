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

## WebGPU status

**The export format exists; the complete WebGPU inference runtime is not implemented yet.**
The next runtime milestone must implement tokenization, embedding lookup, causal
attention, normalization, MLPs, sampling, and WebGPU compute shaders that read
this manifest and binary format. Until that runtime exists, use
`python3 model/chat.py` for inference with a trained checkpoint.

WebGPU requires a compatible browser/device and a secure context (usually
HTTPS or localhost). A fallback path should clearly report when WebGPU is
unavailable rather than silently claiming GPU acceleration.

## Quantization notes

INT8 quantization stores most matrix weights at roughly one byte per parameter,
plus scales and metadata. It is not a guarantee of a particular quality level;
evaluate generated text against the original FP32 checkpoint. The simple
per-tensor scheme here prioritizes a clear first export format. More advanced
per-channel or group-wise quantization can be added after correctness tests.
