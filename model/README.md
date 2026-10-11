# Roger 0.1 Spark

Roger 0.1 Spark is the first **from-scratch** model experiment for RogerVGB.

## Non-negotiables

- No Ollama.
- No API keys or hosted model calls.
- No pretrained weights.
- The Transformer weights are initialized randomly by Roger's own code.
- Model files are created locally when you train; they are not checked into Git.

## Important reality check

The cloud-training architecture now targets roughly 100 million parameters (up from about 25 million). The separate local config may target a different size. Parameter count is capacity, **not intelligence**. A random 250M model is not a chatbot. The original hand-written corpus is supplemented at training time with roughly 600,000 characters of instruction/dialogue data from Dolly and 300,000 characters of TinyStories. This makes the combined corpus much larger while retaining a meaningful share of conversational examples. The fetched text is not committed to the repository; sources and licenses are documented in data/SOURCES.md. A capable model requires a much larger, high-quality, legally usable training corpus, substantial compute, evaluation, and later instruction tuning.

A 100M-parameter training run may be slow and memory-intensive on a 16 GB Mac. Start with the tiny configuration to validate the pipeline first. Do not expect the tiny model or starter dataset to answer questions intelligently.

## Setup

Install Python 3.10+ and the dependencies:

```bash
python3 -m pip install -r requirements-model.txt
```

## Test the pipeline

```bash
python3 model/train.py --tiny --steps 20 --batch-size 1 --sequence-length 64
```

The tiny run writes a checkpoint to `models/roger-0.1-spark/`. This tests that training works; it does not create a smart model.

## Continue training from a checkpoint

The training script now supports resuming while preserving the existing tokenizer and model configuration:

```bash
python3 model/train.py --resume --steps 500 --output models/roger-0.1-spark
```

With `--resume`, `--steps` means 500 **additional** steps. The output directory must already contain `checkpoint.pt` and `tokenizer.json`. Reusing the same tokenizer matters: retraining it changes token IDs and makes old weights incompatible. The checkpoint's optimizer state is restored when available.

## Train the cloud 100M architecture

First replace `data/roger_training.txt` with a **large** clean corpus you have permission to train on. One plain-text file is supported in this initial version.

```bash
python3 model/train.py --data data/roger_training.txt --steps 1000 --batch-size 1 --sequence-length 256
```

The cloud workflow architecture is configured in `model/cloud_config.json`. Training uses a capped validation sample, logs approximate tokens/second, and automatically uses CUDA mixed precision when a compatible GPU is available. The default GitHub-hosted runner is CPU-only. To use GPU mode, configure a trusted self-hosted Linux x64 runner with the `gpu` label, then manually dispatch the workflow with `compute=self-hosted-gpu`; GPU mode uses batch size 4. Never run untrusted pull-request code on a self-hosted runner.

## Chat with a trained checkpoint

```bash
python3 model/chat.py
```

The command intentionally refuses to run before a checkpoint and tokenizer exist. No model weights are bundled in this repository.

## Quantize and export for WebGPU

After training a checkpoint, export it into the browser-oriented packed format:

```bash
python3 model/export_webgpu.py
```

The exporter writes `webgpu/model/manifest.json` and `webgpu/model/weights.bin` locally. Matrix weights use symmetric per-tensor INT8; floating-point vectors such as normalization parameters use FP16. This reduces storage, but the impact on model quality should be evaluated.

**Important:** this creates quantized files, not a working browser model. The full WebGPU runtime still needs to be implemented. See [../webgpu/README.md](../webgpu/README.md) for the exact status and planned components.

## Current scope

The training and standalone inference path use a custom PyTorch Transformer, with Apple's MPS backend when available. Spark is not yet instruction-tuned, so even after pretraining it may continue text rather than behave like a polished assistant. WebGPU inference is a separate next milestone; do not mistake an exported quantized checkpoint for a functioning browser runtime.
