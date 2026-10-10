# RogerVGB

Roger Very Good Bot — your AI, your rules.

A modular AI workspace. The existing web chat currently has an Ollama adapter, while **Roger 0.1 Spark** is a separate, from-scratch model project that does not use Ollama, API keys, or pretrained weights.

## Roger 0.1 Spark — custom model

The initial model architecture and training code live in `model/`.

- Decoder-only Transformer initialized from random weights.
- Default architecture targets about **252 million parameters** with a 24,000-token vocabulary.
- Trains and runs in its own Python/PyTorch process; Apple Silicon MPS is used when available.
- No Ollama, API keys, pretrained model downloads, or hosted inference calls.
- Trained weights are generated locally and are not committed to Git.
- An experimental browser inference page uses WebGPU compute shaders with INT8 matrix weights and FP16 normalization vectors.

**Important:** a parameter count is not a measure of intelligence by itself. Spark is not a capable assistant yet; it needs a large, high-quality training corpus, extensive training, evaluation, and instruction tuning. The included corpus is only a tiny smoke-test sample.

See [model/README.md](model/README.md) for setup, training, inference, and quantized export commands. The [WebGPU README](webgpu/README.md) documents the browser-runtime plan.

**One-click training experiment:** open the repository's Actions tab and run **Train Roger Spark (experimental)** to continue training the published small model on GitHub's CPU runner and publish the new browser weights. See [the cloud-training guide](docs/roger-spark-cloud-training.md). This is only a small experiment: the current model and dataset are too small to become a capable assistant, and the workflow is not GPU-backed.

**Status note:** quantized export is implemented; the complete WebGPU inference runtime is not yet implemented.

## Existing web frontend

The current static frontend still includes:

1. **Chat:** streamed conversations and saved chat history.
2. **Custom assistants:** names, descriptions, instructions, and preferred models.
3. **Model arena:** compare answers from two configured local models.

### Run the existing frontend

1. Start your configured Ollama server and make a model available.
2. In this repository directory, run `python3 -m http.server 8000`.
3. Open http://localhost:8000.
4. Configure the Ollama URL and model in settings.

This describes the existing web chat only; it is not the Spark runtime. **Roger Spark currently runs separately and is not wired into the web UI yet.**

### Browser access / CORS

The existing frontend sends requests directly from your browser to the configured Ollama server. Depending on your setup, cross-origin requests may be blocked. Do not expose Ollama to the public internet.

## Modules

- `index.html`, `styles.css`: web interface
- `js/api.js`: existing Ollama HTTP/streaming adapter
- `js/storage.js`: browser-local persistence
- `js/ui.js`: DOM helpers and Markdown rendering
- `js/chat.js`: conversations and streamed chat
- `js/assistants.js`: custom assistant CRUD
- `js/arena.js`: model comparisons
- `js/main.js`: app wiring
- `model/roger_spark.py`: custom Transformer architecture
- `model/train.py`: from-scratch tokenizer/model training
- `model/chat.py`: standalone local inference
- `model/export_webgpu.py`: INT8/FP16 export and tokenizer bundling
- `webgpu/engine.js`: experimental WebGPU inference engine
- `webgpu/index.html`: browser chat page for Spark
- `webgpu/README.md`: export and run instructions

## Privacy and limitations

The current web chat stores conversations, assistants, and settings in browser localStorage and sends prompts to the configured Ollama server. Spark training and inference are local Python processes. No Spark weights are bundled in the repository.

## Development approach

Test the small training pipeline first, then improve the dataset and training loop. Don't mistake a successful training run or a 250M parameter count for proof that the model is intelligent; evaluate it with held-out text and concrete tasks.
