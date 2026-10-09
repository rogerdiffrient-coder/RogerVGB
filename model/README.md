# Roger 0.1 Spark

Roger 0.1 Spark is the first **from-scratch** model experiment for RogerVGB.

## Non-negotiables

- No Ollama.
- No API keys or hosted model calls.
- No pretrained weights.
- The Transformer weights are initialized randomly by Roger's own code.
- Model files are created locally when you train; they are not checked into Git.

## Important reality check

The default architecture targets roughly 250 million parameters. Parameter count is capacity, **not intelligence**. A random 250M model is not a chatbot. The included starter text is only a pipeline test and will not make the model useful. A capable model requires a much larger, high-quality, legally usable training corpus, substantial compute, evaluation, and later instruction tuning.

A full 250M-parameter training run may be slow and memory-intensive on a 16 GB Mac. Start with the tiny configuration to validate the pipeline first. Do not expect the tiny model or starter dataset to answer questions intelligently.

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

## Train the 250M architecture

First replace `data/roger_training.txt` with a **large** clean corpus you have permission to train on. One plain-text file is supported in this initial version.

```bash
python3 model/train.py --data data/roger_training.txt --steps 1000 --batch-size 1 --sequence-length 256
```

The architecture is configured in `model/config.json`. Training from scratch usually needs far more than 1,000 steps to learn useful language. Keep the machine plugged in, monitor memory and temperature, and stop the run if macOS becomes unresponsive.

## Chat with a trained checkpoint

```bash
python3 model/chat.py
```

The command intentionally refuses to run before a checkpoint and tokenizer exist. No model weights are bundled in this repository.

## Current scope

This first milestone is a custom PyTorch Transformer and training loop, using Apple's MPS backend when available. It runs as a local Python process and does **not** yet run inside a browser. Browser inference would need a separate export/runtime implementation. It is also not yet instruction-tuned, so even after basic pretraining it may continue text rather than behave like a polished assistant.
