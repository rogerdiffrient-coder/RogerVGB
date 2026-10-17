"""Train prompt-conditioned Roger Spark with clean story boundaries and validation."""
import argparse
import random
import re
from pathlib import Path

import torch
from torch.nn import functional as F
from datasets import load_dataset, load_from_disk
from tokenizers import ByteLevelBPETokenizer, Tokenizer
from roger_spark import (
    RogerSpark, TARGET_PARAMETERS, DEFAULT_WIDTH, DEFAULT_LAYERS,
    DEFAULT_HEADS, DEFAULT_MAX_SEQ, DEFAULT_VOCAB_SIZE, DEFAULT_FFN_DIMS,
)

STORY_CHAR_LIMIT = 12_000
TOKENIZER_TRAIN_STORIES = 100_000
TOKENIZER_DIR = Path("training/tokenizer_prompted_v4")
TRAIN_CACHE = Path("training/tokenized_tinystories_prompted_v4_train")
VALID_CACHE = Path("training/tokenized_tinystories_prompted_v4_validation")

ANIMALS = ("dog", "cat", "rabbit", "bunny", "bird", "horse", "puppy", "kitten", "duck", "fish")
ANIMAL_NORMALIZE = {"puppy": "dog", "kitten": "cat", "bunny": "rabbit"}

def story_prefix(story):
    """Create a prompt only when the story gives us reasonable evidence for it."""
    # Prefer explicit name/type constructions such as "Rex the dog" or "a dog named Rex".
    named_animal = re.search(
        r"\b([A-Z][a-z]{1,15})\s+the\s+(dog|cat|rabbit|bunny|bird|horse|puppy|kitten|duck|fish)\b",
        story,
    )
    animal_named = re.search(
        r"\b(?:a|the|my|little|small|big)\s+(dog|cat|rabbit|bunny|bird|horse|puppy|kitten|duck|fish)"
        r"\s+(?:was\s+)?(?:named|called)\s+([A-Z][a-z]{1,15})\b",
        story,
        re.IGNORECASE,
    )
    if named_animal:
        name, animal = named_animal.group(1), named_animal.group(2).lower()
        animal = ANIMAL_NORMALIZE.get(animal, animal)
        return f"Write me a story about {name} the {animal}.\nStory:\n"
    if animal_named:
        animal, name = animal_named.group(1).lower(), animal_named.group(2)
        animal = ANIMAL_NORMALIZE.get(animal, animal)
        return f"Write me a story about {name} the {animal}.\nStory:\n"

    # A character name alone is safer than inventing an animal identity.
    named = re.search(r"\b(?:named|called)\s+([A-Z][a-z]{1,15})\b", story)
    if named:
        return f"Write me a story about {named.group(1)}.\nStory:\n"

    # Otherwise condition on a broad topic that actually appears in the story.
    lower = story.lower()
    for animal in ANIMALS:
        if re.search(r"\b" + re.escape(animal) + r"\b", lower):
            normalized = ANIMAL_NORMALIZE.get(animal, animal)
            return f"Write me a story about a {normalized}.\nStory:\n"
    return "Write me a story.\nStory:\n"

def encode_stories(dataset, tokenizer, max_seq, eos_id, label):
    """Keep each story's windows separate; never concatenate unrelated stories."""
    rows = []
    block_len = max_seq + 1
    for story in dataset["text"]:
        story = str(story)[:STORY_CHAR_LIMIT].strip()
        if not story:
            continue
        ids = tokenizer.encode(story_prefix(story) + story).ids + [eos_id]
        # First window contains the instruction. Later windows are ordinary continuation
        # examples from this same story, never mixed with another story.
        for start in range(0, len(ids) - 1, max_seq):
            chunk = ids[start:start + block_len]
            if len(chunk) >= 2:
                rows.append({"input_ids": chunk})
    if not rows:
        raise ValueError(f"No usable {label} sequences were produced")
    from datasets import Dataset
    result = Dataset.from_list(rows)
    print(f"Prepared {len(result):,} independent {label} windows", flush=True)
    return result

def evaluate(model, tokenized, vocab_size, device, batches=64):
    """Estimate validation loss on a deterministic subset of held-out windows."""
    model.eval()
    rng = random.Random(2026)
    indices = list(range(len(tokenized)))
    if len(indices) > batches:
        indices = sorted(rng.sample(indices, batches))
    losses = []
    with torch.no_grad():
        for idx in indices:
            seq = tokenized[idx]["input_ids"]
            if len(seq) < 2:
                continue
            x = torch.tensor([seq[:-1]], dtype=torch.int32, device=device)
            y = torch.tensor([seq[1:]], dtype=torch.long, device=device)
            logits = model(x)
            losses.append(F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1)).item())
    model.train()
    return sum(losses) / max(1, len(losses))

def generate_greedy(model, tokenizer, prompt, eos_id, max_seq, max_tokens, device):
    prompt_ids = tokenizer.encode(prompt).ids[-max(1, max_seq - max_tokens):]
    generated = []
    # Suppress immediate repeated n-grams in the diagnostic only. This is not a
    # substitute for training quality and is deliberately kept out of model logits
    # during optimization and validation.
    with torch.no_grad():
        for _ in range(max_tokens):
            context = (prompt_ids + generated)[-max_seq:]
            logits = model(torch.tensor([context], dtype=torch.int32, device=device))[0, -1].clone()
            if len(generated) >= 2:
                last_bigram = tuple(generated[-2:])
                for token_id in range(tokenizer.get_vocab_size()):
                    if tuple(generated[-1:] + [token_id]) == last_bigram:
                        logits[token_id] = -float("inf")
            next_id = int(logits.argmax().item())
            if next_id == eos_id:
                break
            generated.append(next_id)
    return tokenizer.decode(generated)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=75_000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-seq", type=int, default=DEFAULT_MAX_SEQ)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--data", default="roneneldan/TinyStories")
    parser.add_argument("--checkpoint", default="training/checkpoints/roger_spark_latest.pt")
    parser.add_argument("--tokenizer-out", default="roger_spark_tokenizer.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--smoke-every", type=int, default=500)
    parser.add_argument("--smoke-tokens", type=int, default=40)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=64)
    parser.add_argument("--save-every", type=int, default=500)
    args = parser.parse_args()

    random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    if args.require_cuda and not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this run; refusing to start a slow CPU run.")
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS was requested, but this PyTorch installation does not have working MPS support.")
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but CUDA is not available.")
    print(f"Roger Spark v3: {TARGET_PARAMETERS:,} parameters; device={device}; context={args.max_seq}; lr={args.lr:g}", flush=True)

    tokenizer_json = TOKENIZER_DIR / "tokenizer.json"
    if not tokenizer_json.exists():
        print(f"Loading source dataset {args.data}", flush=True)
        source_train = load_dataset(args.data, split="train")
        source_validation = load_dataset(args.data, split="validation")
        if "text" not in source_train.column_names or "text" not in source_validation.column_names:
            raise ValueError("Expected text columns in both train and validation splits")
        TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
        bpe = ByteLevelBPETokenizer()
        sample_count = min(TOKENIZER_TRAIN_STORIES, len(source_train))
        bpe.train_from_iterator(
            (str(row["text"])[:STORY_CHAR_LIMIT] for row in source_train.select(range(sample_count))),
            vocab_size=DEFAULT_VOCAB_SIZE, min_frequency=2,
            special_tokens=["<pad>", "<eos>"],
        )
        tokenizer = bpe._tokenizer
        tokenizer.save(str(tokenizer_json))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_json))
        print("Loaded v4 tokenizer", flush=True)

    vocab_size = tokenizer.get_vocab_size()
    if vocab_size != DEFAULT_VOCAB_SIZE:
        raise ValueError(f"Tokenizer vocabulary is {vocab_size}, expected {DEFAULT_VOCAB_SIZE}")
    pad_id, eos_id = tokenizer.token_to_id("<pad>"), tokenizer.token_to_id("<eos>")
    if pad_id != 0 or eos_id != 1:
        raise ValueError(f"Unexpected special token IDs: pad={pad_id}, eos={eos_id}")
    Path(args.tokenizer_out).parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(args.tokenizer_out)

    if TRAIN_CACHE.exists() and VALID_CACHE.exists():
        train_data = load_from_disk(str(TRAIN_CACHE))
        valid_data = load_from_disk(str(VALID_CACHE))
        print(f"Loaded v4 caches: train={len(train_data):,}, validation={len(valid_data):,}", flush=True)
    else:
        source_train = load_dataset(args.data, split="train")
        source_validation = load_dataset(args.data, split="validation")
        train_data = encode_stories(source_train, tokenizer, args.max_seq, eos_id, "train")
        valid_data = encode_stories(source_validation, tokenizer, args.max_seq, eos_id, "validation")
        TRAIN_CACHE.parent.mkdir(parents=True, exist_ok=True)
        train_data.save_to_disk(str(TRAIN_CACHE))
        valid_data.save_to_disk(str(VALID_CACHE))
        print("Saved v4 train/validation caches", flush=True)

    model = RogerSpark(
        vocab_size=vocab_size, width=DEFAULT_WIDTH, layers=DEFAULT_LAYERS,
        heads=DEFAULT_HEADS, max_seq=args.max_seq, ffn_dims=list(DEFAULT_FFN_DIMS),
    ).to(device)
    assert model.count_parameters() == TARGET_PARAMETERS
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.1,
        fused=(device == "cuda"), foreach=(device == "cpu"),
    )
    start_step = 0
    checkpoint_path = Path(args.checkpoint)
    if args.resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint.get("training_version") != 4:
            raise SystemExit("This checkpoint uses an older training/data format. Start a fresh v4 run without --resume.")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])
        print(f"Resuming v4 checkpoint at step {start_step:,}", flush=True)

    use_cuda_amp = device == "cuda"
    amp_dtype = torch.bfloat16 if use_cuda_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda_amp and amp_dtype == torch.float16)
    recent_losses = []

    def run_diagnostics(step_label):
        model.eval()
        prompts = [
            "Write me a story about Rex the dog.\nStory:\n",
            "Write me a story about a cat.\nStory:\n",
            "Write me a story about Lily.\nStory:\n",
        ]
        print(f"\n=== SPARK GENERATION CHECK @ {step_label} ===", flush=True)
        for prompt in prompts:
            generated = generate_greedy(model, tokenizer, prompt, eos_id, args.max_seq, args.smoke_tokens, device)
            print(prompt + generated, flush=True)
            print("---", flush=True)
        model.train()

    model.train()
    for step in range(start_step + 1, args.steps + 1):
        batch = [train_data[random.randrange(len(train_data))]["input_ids"] for _ in range(args.batch_size)]
        # Windows have variable length; train only on the available next-token pairs.
        max_len = max(len(seq) for seq in batch)
        x_rows, y_rows = [], []
        for seq in batch:
            x = seq[:-1]
            y = seq[1:]
            x_rows.append(x + [pad_id] * (max_len - 1 - len(x)))
            y_rows.append(y + [-100] * (max_len - 1 - len(y)))
        x = torch.tensor(x_rows, dtype=torch.int32, device=device)
        y = torch.tensor(y_rows, dtype=torch.long, device=device)
        optimizer.zero_grad(set_to_none=True)
        if use_cuda_amp:
            with torch.autocast("cuda", dtype=amp_dtype):
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1), ignore_index=-100)
        else:
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1), ignore_index=-100)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss at step {step}")
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        recent_losses.append(loss.item())

        if step == 1 or step % 100 == 0:
            average = sum(recent_losses[-100:]) / len(recent_losses[-100:])
            print(f"step {step:,}/{args.steps:,} train_loss_100={average:.4f} device={device}", flush=True)
        if args.eval_every > 0 and step % args.eval_every == 0:
            val_loss = evaluate(model, valid_data, vocab_size, device, batches=args.eval_batches)
            train_average = sum(recent_losses[-100:]) / len(recent_losses[-100:])
            print(f"METRICS step={step} train_loss_100={train_average:.4f} validation_loss={val_loss:.4f}", flush=True)
        if args.smoke_every > 0 and step % args.smoke_every == 0:
            run_diagnostics(f"step {step:,}")
        if step % args.save_every == 0 or step == args.steps:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "training_version": 4, "step": step, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "config": model.config,
                "parameter_count": model.count_parameters(),
            }, checkpoint_path)
            print(f"Checkpoint saved at step {step:,}: {checkpoint_path}", flush=True)

    final_path = Path("training/roger_spark_final.pt")
    final_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "training_version": 4, "step": args.steps, "model": model.cpu().state_dict(),
        "config": model.config, "parameter_count": model.count_parameters(),
    }, final_path)
    print(f"Training complete: {model.count_parameters():,} parameters -> {final_path}", flush=True)

if __name__ == "__main__":
    main()
