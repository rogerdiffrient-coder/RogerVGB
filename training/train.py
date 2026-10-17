"""Train prompt-conditioned Roger Spark and run a Rex-story smoke test."""
import argparse, json, random, re
from pathlib import Path
import torch
from torch.nn import functional as F
from datasets import Dataset, load_dataset, load_from_disk
from tokenizers import ByteLevelBPETokenizer, Tokenizer
from roger_spark import (
    RogerSpark, TARGET_PARAMETERS, DEFAULT_WIDTH, DEFAULT_LAYERS,
    DEFAULT_HEADS, DEFAULT_MAX_SEQ, DEFAULT_VOCAB_SIZE, DEFAULT_FFN_DIMS,
)

STORY_CHAR_LIMIT = 12_000
TOKENIZER_TRAIN_STORIES = 100_000
TOKENIZER_DIR = Path("training/tokenizer_prompted_v2")
TOKENIZED_CACHE = Path("training/tokenized_tinystories_prompted_v2")

def story_prefix(story):
    # TinyStories often explicitly introduces a character as "named Rex".
    match = re.search(r"\b(?:named|called)\s+([A-Z][a-z]{1,15})\b", story)
    if match:
        return f"Write me a story about {match.group(1)}.\nStory:\n"
    return "Write me a story.\nStory:\n"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=75_000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-seq", type=int, default=DEFAULT_MAX_SEQ)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--data", default="roneneldan/TinyStories")
    parser.add_argument("--checkpoint", default="training/checkpoints/roger_spark_latest.pt")
    parser.add_argument("--tokenizer-out", default="roger_spark_tokenizer.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--save-every", type=int, default=5000)
    args = parser.parse_args()

    random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    if args.require_cuda and not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this run; refusing to start a slow CPU run.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Roger Spark v2: {TARGET_PARAMETERS:,} parameters; device={device}; context={args.max_seq}", flush=True)

    tokenizer_json = TOKENIZER_DIR / "tokenizer.json"
    dataset = None
    if not tokenizer_json.exists() or not TOKENIZED_CACHE.exists():
        print(f"Loading source dataset {args.data}", flush=True)
        dataset = load_dataset(args.data, split="train")
        if "text" not in dataset.column_names:
            raise ValueError(f"Expected text column; got {dataset.column_names}")

    if tokenizer_json.exists():
        tokenizer = Tokenizer.from_file(str(tokenizer_json))
        print("Loaded cached prompt-v2 tokenizer", flush=True)
    else:
        bpe_tokenizer = ByteLevelBPETokenizer()
        sample_count = min(TOKENIZER_TRAIN_STORIES, len(dataset))
        def tokenizer_texts():
            for row in dataset.select(range(sample_count)):
                yield row["text"][:STORY_CHAR_LIMIT]
        bpe_tokenizer.train_from_iterator(
            tokenizer_texts(), vocab_size=DEFAULT_VOCAB_SIZE, min_frequency=2,
            special_tokens=["<pad>", "<eos>"],
        )
        TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
        tokenizer = bpe_tokenizer._tokenizer
        tokenizer.save(str(tokenizer_json))

    vocab_size = tokenizer.get_vocab_size()
    if vocab_size != DEFAULT_VOCAB_SIZE:
        raise ValueError(f"Tokenizer vocabulary is {vocab_size}, expected {DEFAULT_VOCAB_SIZE}")
    pad_id, eos_id = tokenizer.token_to_id("<pad>"), tokenizer.token_to_id("<eos>")
    if pad_id != 0 or eos_id != 1:
        raise ValueError(f"Unexpected special token IDs: pad={pad_id}, eos={eos_id}")
    Path(args.tokenizer_out).parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(args.tokenizer_out)

    if TOKENIZED_CACHE.exists():
        tokenized = load_from_disk(str(TOKENIZED_CACHE))
        print(f"Loaded packed prompt-v2 cache: {len(tokenized):,} sequences", flush=True)
    else:
        def encode_and_pack(batch):
            stream = []
            for story in batch["text"]:
                story = story[:STORY_CHAR_LIMIT].strip()
                if not story:
                    continue
                stream.extend(tokenizer.encode(story_prefix(story) + story).ids)
                stream.append(eos_id)
            block_len = args.max_seq + 1
            usable = len(stream) // block_len * block_len
            blocks = [stream[i:i + block_len] for i in range(0, usable, block_len)]
            return {"input_ids": blocks}
        tokenized = dataset.map(
            encode_and_pack, batched=True, batch_size=1000,
            remove_columns=dataset.column_names,
            desc="Encoding and packing prompt-conditioned stories",
        )
        TOKENIZED_CACHE.parent.mkdir(parents=True, exist_ok=True)
        tokenized.save_to_disk(str(TOKENIZED_CACHE))
        print(f"Saved packed training cache: {TOKENIZED_CACHE}", flush=True)
    del dataset
    if len(tokenized) == 0:
        raise ValueError("Tokenized dataset is empty")

    model = RogerSpark(
        vocab_size=vocab_size, width=DEFAULT_WIDTH, layers=DEFAULT_LAYERS,
        heads=DEFAULT_HEADS, max_seq=args.max_seq, ffn_dims=list(DEFAULT_FFN_DIMS),
    ).to(device)
    assert model.count_parameters() == TARGET_PARAMETERS
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1, fused=(device == "cuda"))
    start_step = 0
    checkpoint_path = Path(args.checkpoint)
    if args.resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])
        print(f"Resuming at step {start_step:,}", flush=True)

    use_cuda_amp = device == "cuda"
    amp_dtype = torch.bfloat16 if use_cuda_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda_amp and amp_dtype == torch.float16)
    model.train()
    for step in range(start_step + 1, args.steps + 1):
        # Each cached row is already a full fixed-length block: no per-step padding
        # and far fewer Python/dataset operations than sampling variable-length stories.
        batch = [tokenized[random.randrange(len(tokenized))]["input_ids"] for _ in range(args.batch_size)]
        x = torch.tensor([seq[:-1] for seq in batch], dtype=torch.int32, device=device)
        y = torch.tensor([seq[1:] for seq in batch], dtype=torch.long, device=device)
        optimizer.zero_grad(set_to_none=True)
        if use_cuda_amp:
            with torch.autocast("cuda", dtype=amp_dtype):
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
        else:
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
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

        if step == 1 or step % 25 == 0:
            print(f"step {step:,}/{args.steps:,} loss={loss.item():.4f} device={device}", flush=True)
        # Avoid repeatedly writing a huge model+optimizer checkpoint mid-chunk.
        if step % args.save_every == 0 or step == args.steps:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "config": model.config, "parameter_count": model.count_parameters(),
            }, checkpoint_path)
            print(f"Checkpoint saved at step {step:,}: {checkpoint_path}", flush=True)

    # Smoke test requested by Roger: make sure a Rex story can be generated end-to-end.
    model.eval()
    smoke_prompt = "Write me a story about Rex.\nStory:\n"
    smoke_ids = tokenizer.encode(smoke_prompt).ids[-(args.max_seq - 40):]
    generated = []
    with torch.no_grad():
        for _ in range(40):
            context = (smoke_ids + generated)[-args.max_seq:]
            logits = model(torch.tensor([context], dtype=torch.int32, device=device))
            next_id = int(logits[0, -1].argmax().item())
            if next_id == eos_id:
                break
            generated.append(next_id)
    print("\n=== REX STORY SMOKE TEST ===", flush=True)
    print(smoke_prompt + tokenizer.decode(generated), flush=True)
    print("=== END REX STORY SMOKE TEST ===\n", flush=True)

    final_path = Path("training/roger_spark_final.pt")
    final_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step": args.steps, "model": model.cpu().state_dict(),
        "config": model.config, "parameter_count": model.count_parameters(),
    }, final_path)
    print(f"Training complete: {model.count_parameters():,} parameters -> {final_path}", flush=True)

if __name__ == "__main__":
    main()
