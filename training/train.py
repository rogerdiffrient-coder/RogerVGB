"""Train exact-25M Roger Spark on TinyStories plus a small chat-format mix."""
import argparse, json, random
from pathlib import Path
import torch
from torch.nn import functional as F
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from roger_spark import (
    RogerSpark, TARGET_PARAMETERS, DEFAULT_WIDTH, DEFAULT_LAYERS,
    DEFAULT_HEADS, DEFAULT_MAX_SEQ, DEFAULT_VOCAB_SIZE, DEFAULT_FFN_DIMS,
)

STORY_CHAR_LIMIT = 12_000
TOKENIZER_TRAIN_STORIES = 100_000
CHAT_MIX_PROBABILITY = 0.15

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=75_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-seq", type=int, default=DEFAULT_MAX_SEQ)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--data", default="roneneldan/TinyStories")
    parser.add_argument("--checkpoint", default="training/checkpoints/roger_spark_latest.pt")
    parser.add_argument("--tokenizer-out", default="roger_spark_tokenizer.json")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Loading {args.data} train split; device={device}", flush=True)
    dataset = load_dataset(args.data, split="train")
    if "text" not in dataset.column_names:
        raise ValueError(f"Expected TinyStories text column; got {dataset.column_names}")

    # Train a real byte-level BPE tokenizer from the story text, not from
    # prompt/response JSON fields. The byte alphabet prevents unknown bytes.
    tokenizer = ByteLevelBPETokenizer()
    sample_count = min(TOKENIZER_TRAIN_STORIES, len(dataset))
    def tokenizer_texts():
        for row in dataset.select(range(sample_count)):
            yield row["text"][:STORY_CHAR_LIMIT]
    tokenizer.train_from_iterator(
        tokenizer_texts(),
        vocab_size=DEFAULT_VOCAB_SIZE,
        min_frequency=2,
        special_tokens=["<pad>", "<eos>"],
    )
    vocab_size = tokenizer.get_vocab_size()
    if vocab_size != DEFAULT_VOCAB_SIZE:
        raise ValueError(f"Tokenizer vocabulary is {vocab_size}, expected exactly {DEFAULT_VOCAB_SIZE}")
    pad_id = tokenizer.token_to_id("<pad>")
    eos_id = tokenizer.token_to_id("<eos>")
    if pad_id != 0 or eos_id != 1:
        raise ValueError(f"Unexpected special token IDs: pad={pad_id}, eos={eos_id}")
    Path(args.tokenizer_out).parent.mkdir(parents=True, exist_ok=True)
    tokenizer._tokenizer.save(args.tokenizer_out)
    print(f"Tokenizer ready: vocab={vocab_size:,}; pad={pad_id}; eos={eos_id}", flush=True)

    def encode_stories(batch):
        encoded = []
        for story in batch["text"]:
            # The story is the response; this teaches the model the same
            # User/Roger boundary used by the browser chat UI.
            text = "User: Tell me a short story.\nRoger: " + story[:STORY_CHAR_LIMIT]
            ids = tokenizer.encode(text).ids
            ids.append(eos_id)
            encoded.append(ids)
        return {"input_ids": encoded}

    tokenized = dataset.map(
        encode_stories,
        batched=True,
        batch_size=1000,
        remove_columns=dataset.column_names,
        desc="Tokenizing TinyStories with strict per-story truncation",
    )
    del dataset
    chat_examples = []
    chat_path = Path("training/data.jsonl")
    if chat_path.exists():
        for line in chat_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if "prompt" in item and "response" in item:
                text = f"User: {str(item['prompt'])[:1000]}\nRoger: {str(item['response'])[:2000]}"
                ids = tokenizer.encode(text).ids[:args.max_seq]
                if ids:
                    ids.append(eos_id)
                    chat_examples.append(ids)
    print(f"Tokenized {len(tokenized):,} stories; supplemental chat examples={len(chat_examples):,}", flush=True)

    ffn_dims = list(DEFAULT_FFN_DIMS)
    model = RogerSpark(
        vocab_size=vocab_size, width=DEFAULT_WIDTH, layers=DEFAULT_LAYERS,
        heads=DEFAULT_HEADS, max_seq=args.max_seq, ffn_dims=ffn_dims,
    ).to(device)
    assert model.count_parameters() == TARGET_PARAMETERS
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
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
        batch_x, batch_y = [], []
        for _ in range(args.batch_size):
            if chat_examples and random.random() < CHAT_MIX_PROBABILITY:
                seq = chat_examples[random.randrange(len(chat_examples))]
            else:
                seq = tokenized[random.randrange(len(tokenized))]["input_ids"]
            if len(seq) > args.max_seq + 1:
                start = random.randrange(len(seq) - args.max_seq)
                seq = seq[start:start + args.max_seq + 1]
            else:
                seq = list(seq)
            x, y = seq[:-1], seq[1:]
            pad = args.max_seq - len(x)
            batch_x.append(F.pad(torch.tensor(x, dtype=torch.int32), (0, pad), value=pad_id))
            batch_y.append(F.pad(torch.tensor(y, dtype=torch.long), (0, pad), value=-100))
        x = torch.stack(batch_x).to(device)
        y = torch.stack(batch_y).to(device)
        optimizer.zero_grad(set_to_none=True)
        if use_cuda_amp:
            with torch.autocast("cuda", dtype=amp_dtype):
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1), ignore_index=-100)
        else:
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1), ignore_index=-100)
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
        if step % 1000 == 0 or step == args.steps:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "config": model.config, "parameter_count": model.count_parameters(),
            }, checkpoint_path)
            print(f"Checkpoint saved at step {step:,}: {checkpoint_path}", flush=True)

    final_path = Path("training/roger_spark_final.pt")
    final_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step": args.steps, "model": model.cpu().state_dict(),
        "config": model.config, "parameter_count": model.count_parameters(),
    }, final_path)
    print(f"Training complete: {model.count_parameters():,} parameters -> {final_path}", flush=True)

if __name__ == "__main__":
    main()
