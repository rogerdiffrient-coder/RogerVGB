"""Build a larger from-scratch training corpus from the original examples + TinyStories.

TinyStories is fetched through the Hugging Face dataset viewer API, so its multi-GB
raw files are not downloaded or committed. The fetched stories are temporary build
inputs; this script writes a local expanded corpus for the training workflow.
"""
import argparse
import json
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

DATASET = "roneneldan/TinyStories"
API = "https://datasets-server.huggingface.co/rows"
BATCH_SIZE = 100


def fetch_batch(offset):
    query = urlencode({
        "dataset": DATASET,
        "config": "default",
        "split": "train",
        "offset": offset,
        "length": BATCH_SIZE,
    })
    request = Request(
        f"{API}?{query}",
        headers={"User-Agent": "RogerVGB-corpus-builder/1.0"},
    )
    last_error = None
    for attempt in range(4):
        try:
            with urlopen(request, timeout=45) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return payload.get("rows", [])
        except Exception as error:
            last_error = error
            if attempt < 3:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Could not fetch TinyStories rows at offset {offset}: {last_error}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="data/roger_training.txt")
    parser.add_argument("--output", default="data/roger_training_expanded.txt")
    parser.add_argument("--target-story-chars", type=int, default=1_000_000)
    args = parser.parse_args()

    base_path = Path(args.base)
    if not base_path.is_file():
        raise SystemExit(f"Base corpus not found: {base_path}")
    base_text = base_path.read_text(encoding="utf-8").strip()
    if args.target_story_chars < 100_000:
        raise SystemExit("Use at least 100,000 story characters for a useful expansion.")

    stories = []
    story_chars = 0
    offset = 0
    while story_chars < args.target_story_chars:
        rows = fetch_batch(offset)
        if not rows:
            break
        for item in rows:
            story = item.get("row", {}).get("text", "")
            if not isinstance(story, str):
                continue
            story = " ".join(story.replace("\\r", " ").split())
            if len(story) < 40:
                continue
            stories.append(story)
            story_chars += len(story) + 2
            if story_chars >= args.target_story_chars:
                break
        offset += len(rows)
        print(f"Fetched {offset} rows; collected {story_chars:,}/{args.target_story_chars:,} story characters.")
        if len(rows) < BATCH_SIZE:
            break

    if story_chars < args.target_story_chars * 0.9:
        raise RuntimeError(
            f"Dataset returned only {story_chars:,} story characters; "
            f"expected at least {args.target_story_chars * 0.9:,.0f}."
        )

    expanded = base_text + "\n\n" + "\n\n".join(stories)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(expanded + "\n", encoding="utf-8")
    multiplier = len(expanded) / max(1, len(base_text))
    print(f"Base corpus: {len(base_text):,} characters")
    print(f"Added TinyStories: {story_chars:,} characters across {len(stories):,} stories")
    print(f"Expanded corpus: {len(expanded):,} characters ({multiplier:.1f}x the original)")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
