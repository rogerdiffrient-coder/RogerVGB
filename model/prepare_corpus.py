"""Build a balanced from-scratch corpus from original examples, Dolly, and TinyStories.

Only a deterministic slice is fetched through the Hugging Face dataset viewer;
large raw dataset files are not downloaded or committed.
"""
import argparse
import json
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROWS_API = "https://datasets-server.huggingface.co/rows"
BATCH_SIZE = 100


def fetch_batch(dataset, config, split, offset):
    query = urlencode({
        "dataset": dataset, "config": config, "split": split,
        "offset": offset, "length": BATCH_SIZE,
    })
    request = Request(f"{ROWS_API}?{query}",
                      headers={"User-Agent": "RogerVGB-corpus-builder/2.0"})
    last_error = None
    for attempt in range(4):
        try:
            with urlopen(request, timeout=45) as response:
                return json.loads(response.read().decode("utf-8")).get("rows", [])
        except Exception as error:
            last_error = error
            if attempt < 3:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Could not fetch {dataset} rows at offset {offset}: {last_error}")


def collect_stories(target_chars):
    dataset, config, split = "roneneldan/TinyStories", "default", "train"
    stories, total, offset = [], 0, 0
    while total < target_chars:
        rows = fetch_batch(dataset, config, split, offset)
        if not rows:
            break
        for item in rows:
            story = item.get("row", {}).get("text", "")
            if not isinstance(story, str):
                continue
            story = " ".join(story.split())
            if len(story) < 40:
                continue
            stories.append(story)
            total += len(story) + 2
            if total >= target_chars:
                break
        offset += len(rows)
        print(f"TinyStories: {total:,}/{target_chars:,} characters")
        if len(rows) < BATCH_SIZE:
            break
    if total < target_chars * 0.9:
        raise RuntimeError(f"TinyStories slice too small: {total:,} characters.")
    return stories, total


def collect_dialogue(target_chars):
    dataset, config, split = "databricks/databricks-dolly-15k", "default", "train"
    examples, total, offset = [], 0, 0
    while total < target_chars:
        rows = fetch_batch(dataset, config, split, offset)
        if not rows:
            break
        for item in rows:
            row = item.get("row", {})
            instruction = row.get("instruction", "")
            response = row.get("response", "")
            context = row.get("context", "")
            if not isinstance(instruction, str) or not isinstance(response, str):
                continue
            instruction = instruction.strip()
            response = response.strip()
            context = context.strip() if isinstance(context, str) else ""
            if not instruction or not response:
                continue
            prompt = instruction
            if context:
                prompt += "\nContext: " + context
            example = f"[USER] {prompt}\n[ROGER] {response}"
            examples.append(example)
            total += len(example) + 2
            if total >= target_chars:
                break
        offset += len(rows)
        print(f"Dolly instruction/dialogue: {total:,}/{target_chars:,} characters")
        if len(rows) < BATCH_SIZE:
            break
    if total < target_chars * 0.8:
        raise RuntimeError(f"Dolly slice too small: {total:,} characters.")
    return examples, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="data/roger_training.txt")
    parser.add_argument("--output", default="data/roger_training_expanded.txt")
    parser.add_argument("--target-story-chars", type=int, default=300_000)
    parser.add_argument("--target-dialogue-chars", type=int, default=600_000)
    args = parser.parse_args()
    base_path = Path(args.base)
    if not base_path.is_file():
        raise SystemExit(f"Base corpus not found: {base_path}")
    base_text = base_path.read_text(encoding="utf-8").strip()
    if args.target_story_chars < 100_000 or args.target_dialogue_chars < 100_000:
        raise SystemExit("Use at least 100,000 characters for each external dataset slice.")

    stories, story_chars = collect_stories(args.target_story_chars)
    dialogue, dialogue_chars = collect_dialogue(args.target_dialogue_chars)
    expanded = base_text + "\n\n" + "\n\n".join(dialogue) + "\n\n" + "\n\n".join(stories)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(expanded + "\n", encoding="utf-8")
    multiplier = len(expanded) / max(1, len(base_text))
    print(f"Original corpus: {len(base_text):,} characters")
    print(f"Added Dolly dialogue: {dialogue_chars:,} characters across {len(dialogue):,} examples")
    print(f"Added TinyStories: {story_chars:,} characters across {len(stories):,} stories")
    print(f"Combined corpus: {len(expanded):,} characters ({multiplier:.1f}x original)")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
