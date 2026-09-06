"""
Download CORD v2 dataset and convert to Donut-compatible format.

CORD provides receipt images + JSON annotations with menu items, subtotals, and totals.
Donut expects a special token sequence where each JSON key becomes <s_key>value</s_key>.

Usage:
    python prepare_data.py
"""

import json
import os
from pathlib import Path

from datasets import load_dataset
from transformers import DonutProcessor


DATASET_NAME = "naver-clova-ix/cord-v2"
BASE_MODEL = "naver-clova-ix/donut-base"
OUTPUT_DIR = Path("data")
TASK_PROMPT = "<s_cord-v2>"


def json_to_token_sequence(obj):
    """Recursively convert a JSON object into Donut's special token sequence.

    Example:
        {"nm": "Latte", "price": "5.00"}
        → "<s_nm>Latte</s_nm><s_price>5.00</s_price>"
    """
    if isinstance(obj, dict):
        tokens = ""
        for key, value in obj.items():
            tokens += f"<s_{key}>{json_to_token_sequence(value)}</s_{key}>"
        return tokens
    elif isinstance(obj, list):
        return "<sep/>".join(json_to_token_sequence(item) for item in obj)
    else:
        return str(obj) if obj is not None else ""


def collect_special_tokens(obj, tokens=None):
    """Walk the JSON and collect all unique <s_key> / </s_key> tokens."""
    if tokens is None:
        tokens = set()
    if isinstance(obj, dict):
        for key, value in obj.items():
            tokens.add(f"<s_{key}>")
            tokens.add(f"</s_{key}>")
            collect_special_tokens(value, tokens)
    elif isinstance(obj, list):
        for item in obj:
            collect_special_tokens(item, tokens)
    return tokens


def process_split(dataset_split, split_name, processor, output_dir):
    """Process one split of the CORD dataset and save images + metadata."""
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    img_dir = split_dir / "images"
    img_dir.mkdir(exist_ok=True)

    metadata = []
    all_special_tokens = set()

    for idx, example in enumerate(dataset_split):
        image = example["image"]
        ground_truth = json.loads(example["ground_truth"])

        gt_parse = ground_truth.get("gt_parse", ground_truth)
        token_seq = TASK_PROMPT + json_to_token_sequence(gt_parse)
        all_special_tokens.update(collect_special_tokens(gt_parse))

        img_filename = f"receipt_{idx:04d}.png"
        image.save(img_dir / img_filename)

        metadata.append({
            "file_name": f"images/{img_filename}",
            "ground_truth": token_seq,
        })

    with open(split_dir / "metadata.jsonl", "w") as f:
        for entry in metadata:
            f.write(json.dumps(entry) + "\n")

    print(f"  {split_name}: {len(metadata)} examples saved to {split_dir}")
    return all_special_tokens


def main():
    print(f"Loading dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME)
    print(f"  Splits: {list(dataset.keys())}")
    for split_name, split_data in dataset.items():
        print(f"  {split_name}: {len(split_data)} examples")

    print(f"\nLoading processor: {BASE_MODEL}")
    processor = DonutProcessor.from_pretrained(BASE_MODEL)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_tokens = set()

    print("\nProcessing splits...")
    for split_name in dataset:
        tokens = process_split(dataset[split_name], split_name, processor, OUTPUT_DIR)
        all_tokens.update(tokens)

    all_tokens.add(TASK_PROMPT)
    all_tokens = sorted(all_tokens)

    tokens_file = OUTPUT_DIR / "special_tokens.json"
    with open(tokens_file, "w") as f:
        json.dump(all_tokens, f, indent=2)
    print(f"\n{len(all_tokens)} special tokens saved to {tokens_file}")

    num_added = processor.tokenizer.add_tokens(all_tokens, special_tokens=True)
    print(f"{num_added} new tokens added to tokenizer")

    processor_dir = OUTPUT_DIR / "processor"
    processor.save_pretrained(processor_dir)
    print(f"Updated processor saved to {processor_dir}")

    print("\n--- Sample ---")
    sample_meta_path = OUTPUT_DIR / "train" / "metadata.jsonl"
    if sample_meta_path.exists():
        with open(sample_meta_path) as f:
            sample = json.loads(f.readline())
        print(f"File: {sample['file_name']}")
        print(f"Target: {sample['ground_truth'][:200]}...")

    print("\nPhase 1 complete. Data is ready for training.")


if __name__ == "__main__":
    main()
