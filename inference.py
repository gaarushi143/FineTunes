"""
Run inference with the fine-tuned Donut model on receipt images.

=== What this script does ===

1. Loads the base Donut model (the frozen ~200M parameter vision-language model)
2. Overlays the trained LoRA adapter on top (the small ~1-2M parameters we fine-tuned)
3. Feeds a receipt image through the model and generates a structured token sequence
4. Converts that token sequence back into a readable JSON dictionary

=== Two modes of operation ===

  Single image mode:
    python inference.py --image path/to/receipt.png
    → Prints the extracted JSON for that one receipt

  Test split evaluation mode (default):
    python inference.py
    → Runs on all test receipts from CORD, compares predictions to ground truth,
      and prints a metrics summary table

=== Prerequisites ===

- Run `prepare_data.py` first (creates data/ directory with processed CORD dataset)
- Run `train.py` first (creates checkpoints/final_adapter/ with the LoRA weights)
"""

import argparse
import json
import re
from pathlib import Path

import torch
from PIL import Image
from rich.console import Console
from rich.table import Table
from transformers import DonutProcessor, VisionEncoderDecoderModel
from peft import PeftModel


# ---------------------------------------------------------------------------
# Configuration — matches the constants in prepare_data.py and train.py
# ---------------------------------------------------------------------------

DATA_DIR = Path("data")
PROCESSOR_DIR = DATA_DIR / "processor"
DEFAULT_ADAPTER_PATH = Path("checkpoints/final_adapter")
BASE_MODEL = "naver-clova-ix/donut-base"
TASK_PROMPT = "<s_cord-v2>"
MAX_LENGTH = 768

console = Console()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(adapter_path):
    """
    Load the base Donut model and overlay the trained LoRA adapter.

    === How this works ===

    1. We load the DonutProcessor from data/processor/ — this is the tokenizer +
       image processor that prepare_data.py saved, with all the CORD special tokens
       (<s_menu>, </s_menu>, <s_nm>, etc.) already registered.

    2. We load the base VisionEncoderDecoderModel from the HuggingFace hub.
       This gives us the full Donut architecture with its original pre-trained weights.

    3. We resize the decoder's token embeddings to match our processor's vocabulary
       (which is larger than the base model's because we added special tokens).

    4. We wrap the model with PeftModel.from_pretrained(), which loads the LoRA
       adapter weights and injects them into the attention layers. The base weights
       stay frozen — the adapter's small A/B matrices are added on top.

    Args:
        adapter_path: Path to the directory containing the saved LoRA adapter
                      (the output of train.py, typically checkpoints/final_adapter/)

    Returns:
        model:     The Donut model with LoRA adapter applied, in eval mode
        processor: DonutProcessor with CORD special tokens
    """
    console.print(f"Loading processor from: [cyan]{PROCESSOR_DIR}[/]")
    processor = DonutProcessor.from_pretrained(PROCESSOR_DIR)

    console.print(f"Loading base model: [cyan]{BASE_MODEL}[/]")
    model = VisionEncoderDecoderModel.from_pretrained(BASE_MODEL)

    # Expand embedding layer to fit special tokens (same as in train.py)
    model.decoder.resize_token_embeddings(len(processor.tokenizer))

    console.print(f"Loading LoRA adapter from: [cyan]{adapter_path}[/]")
    model = PeftModel.from_pretrained(model, str(adapter_path))

    # Merge LoRA weights into the base model for faster inference.
    # This folds the A/B matrices into the original weight matrices so there's
    # no overhead during the forward pass. The result is mathematically identical
    # but runs faster because there are no extra matrix multiplications.
    model = model.merge_and_unload()

    # Set to evaluation mode — disables dropout and other training-only behaviors
    model.eval()

    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    console.print(f"Model loaded on: [green]{device}[/]\n")

    return model, processor


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict(model, processor, image):
    """
    Run inference on a single receipt image.

    === The generation process ===

    1. The image is preprocessed by DonutProcessor:
       - Resized to the model's expected dimensions (1920x2560)
       - Normalized and converted to a float tensor

    2. The Swin encoder processes the image into a sequence of visual embeddings.
       These embeddings capture the spatial layout, text shapes, and visual features
       of the receipt.

    3. The BART decoder generates tokens one at a time (autoregressive generation):
       - Starts with the task prompt token: <s_cord-v2>
       - At each step, it looks at the visual embeddings + all tokens generated so far
       - It predicts the most likely next token
       - Generation stops when it produces the end-of-sequence token (</s>)

    4. The raw output is a string of special tokens like:
       "<s_cord-v2><s_menu><s_nm>Latte</s_nm><s_price>5.00</s_price></s_menu>..."

    5. processor.token2json() parses that token string back into a Python dict:
       {"menu": [{"nm": "Latte", "price": "5.00"}], ...}

    Args:
        model:     The fine-tuned Donut model (base + LoRA merged)
        processor: DonutProcessor with CORD special tokens
        image:     PIL Image of a receipt

    Returns:
        A Python dictionary with the extracted receipt fields
    """
    device = model.device

    # Preprocess the image into pixel values the Swin encoder expects
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

    # Prepare the decoder's starting input — just the task prompt token <s_cord-v2>
    # The decoder needs this as a "seed" to know what kind of output to generate.
    decoder_input_ids = processor.tokenizer(
        TASK_PROMPT,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids.to(device)

    # Generate the full token sequence autoregressively
    with torch.no_grad():
        outputs = model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=MAX_LENGTH,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            # Prevent the model from generating unknown tokens
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
            use_cache=True,
        )

    # Decode token IDs back into a string of special tokens
    raw_sequence = processor.batch_decode(outputs)[0]

    # Clean up the sequence:
    #   1. Remove </s> (end of sequence) and <pad> tokens
    #   2. Remove the leading task prompt token (<s_cord-v2>)
    raw_sequence = raw_sequence.replace(processor.tokenizer.eos_token, "") \
                               .replace(processor.tokenizer.pad_token, "")
    raw_sequence = re.sub(r"<.*?>", "", raw_sequence, count=1).strip()

    # Convert the special token sequence into a Python dictionary
    # e.g., "<s_menu><s_nm>Latte</s_nm></s_menu>" → {"menu": [{"nm": "Latte"}]}
    result = processor.token2json(raw_sequence)

    return result


# ---------------------------------------------------------------------------
# Single image mode
# ---------------------------------------------------------------------------

def run_single_image(model, processor, image_path):
    """Run inference on one image and pretty-print the result."""
    console.print(f"Processing: [cyan]{image_path}[/]\n")
    image = Image.open(image_path).convert("RGB")
    result = predict(model, processor, image)
    console.print_json(json.dumps(result, indent=2))


# ---------------------------------------------------------------------------
# Test split evaluation mode
# ---------------------------------------------------------------------------

def flatten_json(obj, prefix=""):
    """
    Flatten a nested dict into dot-separated keys for field-level comparison.

    Example:
        {"total": {"total_price": "21.60"}}
        → {"total.total_price": "21.60"}

    This lets us compare individual fields between prediction and ground truth.
    """
    flat = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            new_key = f"{prefix}.{key}" if prefix else key
            flat.update(flatten_json(value, new_key))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            flat.update(flatten_json(item, f"{prefix}[{i}]"))
    else:
        flat[prefix] = str(obj) if obj is not None else ""
    return flat


def run_evaluation(model, processor):
    """
    Run the model on all test split receipts and compute metrics.

    === Metrics computed ===

    1. Valid JSON rate: What percentage of outputs successfully parse into
       a dictionary? The fine-tuned model should achieve close to 100%.

    2. Field accuracy: For each receipt, we flatten both the predicted and
       ground truth JSON into dot-separated keys (e.g., "total.total_price"),
       then check how many fields match exactly.

    3. Per-example results table: Shows each test receipt with its accuracy
       so you can spot which receipts the model struggles with.
    """
    test_dir = DATA_DIR / "test"
    metadata_path = test_dir / "metadata.jsonl"

    if not metadata_path.exists():
        console.print("[red]ERROR: data/test/metadata.jsonl not found.[/]")
        console.print("Run `python prepare_data.py` first.")
        return

    with open(metadata_path) as f:
        samples = [json.loads(line) for line in f]

    console.print(f"Evaluating on [cyan]{len(samples)}[/] test receipts...\n")

    results = []
    valid_json_count = 0
    total_fields = 0
    correct_fields = 0

    for idx, sample in enumerate(samples):
        image_path = test_dir / sample["file_name"]
        image = Image.open(image_path).convert("RGB")

        # Get the ground truth by parsing the token sequence back to JSON
        gt_token_seq = sample["ground_truth"]
        gt_clean = gt_token_seq.replace(processor.tokenizer.eos_token, "") \
                               .replace(processor.tokenizer.pad_token, "")
        gt_clean = re.sub(r"<.*?>", "", gt_clean, count=1).strip()
        gt_json = processor.token2json(gt_clean)

        # Run prediction
        pred_json = predict(model, processor, image)

        # Check if output is a valid dict
        is_valid = isinstance(pred_json, dict)
        if is_valid:
            valid_json_count += 1

        # Field-level comparison
        gt_flat = flatten_json(gt_json)
        pred_flat = flatten_json(pred_json) if is_valid else {}

        n_fields = len(gt_flat)
        n_correct = sum(1 for k in gt_flat if gt_flat[k] == pred_flat.get(k))
        total_fields += n_fields
        correct_fields += n_correct

        accuracy = n_correct / n_fields if n_fields > 0 else 0.0

        results.append({
            "idx": idx,
            "file": sample["file_name"],
            "valid_json": is_valid,
            "fields": n_fields,
            "correct": n_correct,
            "accuracy": accuracy,
        })

        console.print(f"  [{idx + 1}/{len(samples)}] {sample['file_name']} — "
                       f"accuracy: {accuracy:.0%}")

    # --- Print summary ---
    console.print("\n")

    # Summary table
    table = Table(title="Evaluation Results")
    table.add_column("#", style="dim", width=4)
    table.add_column("Receipt", style="cyan")
    table.add_column("Valid JSON", justify="center")
    table.add_column("Fields", justify="right")
    table.add_column("Correct", justify="right")
    table.add_column("Accuracy", justify="right")

    for r in results:
        valid_str = "[green]Yes[/]" if r["valid_json"] else "[red]No[/]"
        acc_str = f"{r['accuracy']:.0%}"
        acc_color = "green" if r["accuracy"] >= 0.8 else "yellow" if r["accuracy"] >= 0.5 else "red"
        table.add_row(
            str(r["idx"]),
            r["file"],
            valid_str,
            str(r["fields"]),
            str(r["correct"]),
            f"[{acc_color}]{acc_str}[/]",
        )

    console.print(table)

    # Overall metrics
    valid_rate = valid_json_count / len(samples) if samples else 0
    field_accuracy = correct_fields / total_fields if total_fields > 0 else 0

    console.print(f"\n[bold]Overall Metrics:[/]")
    console.print(f"  Valid JSON rate:    {valid_rate:.0%} ({valid_json_count}/{len(samples)})")
    console.print(f"  Field accuracy:     {field_accuracy:.0%} ({correct_fields}/{total_fields})")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference with fine-tuned Donut on receipt images"
    )
    parser.add_argument(
        "--image", type=str, default=None,
        help="Path to a single receipt image. If omitted, runs evaluation on the test split."
    )
    parser.add_argument(
        "--adapter_path", type=str, default=str(DEFAULT_ADAPTER_PATH),
        help=f"Path to the LoRA adapter directory (default: {DEFAULT_ADAPTER_PATH})"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    adapter_path = Path(args.adapter_path)

    # Verify the adapter exists
    if not adapter_path.exists():
        console.print(f"[red]ERROR: Adapter not found at {adapter_path}[/]")
        console.print("Run `python train.py` first to fine-tune the model.")
        return

    # Verify the processor exists
    if not PROCESSOR_DIR.exists():
        console.print(f"[red]ERROR: Processor not found at {PROCESSOR_DIR}[/]")
        console.print("Run `python prepare_data.py` first.")
        return

    # Load model
    model, processor = load_model(adapter_path)

    if args.image:
        # Single image mode
        run_single_image(model, processor, args.image)
    else:
        # Test split evaluation mode
        run_evaluation(model, processor)


if __name__ == "__main__":
    main()
