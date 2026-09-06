"""
Gradio web app for the receipt recognizer.

=== What this does ===

Provides a drag-and-drop web interface where you can upload a receipt image
and get back structured JSON with the extracted fields (menu items, totals, etc.).

Under the hood, it loads the fine-tuned Donut model (base + LoRA adapter) and
runs the same inference pipeline as inference.py.

=== Usage ===

    python app.py

    # Custom adapter path:
    python app.py --adapter_path checkpoints/checkpoint-400

    # Make it accessible on your local network:
    python app.py --share

Then open the URL printed in the terminal (typically http://localhost:7860).
"""

import argparse
import json
from pathlib import Path

import gradio as gr
from PIL import Image

from inference import load_model, predict, PROCESSOR_DIR, DEFAULT_ADAPTER_PATH


# ---------------------------------------------------------------------------
# Globals — loaded once at startup, reused for every request
# ---------------------------------------------------------------------------

model = None
processor = None


def process_receipt(image):
    """
    Gradio callback: takes an uploaded image, runs inference, returns formatted JSON.

    Gradio passes the uploaded image as a PIL Image (or numpy array depending on
    the input type). We convert it to RGB, run the Donut model, and return the
    result as a pretty-printed JSON string.

    Args:
        image: PIL Image from the Gradio upload component

    Returns:
        A formatted JSON string with the extracted receipt fields, or an error message
    """
    if image is None:
        return "No image uploaded."

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    image = image.convert("RGB")

    result = predict(model, processor, image)
    return json.dumps(result, indent=2, ensure_ascii=False)


def build_app():
    """
    Build the Gradio interface.

    Layout:
      ┌─────────────────────────────────────────────┐
      │           Receipt Recognizer                 │
      │  Upload a receipt to extract structured data │
      ├──────────────────┬──────────────────────────┤
      │                  │                          │
      │  [Image Upload]  │  [JSON Output]           │
      │                  │                          │
      └──────────────────┴──────────────────────────┘
    """
    demo = gr.Interface(
        fn=process_receipt,
        inputs=gr.Image(
            type="pil",
            label="Upload Receipt",
        ),
        outputs=gr.Code(
            language="json",
            label="Extracted Data",
        ),
        title="Receipt Recognizer",
        description=(
            "Upload a receipt image to extract structured data using a "
            "fine-tuned Donut model. The model outputs menu items, subtotals, "
            "and totals as JSON."
        ),
        examples=[
            # If test images exist, use a few as clickable examples
            str(path) for path in sorted(
                (Path("data/test/images")).glob("*.png")
            )[:3]
        ] if (Path("data/test/images")).exists() else None,
        flagging_mode="never",
    )
    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Gradio app for receipt recognition")
    parser.add_argument(
        "--adapter_path", type=str, default=str(DEFAULT_ADAPTER_PATH),
        help=f"Path to the LoRA adapter directory (default: {DEFAULT_ADAPTER_PATH})"
    )
    parser.add_argument(
        "--share", action="store_true",
        help="Create a public Gradio link (useful for Colab or remote servers)"
    )
    parser.add_argument(
        "--port", type=int, default=7860,
        help="Port to run the server on (default: 7860)"
    )
    return parser.parse_args()


def main():
    global model, processor

    args = parse_args()
    adapter_path = Path(args.adapter_path)

    if not adapter_path.exists():
        print(f"ERROR: Adapter not found at {adapter_path}")
        print("Run `python train.py` first to fine-tune the model.")
        return

    if not PROCESSOR_DIR.exists():
        print(f"ERROR: Processor not found at {PROCESSOR_DIR}")
        print("Run `python prepare_data.py` first.")
        return

    # Load model once at startup — this takes ~10-30 seconds
    print("Loading model (this may take a moment)...")
    model, processor = load_model(adapter_path)
    print("Model loaded. Starting Gradio server...\n")

    demo = build_app()
    demo.launch(
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
