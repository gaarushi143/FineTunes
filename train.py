"""
Fine-tune Donut (naver-clova-ix/donut-base) on the CORD v2 receipt dataset using LoRA.

=== What this script does ===

1. Loads the pre-trained Donut model (a vision encoder + text decoder)
2. Applies LoRA (Low-Rank Adaptation) — a technique that freezes the original model
   weights and injects small trainable matrices into the attention layers. This means
   we only train ~1-2M parameters instead of the full ~200M, so it fits on a free GPU.
3. Loads the CORD receipt images and their target token sequences (produced by prepare_data.py)
4. Trains the model to look at a receipt image and generate the correct token sequence
5. Saves the trained LoRA adapter (a small ~10-50MB file, not the full model)

=== Prerequisites ===

- Run `prepare_data.py` first to download CORD and generate the data/ directory
- GPU recommended (Google Colab free tier T4 works). CPU will be extremely slow.

=== Usage ===

    python train.py

    # Override defaults via command line:
    python train.py --epochs 5 --batch_size 1 --learning_rate 3e-5

    # Resume from a checkpoint:
    python train.py --resume_from checkpoints/checkpoint-500
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    DonutProcessor,
    VisionEncoderDecoderConfig,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType


# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

DATA_DIR = Path("data")
PROCESSOR_DIR = DATA_DIR / "processor"       # Updated processor from prepare_data.py
OUTPUT_DIR = Path("checkpoints")              # Where LoRA adapter checkpoints are saved
BASE_MODEL = "naver-clova-ix/donut-base"
TASK_PROMPT = "<s_cord-v2>"

# Training hyperparameters
DEFAULT_EPOCHS = 3
DEFAULT_BATCH_SIZE = 2                        # Small batch size to fit on T4 (16GB VRAM)
DEFAULT_GRAD_ACCUM_STEPS = 8                  # Effective batch size = 2 * 8 = 16
DEFAULT_LEARNING_RATE = 5e-5
MAX_LENGTH = 768                              # Max token length for decoder output


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class CORDDataset(Dataset):
    """
    PyTorch Dataset that loads receipt images and their target token sequences.

    Each item returned is a dict with:
      - pixel_values: the receipt image processed into a tensor (shape expected by Swin encoder)
      - labels: the target token sequence encoded as token IDs (what the decoder should output)

    The data comes from prepare_data.py's output:
      data/{split}/metadata.jsonl  — maps image filenames to ground truth token sequences
      data/{split}/images/         — the actual receipt PNG files
    """

    def __init__(self, split_name, processor, max_length=MAX_LENGTH):
        """
        Args:
            split_name: "train", "validation", or "test"
            processor:  DonutProcessor instance (handles both image preprocessing and tokenization)
            max_length: Maximum number of tokens for the target sequence. Sequences longer
                        than this are truncated. CORD receipts typically need 200-500 tokens.
        """
        self.processor = processor
        self.max_length = max_length
        self.split_dir = DATA_DIR / split_name

        # Load metadata — each line is {"file_name": "images/receipt_0001.png", "ground_truth": "<s_cord-v2>..."}
        metadata_path = self.split_dir / "metadata.jsonl"
        with open(metadata_path) as f:
            self.samples = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # --- Image processing ---
        # DonutProcessor's image_processor converts the PIL image into a tensor:
        #   1. Resizes to the model's expected input size (1920x2560 for donut-base)
        #   2. Normalizes pixel values
        #   3. Converts to a float tensor of shape (3, H, W)
        image_path = self.split_dir / sample["file_name"]
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()

        # --- Target token sequence ---
        # The ground truth is a string like "<s_cord-v2><s_menu><s_nm>Latte</s_nm>..."
        # We tokenize it into integer IDs that the BART decoder will learn to predict.
        # add_special_tokens=False because our sequence already starts with <s_cord-v2>
        target_sequence = sample["ground_truth"]
        labels = self.processor.tokenizer(
            target_sequence,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze()

        # Replace padding token IDs with -100 so the loss function ignores them.
        # PyTorch's CrossEntropyLoss skips positions where the label is -100,
        # so the model isn't penalized for what it predicts in padding positions.
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "labels": labels,
        }


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------

def load_model_and_processor():
    """
    Load the Donut base model and the updated processor (with CORD special tokens).

    Returns:
        model:     VisionEncoderDecoderModel with LoRA adapters injected
        processor: DonutProcessor with CORD-specific special tokens added

    The model architecture:
        Swin Transformer (encoder) → processes the receipt image into embeddings
        BART (decoder)             → generates the target token sequence from those embeddings

    We load the processor from data/processor/ (not from the HuggingFace hub) because
    prepare_data.py added special tokens like <s_menu>, </s_menu>, <s_nm>, etc. to it.
    """
    # Load the processor that prepare_data.py saved (it has our special tokens)
    processor = DonutProcessor.from_pretrained(PROCESSOR_DIR)

    # Load the base Donut model
    model = VisionEncoderDecoderModel.from_pretrained(BASE_MODEL)

    # --- Resize token embeddings ---
    # The base model's vocabulary doesn't include our special tokens (<s_menu>, etc.).
    # We need to expand the decoder's embedding layer so each new token gets its own
    # learnable vector. Initially these are random, but training will teach the model
    # what they mean.
    model.decoder.resize_token_embeddings(len(processor.tokenizer))

    # --- Configure the decoder for generation ---
    # These settings tell the model how to start and stop generating tokens:
    #   decoder_start_token_id: the first token fed to the decoder (<s_cord-v2>)
    #   pad_token_id:           used to fill sequences shorter than max_length
    #   eos_token_id:           when the model generates this, it stops
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(TASK_PROMPT)
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.eos_token_id

    # Max length the decoder can generate during inference
    model.config.max_length = MAX_LENGTH

    return model, processor


def apply_lora(model):
    """
    Wrap the model with LoRA (Low-Rank Adaptation) adapters.

    === What LoRA does ===

    Instead of updating all ~200M parameters during training, LoRA:
    1. Freezes every original weight in the model
    2. Adds a small pair of matrices (A and B) next to each target layer
    3. During forward pass: output = original_output + B @ A @ input
    4. Only A and B are trainable — typically ~1-2M parameters total

    === Why this matters ===

    - Memory: ~4x less VRAM than full fine-tuning (fits on a free T4 GPU)
    - Speed:  Fewer parameters to update = faster training
    - Storage: The saved adapter is ~10-50MB, not a full model copy
    - Safety: Original model knowledge is preserved; LoRA adds task-specific behavior on top

    === Configuration explained ===

    - r=8:             Rank of the A/B matrices. Higher = more capacity but more parameters.
                       8 is a good default for small datasets like CORD (800 examples).
    - lora_alpha=32:   Scaling factor. The LoRA output is multiplied by (alpha / r).
                       alpha=32 with r=8 means a 4x scaling — this controls how much
                       influence the LoRA weights have vs. the frozen base weights.
    - lora_dropout=0.05: Small dropout on LoRA layers to prevent overfitting.
    - target_modules:  Which layers get LoRA adapters. We target the attention layers
                       in the BART decoder (q_proj and v_proj = query and value projections).
                       These are where the model decides "what to attend to" — the most
                       impactful layers for learning new output patterns.
    """
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        task_type=TaskType.SEQ_2_SEQ_LM,
    )

    model = get_peft_model(model, lora_config)

    # Print a summary showing total params vs. trainable params
    model.print_trainable_parameters()
    # Expected output: something like
    #   "trainable params: 1,474,560 || all params: 201,234,432 || trainable%: 0.73%"

    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(model, processor, args):
    """
    Run the training loop using HuggingFace's Seq2SeqTrainer.

    === How training works ===

    For each batch of receipts:
    1. The Swin encoder processes each receipt image into visual embeddings
    2. The BART decoder tries to generate the target token sequence
    3. The loss measures how different the decoder's predictions are from the target
    4. Backpropagation updates ONLY the LoRA weights (everything else is frozen)
    5. Over time, the loss decreases as the model learns the receipt → token mapping

    === What to watch for ===

    - Training loss should decrease from ~2-3 to ~0.5-1.0 over 3 epochs
    - If loss plateaus early → training data may lack variety
    - If loss drops near 0 → overfitting (memorizing, not generalizing)
    - Each epoch takes ~10-20 min on a T4 GPU with 800 training examples
    """

    # Load train and validation datasets
    train_dataset = CORDDataset("train", processor)
    val_dataset = CORDDataset("validation", processor)
    print(f"\nDataset sizes — train: {len(train_dataset)}, validation: {len(val_dataset)}")

    # --- Training arguments ---
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_DIR),

        # How many times the model sees the full training set
        num_train_epochs=args.epochs,

        # Batch size per GPU. We keep this small (2) to fit in VRAM.
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,

        # Gradient accumulation simulates a larger batch size without using more VRAM.
        # With batch_size=2 and grad_accum=8, the effective batch size is 16.
        # The model accumulates gradients over 8 mini-batches before updating weights.
        gradient_accumulation_steps=DEFAULT_GRAD_ACCUM_STEPS,

        # Learning rate — how big each weight update step is.
        # 5e-5 is a standard starting point for fine-tuning transformer models.
        learning_rate=args.learning_rate,

        # Warmup: gradually increase learning rate from 0 to the target over the first
        # 100 steps. This prevents the model from making wild updates early on when
        # the LoRA weights are still random.
        warmup_steps=100,

        # FP16 mixed precision — uses 16-bit floats for forward/backward passes
        # instead of 32-bit. Halves memory usage with minimal accuracy loss.
        # Only works on GPU (ignored on CPU).
        fp16=torch.cuda.is_available(),

        # Logging and evaluation schedule
        logging_steps=25,                       # Print loss every 25 steps
        eval_strategy="steps",
        eval_steps=200,                         # Run validation every 200 steps
        save_strategy="steps",
        save_steps=200,                         # Save a checkpoint every 200 steps
        save_total_limit=3,                     # Keep only the 3 most recent checkpoints

        # Use the validation loss to pick the best checkpoint
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,                # Lower loss = better

        # Predict with generate — needed for seq2seq models so the Trainer
        # uses model.generate() during evaluation, not just forward()
        predict_with_generate=True,

        # Misc
        remove_unused_columns=False,            # Keep pixel_values (not a standard text column)
        report_to="none",                       # Disable wandb/tensorboard for simplicity
        dataloader_num_workers=2,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # --- Run training ---
    # If resuming from a checkpoint, pass the checkpoint path
    if args.resume_from:
        print(f"\nResuming training from: {args.resume_from}")
        trainer.train(resume_from_checkpoint=args.resume_from)
    else:
        print("\nStarting training...")
        trainer.train()

    # --- Save the final LoRA adapter ---
    # This saves ONLY the LoRA weights (the small A and B matrices), not the full model.
    # To use the model later, you load the base model + this adapter on top.
    adapter_dir = OUTPUT_DIR / "final_adapter"
    model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)
    print(f"\nLoRA adapter saved to: {adapter_dir}")
    print("Training complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Donut on CORD v2 with LoRA")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help=f"Number of training epochs (default: {DEFAULT_EPOCHS})")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Batch size per device (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_LEARNING_RATE,
                        help=f"Learning rate (default: {DEFAULT_LEARNING_RATE})")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to a checkpoint directory to resume training from")
    return parser.parse_args()


def main():
    args = parse_args()

    # Step 1: Verify that prepare_data.py has been run
    if not PROCESSOR_DIR.exists():
        print("ERROR: data/processor/ not found.")
        print("Run `python prepare_data.py` first to download CORD and prepare the data.")
        return

    # Step 2: Load model and apply LoRA
    print("Loading Donut base model...")
    model, processor = load_model_and_processor()
    print("Applying LoRA adapters...")
    model = apply_lora(model)

    # Step 3: Train
    train(model, processor, args)


if __name__ == "__main__":
    main()
