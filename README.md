# Receipt Recognizer — Fine-Tuned Donut for Structured Receipt Extraction

A hands-on project that fine-tunes [Donut](https://huggingface.co/naver-clova-ix/donut-base) (Document Understanding Transformer) to extract structured JSON from receipt images — no OCR required. Built with LoRA for parameter-efficient training on a free GPU.

```
[ Receipt Image ] → [ Swin Encoder ] → [ BART Decoder ] → [ Structured JSON ]
```

## What This Project Covers

| Topic | What You'll Learn |
|-------|-------------------|
| **Vision-Language Models** | How Donut processes document images end-to-end without OCR |
| **LoRA / PEFT** | Parameter-efficient fine-tuning — training ~1% of model weights |
| **Training Pipeline** | Data prep, tokenization, training loop, checkpointing |
| **Evaluation** | Measuring JSON validity, field-level accuracy, spotting overfitting |
| **Deployment** | Serving a model through a Gradio web interface |

---

## How Fine-Tuning Works

### The Core Idea

A pre-trained model like Donut already understands visual features (shapes, text, layout) from training on millions of documents. Fine-tuning teaches it your **specific task** — in our case, extracting receipt fields into a particular JSON schema.

Think of it like hiring someone who already speaks the language fluently, then training them on your company's specific forms and processes.

### Prompting vs. RAG vs. Fine-Tuning

| Approach | What It Does | Best For |
|----------|-------------|----------|
| **Prompting** | Instructions at inference time ("output JSON with these fields") | Quick experiments, flexible tasks |
| **RAG** | Injects external knowledge at query time | Adding facts the model doesn't know |
| **Fine-Tuning** | Changes the model's weights via training examples | Consistent output format, domain-specific behavior, shorter prompts |

Fine-tuning is the right choice here because we need the model to **always** output the same JSON schema, handle messy receipt layouts reliably, and do it without lengthy system prompts.

### How Donut Learns from Receipts

Donut is an **OCR-free** document understanding model. Unlike traditional pipelines that extract text first (OCR) then process it, Donut goes directly from pixels to structured output:

1. **Swin Transformer (encoder)** — processes the receipt image at high resolution (1920x2560 pixels). It learns to identify character shapes, spatial layout, and visual patterns directly from pixels — effectively learning to read as part of learning the task.

2. **BART (decoder)** — generates a sequence of special tokens that represent the structured output. Each JSON key becomes a token pair: `<s_menu>`, `</s_menu>`, `<s_nm>`, `</s_nm>`, etc.

During training, the model sees 800 receipt images paired with their correct token sequences. Over multiple passes (epochs), it learns: "when I see this visual pattern in this position on a receipt, generate these specific tokens."

### Why LoRA Instead of Full Fine-Tuning

Donut has ~200 million parameters. Full fine-tuning would:
- Require expensive GPUs (32GB+ VRAM)
- Risk overfitting on 800 examples
- Produce a full model copy (~800MB) for storage

**LoRA** (Low-Rank Adaptation) solves this by:
- **Freezing** all original weights
- **Injecting** small trainable matrices (A and B) into attention layers
- Training only ~1.5M parameters (~0.7% of the model)
- Producing a tiny adapter file (~10-50MB)

```
Original layer:  output = W @ input
With LoRA:       output = W @ input + B @ A @ input
                          ↑ frozen     ↑ trainable (small)
```

We target `q_proj` and `v_proj` (query and value projections in attention) — these are where the model decides "what to attend to" and are the most impactful layers for learning new output patterns.

---

## Architecture

### Dataset: CORD v2

[CORD](https://huggingface.co/datasets/naver-clova-ix/cord-v2) (Consolidated Receipt Dataset) contains 1,000 real receipt images with structured JSON annotations:

| Split | Count | Purpose |
|-------|-------|---------|
| Train | 800 | Model learns from these |
| Validation | 100 | Monitors for overfitting during training |
| Test | 100 | Final evaluation (model never sees these during training) |

Each receipt is annotated with:
```json
{
  "menu": [
    {"nm": "Pad Thai", "cnt": "1", "price": "12.00"},
    {"nm": "Spring Rolls", "cnt": "2", "price": "8.00"}
  ],
  "sub_total": {"subtotal_price": "20.00", "tax_price": "1.60"},
  "total": {"total_price": "21.60", "cashprice": "25.00", "changeprice": "3.40"}
}
```

### Training Configuration

| Parameter | Value | Why |
|-----------|-------|-----|
| Base model | `naver-clova-ix/donut-base` | Purpose-built for documents |
| LoRA rank (r) | 8 | Good capacity without overfitting on 800 examples |
| LoRA alpha | 32 | Scaling factor (alpha/r = 4x influence) |
| Learning rate | 5e-5 | Standard for transformer fine-tuning |
| Batch size | 2 (effective 16 via gradient accumulation) | Fits T4 GPU VRAM |
| Epochs | 3 | Enough passes to learn patterns without memorizing |
| Precision | FP16 mixed | Halves memory with minimal accuracy loss |

---

## Project Structure

```
FineTunes/
├── prepare_data.py    # Download CORD, convert to Donut token format
├── train.py           # LoRA fine-tuning with HuggingFace Trainer
├── inference.py       # Evaluate on test set or run on single images
├── app.py             # Gradio web UI for drag-and-drop receipt processing
├── requirements.txt   # Python dependencies
├── plan.md            # Implementation plan
├── data/              # Processed dataset (generated by prepare_data.py)
│   ├── train/         # 800 training receipts + metadata
│   ├── validation/    # 100 validation receipts + metadata
│   ├── test/          # 100 test receipts + metadata
│   ├── processor/     # Tokenizer with CORD special tokens
│   └── special_tokens.json
└── checkpoints/       # Saved LoRA adapters (generated by train.py)
```

---

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare the dataset

Downloads CORD v2 from Hugging Face and converts annotations to Donut's token format:

```bash
python prepare_data.py
```

### 3. Train the model

Fine-tunes Donut with LoRA on the CORD training set. Requires a GPU (Google Colab free tier works):

```bash
python train.py
```

Override defaults if needed:

```bash
python train.py --epochs 5 --batch_size 1 --learning_rate 3e-5
```

**What to watch during training:**
- Training loss should decrease from ~2-3 to ~0.5-1.0
- If loss plateaus early — data may lack variety
- If loss drops near 0 — the model is overfitting (memorizing, not learning)

### 4. Evaluate

Run on the held-out test set to measure accuracy:

```bash
python inference.py
```

Or test on a single image:

```bash
python inference.py --image path/to/receipt.png
```

### 5. Launch the web app

```bash
python app.py
```

Open `http://localhost:7860`, drag a receipt image in, and see the extracted JSON.

Use `--share` to get a public link (useful on Colab):

```bash
python app.py --share
```

---

## How to Improve Results

### More / Better Training Data
The single biggest lever. 800 examples is a solid start, but more diverse receipts (different languages, layouts, handwritten amounts, poor lighting) will improve generalization. Options:
- Add the [SROIE dataset](https://huggingface.co/datasets/davanstrien/sroie) for more variety
- Collect your own receipts and annotate with [Label Studio](https://labelstud.io/)
- Apply data augmentation (rotation, brightness, noise) to simulate real-world conditions

### Hyperparameter Tuning
- **LoRA rank**: Increase from 8 to 16 or 32 for more model capacity (watch for overfitting)
- **Learning rate**: Try 1e-5 to 1e-4 range with a cosine schedule
- **Epochs**: More epochs with early stopping based on validation loss
- **Target modules**: Add `k_proj` and `out_proj` to LoRA targets for broader adaptation

### Model Upgrades
- **Donut-large**: Larger encoder for better visual understanding (needs more VRAM)
- **PaLiGemma 3B**: More powerful general-purpose VLM, better on diverse receipt formats
- **Quantization**: Add 4-bit quantization (QLoRA) to train larger models on the same hardware

### Post-Processing
- Add JSON schema validation on outputs
- Implement fallback logic for malformed outputs (re-prompt or regex extraction)
- Normalize extracted values (dates, currency formats)

### Evaluation
- Add Tree Edit Distance (TED) metric for structural similarity
- Test on receipts from different countries/languages
- Measure inference latency and optimize (batch processing, model distillation)

---

## Key Concepts Reference

| Term | Definition |
|------|-----------|
| **LoRA** | Low-Rank Adaptation — injects small trainable matrices into frozen model layers |
| **PEFT** | Parameter-Efficient Fine-Tuning — umbrella term for methods like LoRA |
| **Epoch** | One full pass through all training examples |
| **Training loss** | How wrong the model's predictions are — should decrease over training |
| **Overfitting** | Model memorizes training examples instead of learning general patterns |
| **FP16** | 16-bit floating point — halves memory usage with minimal accuracy loss |
| **Gradient accumulation** | Simulates a larger batch size by accumulating gradients over multiple mini-batches |
| **Autoregressive generation** | Generating tokens one at a time, each conditioned on all previous tokens |

---

## Tech Stack

- **[PyTorch](https://pytorch.org/)** — deep learning framework
- **[Hugging Face Transformers](https://huggingface.co/docs/transformers)** — model loading, training, inference
- **[PEFT](https://huggingface.co/docs/peft)** — LoRA adapter injection
- **[Datasets](https://huggingface.co/docs/datasets)** — CORD v2 dataset loading
- **[Gradio](https://gradio.app/)** — web UI for the demo
- **[Rich](https://rich.readthedocs.io/)** — terminal output formatting for evaluation
