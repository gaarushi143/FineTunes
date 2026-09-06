# AI Receipt Recognizer — Implementation Plan

## Goal
Fine-tune **Donut** (`naver-clova-ix/donut-base`) with LoRA to extract structured JSON from receipt images. Deploy as a Gradio web app.

```
[ Receipt Image ] → [ Fine-Tuned Donut (LoRA) ] → [ Structured JSON ] → [ Gradio UI ]
```

## Architecture Decisions
- **Base model:** `naver-clova-ix/donut-base` — purpose-built for document understanding
- **Dataset:** CORD (`naver-clova-ix/cord-v2`) — real receipt images with structured annotations
- **Fine-tuning method:** LoRA via PEFT on attention layers (`q_proj`, `v_proj`)
- **Quantization:** 4-bit via bitsandbytes (fits on free Colab T4 GPU)
- **Model hosting:** Hugging Face Hub (adapter only — keeps GitHub repo lightweight)

## Repo Structure
```
FineTunes/
├── .gitignore
├── requirements.txt
├── prepare_data.py       # Download CORD, convert to Donut-compatible format
├── train.py              # LoRA fine-tuning script
├── inference.py          # Load base + adapter, run on new receipts
├── app.py                # Gradio drag-and-drop UI
├── data/                 # Processed dataset configs (no raw images in git)
└── README.md
```

---

## Phase 1: Environment & Data Preparation (Days 1–2)

### 1.1 Repo setup
- Create `.gitignore` (model weights, checkpoints, Python caches, `.env`, `venv/`)
- Create `requirements.txt`:
  - `torch>=2.0`
  - `transformers>=4.40`
  - `peft>=0.10`
  - `datasets>=2.18`
  - `accelerate>=0.29`
  - `bitsandbytes>=0.43`
  - `sentencepiece`
  - `Pillow`
  - `gradio`

### 1.2 Data preparation (`prepare_data.py`)
- Load CORD v2 from Hugging Face: `load_dataset("naver-clova-ix/cord-v2")`
- CORD provides receipt images + JSON annotations with fields like company, date, total, line items
- Convert annotations into Donut's expected format: `<s_receipt>` token sequences
- Split: use CORD's built-in train/validation/test splits
- Save processed dataset to `data/` directory

### 1.3 Done when
- [ ] Repo initialized with `.gitignore` and `requirements.txt`
- [ ] `prepare_data.py` downloads CORD and converts to Donut format
- [ ] Can inspect sample image-text pairs

---

## Phase 2: Fine-Tuning (Days 3–4)

### 2.1 Training script (`train.py`)
- Load `naver-clova-ix/donut-base` with 4-bit quantization (`BitsAndBytesConfig`)
- Apply LoRA via PEFT: target attention weight matrices (`q_proj`, `v_proj`)
  - `r=8`, `lora_alpha=32`, `lora_dropout=0.05`
- Use Hugging Face `Trainer` with:
  - Learning rate: `5e-5`
  - Epochs: `3`
  - Batch size: `2` (gradient accumulation to simulate larger batches)
  - FP16 mixed precision
- Log training loss per step

### 2.2 Execution
- Run on Google Colab (free T4 GPU) or local GPU
- Monitor loss — should decrease from ~2-3 to ~0.5-1.0
- Save LoRA adapter checkpoints to `checkpoints/`

### 2.3 Done when
- [ ] Training completes without OOM errors
- [ ] Loss curve shows convergence
- [ ] LoRA adapter saved locally

---

## Phase 3: Evaluation & Model Storage (Day 5)

### 3.1 Inference script (`inference.py`)
- Load base Donut model
- Overlay trained LoRA adapter via `PeftModel.from_pretrained()`
- Run on held-out test split images
- Print extracted JSON and compare against ground truth

### 3.2 Metrics to track
| Metric          | What it measures                                    |
|-----------------|-----------------------------------------------------|
| Valid JSON rate  | % of outputs that parse as valid JSON               |
| Field accuracy   | Correct vendor, date, total extraction              |
| Tree Edit Distance | Structural similarity to ground truth (Donut metric) |

### 3.3 Push to Hugging Face Hub
- `huggingface-cli login`
- `model.push_to_hub("your-username/receipt-recognizer-lora")`
- Only the adapter (~10-50 MB) gets uploaded, not the full model

### 3.4 Done when
- [ ] Inference runs on test images with reasonable accuracy
- [ ] Adapter pushed to Hugging Face Hub
- [ ] Metrics documented

---

## Phase 4: Gradio App & Deployment (Days 6–7)

### 4.1 Web interface (`app.py`)
- Gradio app with:
  - **Input:** Drag-and-drop image uploader
  - **Processing:** Load base model + LoRA adapter, run inference
  - **Output:** Formatted JSON display with extracted fields
- Optional: highlight reimbursable vs. non-reimbursable items

### 4.2 README & portfolio
- Write `README.md` with:
  - Project description and architecture diagram
  - Setup and usage instructions
  - Training metrics and sample results (side-by-side: receipt image → extracted JSON)
  - Link to Hugging Face model

### 4.3 Done when
- [ ] Gradio app runs locally — upload receipt, get JSON back
- [ ] README is complete with metrics and screenshots
- [ ] All scripts pushed to GitHub
