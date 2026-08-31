# Fine-Tuning Learning Plan: Invoice & Receipt Auditor (OpenAI API)

## Context
You know RAG. Now you want to learn **fine-tuning** — training a model on examples so behavior is baked in (consistent output format, domain extraction patterns, reliable structured output without elaborate prompts). This uses OpenAI's fine-tuning API: simple, fast, ~$1-3 total cost.

**Fine-tuning vs. prompting vs. RAG:**
- **Prompting** = instructions at inference time ("please output JSON")
- **RAG** = injecting external knowledge at query time (you know this)
- **Fine-tuning** = changing the model's weights via training examples — the behavior becomes default

---

## Phase 1: Setup & Concepts (25 min)

### Step 1 — Install dependencies
```bash
mkdir -p invoice-auditor/{data,scripts}
cd invoice-auditor
pip install openai rich
export OPENAI_API_KEY="your-key-here"
```

### Step 2 — Learn the core concepts

**What happens when you fine-tune:**
1. You upload training examples (conversations with ideal outputs)
2. OpenAI runs additional training on their base model using your examples
3. You get back a custom model ID that behaves like your examples
4. At inference, it outputs in your trained format without elaborate prompts

**Key terms:**
- **Epochs**: How many times the model sees each training example (default: ~3)
- **Training loss**: How wrong the model is — should decrease over training
- **Overfitting**: Model memorizes examples instead of learning the pattern
- **JSONL format**: One JSON object per line — the training data format

**When fine-tuning helps most:**
- Consistent structured output (your use case — always the same JSON schema)
- Domain-specific terminology and conventions
- Reducing prompt length (behavior is baked in, not instructed)
- Tone/style matching

**When fine-tuning does NOT help:**
- Adding knowledge the model doesn't have (that's RAG)
- Tasks a good prompt already handles well
- Small datasets with little variety

### Done with Phase 1 when:
- [x] Dependencies installed
- [x] You can explain what fine-tuning does differently from prompting and RAG

---

## Phase 2: Build Training Data (35 min)

This is the most important phase. Good training data matters more than any hyperparameter.

### Step 1 — Understand the data format

Each training example is a conversation in JSONL format:
```json
{"messages": [{"role": "system", "content": "You are an expense receipt auditor. Extract structured data from raw receipts and flag policy violations. Respond only with valid JSON."}, {"role": "user", "content": "UBER EATS 08/15/2026 $47.82\nPad Thai x2, Spring Rolls\nTip $8.00\nSubmitted by: John D., Marketing\nCategory: Team Lunch"}, {"role": "assistant", "content": "{\"vendor\":\"Uber Eats\",\"date\":\"2026-08-15\",\"total\":47.82,\"items\":[{\"name\":\"Pad Thai x2\",\"amount\":null},{\"name\":\"Spring Rolls\",\"amount\":null},{\"name\":\"Tip\",\"amount\":8.00}],\"submitter\":\"John D.\",\"department\":\"Marketing\",\"category\":\"Team Lunch\",\"flags\":[{\"rule\":\"TIP_OVER_20_PCT\",\"severity\":\"warning\",\"detail\":\"Tip is ~20.1% of subtotal, exceeds 20% guideline\"}],\"verdict\":\"REVIEW\"}"}]}
```

### Step 2 — Write `scripts/generate_training_data.py`

This script programmatically generates ~50 training examples with variety across:

**Receipt types** (~8-10 each):
- Restaurant / food delivery (clean and messy OCR)
- Taxi / rideshare
- Hotel / lodging
- Office supplies
- Software subscriptions

**Violation types:**
- Over daily meal limit ($75)
- Alcohol present
- Excessive tip (>20%)
- Personal items mixed in
- Missing itemization
- Weekend/off-hours expense

**Receipt messiness levels:**
```
# Clean
"Hilton Downtown - Room 412, Check-in 08/20, Check-out 08/22, $289/night, Total $578.00"

# OCR-garbled
"H1LTON D0WNTOWN Rm412 chkin 8/20 chkout 8/22 $289/nt Tot: $578.O0"

# Minimal
"uber $23.50 airport run 8/14"

# Verbose
"Amazon.com Order #112-3456789 Ship date Aug 14 2026 Qty 1 Logitech MX Master 3S Mouse $89.99..."
```

**Verdict distribution:** ~40% APPROVED, ~35% REVIEW, ~25% REJECTED

### Step 3 — Generate and split the data

Run the script to produce:
- `data/train.jsonl` — 50 training examples
- `data/test.jsonl` — 8-10 held-out examples (NOT used for training)

### Step 4 — Validate the data

Write `scripts/validate_data.py` that checks:
- Every line parses as valid JSON
- Every example has system/user/assistant messages
- Every assistant response is valid JSON
- Reasonable variety in verdicts and violation types

### Done with Phase 2 when:
- [ ] `generate_training_data.py` runs and produces `train.jsonl` + `test.jsonl`
- [ ] `validate_data.py` passes all checks
- [ ] You have ~50 training examples with good variety across receipt types, violations, and messiness levels

---

## Phase 3: Fine-Tune the Model (30 min)

### Step 1 — Write `scripts/fine_tune.py`

```python
from openai import OpenAI
client = OpenAI()

# Upload training file
train_file = client.files.create(
    file=open("data/train.jsonl", "rb"),
    purpose="fine-tune"
)
print(f"File ID: {train_file.id}")

# Launch fine-tuning job
job = client.fine_tuning.jobs.create(
    training_file=train_file.id,
    model="gpt-4o-mini-2024-07-18",
    hyperparameters={"n_epochs": 3}
)
print(f"Job ID: {job.id}")
print(f"Status: {job.status}")
```

### Step 2 — Write `scripts/monitor.py`

```python
from openai import OpenAI
client = OpenAI()

job_id = "YOUR_JOB_ID"  # paste from fine_tune.py output

job = client.fine_tuning.jobs.retrieve(job_id)
print(f"Status: {job.status}")
print(f"Fine-tuned model: {job.fine_tuned_model}")

# Show training events (loss curve)
events = client.fine_tuning.jobs.list_events(fine_tuning_job_id=job_id, limit=20)
for event in events.data:
    print(event.message)
```

### Step 3 — Run fine-tuning and monitor

Run `fine_tune.py`, then periodically run `monitor.py` (~15-25 min wait).

**What to watch:**
- Training loss should decrease from ~2-3 down to ~0.5-1.0
- If it plateaus early → data may lack variety
- If it drops to near 0 → overfitting (memorizing, not learning)
- Status changes: `validating_files` → `queued` → `running` → `succeeded`

### Done with Phase 3 when:
- [ ] Job status is `succeeded`
- [ ] You have a fine-tuned model ID (like `ft:gpt-4o-mini-2024-07-18:personal::XXXXX`)
- [ ] Training loss decreased during training

---

## Phase 4: Evaluate & Compare (30 min)

### Step 1 — Write `scripts/evaluate.py`

The script sends each test receipt to BOTH the base model and your fine-tuned model, then compares:

```python
# Fine-tuned model — minimal prompt
ft_response = client.chat.completions.create(
    model="ft:gpt-4o-mini-2024-07-18:personal::XXXXX",
    messages=[
        {"role": "system", "content": "You are an expense receipt auditor."},
        {"role": "user", "content": test_receipt}
    ]
)

# Base model — needs detailed prompting
base_response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are an expense receipt auditor. Extract structured data from raw receipts and flag policy violations. Respond only with valid JSON matching this schema: {vendor, date, total, items[], submitter, department, category, flags[], verdict}"},
        {"role": "user", "content": test_receipt}
    ]
)
```

### Step 2 — Run evaluation and score

For each test receipt, compare base vs. fine-tuned:

| Metric | What to check |
|--------|--------------|
| **Valid JSON** | Does output parse? Fine-tuned should be ~100% |
| **Schema match** | Same field names every time? Base model often varies |
| **Field extraction** | Correct vendor, date, amounts? |
| **Flag accuracy** | Right violations caught? False positives? |
| **Prompt length** | Fine-tuned needs minimal prompting vs. base |

### Step 3 — Print a clear summary

Use `rich` to print a table showing pass/fail per metric per test case, plus an overall score.

### Done with Phase 4 when:
- [ ] Evaluation runs on all test cases
- [ ] You have a side-by-side comparison showing where fine-tuning improved things
- [ ] Fine-tuned model produces valid JSON on all test cases

---

## Phase 5: Experiment & Reflect (20 min)

### Step 1 — Try edge cases

- Send a receipt type NOT in your training data — how does each model handle it?
- Send a deliberately ambiguous receipt — is fine-tuning more or less cautious?
- Try the fine-tuned model with NO system prompt — does it still output the right format? (This is the "baked in" test)

### Step 2 — Reflect

Write down your answers:
- Where did fine-tuning help most? (usually: format consistency, shorter prompts)
- Where did it NOT help? (usually: novel reasoning, edge cases unlike training data)
- What would more/better training data improve?
- When would you fine-tune vs. prompt-engineer vs. RAG?

### Done with Phase 5 when:
- [ ] You tested at least 2-3 edge cases
- [ ] You can articulate when to use fine-tuning vs. prompting vs. RAG

---

## Project Structure
```
invoice-auditor/
├── data/
│   ├── train.jsonl                # ~50 training examples
│   └── test.jsonl                 # 8-10 held-out evaluation examples
├── scripts/
│   ├── generate_training_data.py  # Creates synthetic training data
│   ├── validate_data.py           # Checks JSONL format and variety
│   ├── fine_tune.py               # Upload + launch fine-tuning job
│   ├── monitor.py                 # Check job status + training events
│   └── evaluate.py                # Compare base vs fine-tuned
└── requirements.txt               # openai, rich
```
