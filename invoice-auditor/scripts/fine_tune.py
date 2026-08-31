"""
Launch a fine-tuning job on Google Gemini.

Usage:
    python3 scripts/fine_tune.py

Requires GEMINI_API_KEY environment variable to be set.
Get a free key at: https://aistudio.google.com/apikey
"""

import json
import os
import sys
from google import genai

def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: Set GEMINI_API_KEY environment variable first.")
        print("Get a free key at: https://aistudio.google.com/apikey")
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    train_path = os.path.join(data_dir, "train_gemini.json")

    if not os.path.exists(train_path):
        print("ERROR: data/train_gemini.json not found.")
        print("Run: python3 scripts/convert_to_gemini.py")
        sys.exit(1)

    # Load training data and convert to Gemini types
    with open(train_path) as f:
        raw_data = json.load(f)

    training_dataset = genai.types.TuningDataset(
        examples=[
            genai.types.TuningExample(
                text_input=ex["text_input"],
                output=ex["output"],
            )
            for ex in raw_data
        ]
    )

    print(f"Loaded {len(raw_data)} training examples")

    # Launch fine-tuning job
    # Using gemini-2.0-flash-lite (supports tuning on free tier)
    base_model = "models/gemini-1.5-flash-001-tuning"
    print(f"\nLaunching fine-tuning on {base_model}...")
    print("(This typically takes 15-30 minutes)\n")

    tuning_job = client.tunings.tune(
        base_model=base_model,
        training_dataset=training_dataset,
        config=genai.types.CreateTuningJobConfig(
            epoch_count=3,
            learning_rate=0.001,
            tuned_model_display_name="invoice-auditor",
        ),
    )

    print(f"  Job Name:   {tuning_job.name}")
    print(f"  Status:     {tuning_job.state}")
    if tuning_job.tuned_model:
        print(f"  Model:      {tuning_job.tuned_model.model}")

    print(f"\nSave this job name: {tuning_job.name}")
    print("Run monitor.py to check progress:")
    print(f'  python3 scripts/monitor.py "{tuning_job.name}"')


if __name__ == "__main__":
    main()
