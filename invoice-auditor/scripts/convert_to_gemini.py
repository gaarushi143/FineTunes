"""
Convert our OpenAI-format JSONL training data to Gemini's fine-tuning format.

Gemini expects a list of training examples where each example has:
  - text_input: the user message
  - output: the assistant response

The system prompt is prepended to the user message.
"""

import json
import os


def convert(input_path, output_path):
    examples = []
    with open(input_path) as f:
        for line in f:
            data = json.loads(line)
            messages = data["messages"]

            system = messages[0]["content"]
            user = messages[1]["content"]
            assistant = messages[2]["content"]

            examples.append({
                "text_input": f"{system}\n\n---\n\n{user}",
                "output": assistant,
            })

    with open(output_path, "w") as f:
        json.dump(examples, f, indent=2)

    return len(examples)


def main():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")

    train_count = convert(
        os.path.join(data_dir, "train.jsonl"),
        os.path.join(data_dir, "train_gemini.json"),
    )
    test_count = convert(
        os.path.join(data_dir, "test.jsonl"),
        os.path.join(data_dir, "test_gemini.json"),
    )

    print(f"Converted {train_count} training examples -> data/train_gemini.json")
    print(f"Converted {test_count} test examples -> data/test_gemini.json")


if __name__ == "__main__":
    main()
