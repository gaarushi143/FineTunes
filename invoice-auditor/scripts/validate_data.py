"""
Validate the training and test JSONL files before uploading to OpenAI.

Checks:
- Every line is valid JSON
- Every example has system/user/assistant messages
- Every assistant response is valid JSON with the expected schema
- Reasonable variety in verdicts and categories
"""

import json
import os
import sys

EXPECTED_FIELDS = {"vendor", "date", "total", "items", "submitter", "department", "category", "flags", "verdict"}
VALID_VERDICTS = {"APPROVED", "REVIEW", "REJECTED"}


def validate_file(path):
    errors = []
    stats = {"verdicts": {}, "categories": {}, "total": 0, "flag_types": {}}

    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            # Check 1: Valid JSON line
            try:
                example = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"Line {i}: Invalid JSON — {e}")
                continue

            # Check 2: Has messages array
            if "messages" not in example:
                errors.append(f"Line {i}: Missing 'messages' key")
                continue

            messages = example["messages"]
            roles = [m.get("role") for m in messages]

            if roles != ["system", "user", "assistant"]:
                errors.append(f"Line {i}: Expected roles [system, user, assistant], got {roles}")
                continue

            # Check 3: Assistant response is valid JSON
            assistant_content = messages[2]["content"]
            try:
                audit = json.loads(assistant_content)
            except json.JSONDecodeError as e:
                errors.append(f"Line {i}: Assistant response is not valid JSON — {e}")
                continue

            # Check 4: Expected schema fields
            missing = EXPECTED_FIELDS - set(audit.keys())
            if missing:
                errors.append(f"Line {i}: Missing fields in audit output: {missing}")

            # Check 5: Valid verdict
            verdict = audit.get("verdict")
            if verdict not in VALID_VERDICTS:
                errors.append(f"Line {i}: Invalid verdict '{verdict}'")

            # Track stats
            stats["total"] += 1
            stats["verdicts"][verdict] = stats["verdicts"].get(verdict, 0) + 1
            cat = audit.get("category", "Unknown")
            stats["categories"][cat] = stats["categories"].get(cat, 0) + 1
            for flag in audit.get("flags", []):
                rule = flag.get("rule", "unknown")
                stats["flag_types"][rule] = stats["flag_types"].get(rule, 0) + 1

    return errors, stats


def main():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    all_passed = True

    for filename in ["train.jsonl", "test.jsonl"]:
        path = os.path.join(data_dir, filename)
        if not os.path.exists(path):
            print(f"MISSING: {filename}")
            all_passed = False
            continue

        print(f"\n{'='*50}")
        print(f"Validating: {filename}")
        print(f"{'='*50}")

        errors, stats = validate_file(path)

        if errors:
            print(f"\nFAILED — {len(errors)} error(s):")
            for e in errors:
                print(f"  {e}")
            all_passed = False
        else:
            print(f"PASSED — {stats['total']} examples, all valid")

        print(f"\n  Verdicts:    {dict(sorted(stats['verdicts'].items()))}")
        print(f"  Categories:  {dict(sorted(stats['categories'].items()))}")
        print(f"  Flag types:  {dict(sorted(stats['flag_types'].items()))}")

        # Warn if low variety
        if filename == "train.jsonl":
            if stats["total"] < 40:
                print(f"\n  WARNING: Only {stats['total']} examples — recommend at least 50")
            if len(stats["verdicts"]) < 3:
                print(f"\n  WARNING: Only {len(stats['verdicts'])} verdict types — should have all 3")
            if len(stats["categories"]) < 3:
                print(f"\n  WARNING: Only {len(stats['categories'])} categories — add more variety")

    print(f"\n{'='*50}")
    if all_passed:
        print("ALL CHECKS PASSED — data is ready for fine-tuning!")
    else:
        print("SOME CHECKS FAILED — fix errors above before fine-tuning.")
    print(f"{'='*50}")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
