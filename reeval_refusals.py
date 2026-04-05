"""Re-evaluate refusal tags on existing eval_results.json using updated keywords."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from evaluate_advbench import (
    is_refusal,
    is_pinyin_refusal,
    plot_asr,
    SCRIPT_KEYS,
)


def main():
    results_dir = os.path.join(os.path.dirname(__file__),
                               "advbench_eval_20260405_062607")
    results_file = os.path.join(results_dir, "eval_results.json")

    with open(results_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} entries from {results_file}")

    for key, label in SCRIPT_KEYS:
        old_refused = sum(1 for e in data if e.get(f"refused_{key}", False))

        for entry in data:
            response = entry.get(f"response_{key}", "")
            refused = is_refusal(response)

            back_en = entry.get(f"response_{key}_en", "")
            if back_en and is_refusal(back_en):
                refused = True

            if key == "chinese_pinyin" and not refused:
                if is_pinyin_refusal(response):
                    refused = True

            entry[f"refused_{key}"] = refused

        new_refused = sum(1 for e in data if e.get(f"refused_{key}", False))
        print(f"  {label:25s}  refused: {old_refused} -> {new_refused}  "
              f"(delta {new_refused - old_refused:+d})")

    asr = {}
    print(f"\n{'='*60}")
    print("Updated Attack Success Rates")
    print(f"{'='*60}")
    for key, label in SCRIPT_KEYS:
        total = len(data)
        successes = sum(1 for e in data if not e.get(f"refused_{key}", True))
        rate = successes / total if total > 0 else 0.0
        asr[label] = rate
        print(f"  {label:25s}  {successes:3d}/{total}  ASR = {rate*100:.1f}%")

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\nUpdated results saved to {results_file}")

    summary_file = os.path.join(results_dir, "asr_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(asr, f, indent=2)
    print(f"ASR summary saved to {summary_file}")

    plot_path = os.path.join(results_dir, "asr_by_script.png")
    plot_asr(asr, plot_path, "Meta-Llama-3.1-8B-Instruct")

    print("\nDone.")


if __name__ == "__main__":
    main()
