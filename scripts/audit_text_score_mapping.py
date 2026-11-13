"""Audit ordinal source-to-prediction mapping without loading any model."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = [args.root / "text" / f"{s}.jsonl" for s in ("train", "validation")]
    prediction_path = args.root / "text-crossencoder-adapt-v1/predictions.json"
    predictions = {p["id"]: p for p in json.loads(prediction_path.read_text(encoding="utf-8"))}
    counts, confusion, errors, inspected = Counter(), Counter(), [], {}
    for split, path in zip(("train", "validation"), paths, strict=True):
        n = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            gold = json.loads(row["gold"])
            for key, definition in json.loads(row["questions"]).items():
                if definition["type"] != "score":
                    continue
                options, target = definition["criteria"], gold[key]
                if not isinstance(options, list) or type(target) is not int:
                    raise ValueError("unexpected ordinal representation")
                if not 0 <= target < len(options) or len(set(options)) != len(options):
                    raise ValueError("invalid/ambiguous ordinal labels")
                n += 1
                if split == "train":
                    continue
                p = predictions[f"n4ze3m/typed-decisions-synth:{row['state_id']}:{key}"]
                if p["options"] != options or p["target"] != options[target]:
                    raise ValueError("source prediction mapping mismatch")
                guess = max(range(len(options)), key=lambda i: p["probabilities"][i])
                distance = abs(guess - target)
                counts[distance] += 1
                confusion[(len(options), target, guess)] += 1
                errors.append(distance / (len(options) - 1))
        inspected[split] = n
    result = {
        "status": "PASS_MAPPING_ONLY",
        "checked_score_questions": inspected,
        "validation_distance_counts": dict(sorted(counts.items())),
        "validation_mean_absolute_index_error": sum(k * v for k, v in counts.items())
        / sum(counts.values()),
        "validation_mean_normalized_index_error": sum(errors) / len(errors),
        "confusion_by_scale": [
            {"options": n, "target": t, "prediction": p, "count": v}
            for (n, t, p), v in sorted(confusion.items())
        ],
        "hashes": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths + [prediction_path]
        },
        "limitations": [
            "Checks label transformation, not upstream label truth.",
            "Ordinal distance is descriptive; equal spacing is not guaranteed.",
            "No new inference, training or model selection.",
        ],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "confusion_by_scale"}, indent=2))


if __name__ == "__main__":
    main()
