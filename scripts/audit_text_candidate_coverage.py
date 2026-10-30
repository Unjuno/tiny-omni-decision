"""Read-only label and candidate-serialization coverage audit."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def audit(path):
    labels, abc, label_sets = Counter(), Counter(), Counter()
    questions = 0
    examples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        gold = json.loads(row["gold"])
        teacher = json.loads(row["teacher"])
        for key, definition in json.loads(row["questions"]).items():
            criteria = definition.get("criteria")
            if definition["type"] != "choice" or not isinstance(criteria, dict):
                continue
            questions += 1
            labels[str(gold[key])] += 1
            if set(criteria) == {"A", "B", "C"}:
                abc[str(gold[key])] += 1
                examples.append({"state_id": row["state_id"], "question_id": key})
            option_keys = list(teacher[key]["probabilities"])
            options = [f"{k}: {criteria[k]}" for k in option_keys]
            label_sets["bare_ABC_candidates"] += int(set(options) == {"A", "B", "C"})
            label_sets["ABC_all_available"] += int({"A", "B", "C"}.issubset(criteria))
    return {
        "choice_dict_questions": questions,
        "top_target_labels": labels.most_common(20),
        "ABC_exact_target_counts": dict(abc),
        "ABC_source_examples": examples,
        "candidate_coverage": dict(label_sets),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {split: audit(args.root / f"{split}.jsonl") for split in ("train", "validation")}
    result["scope"] = "Dictionary-valued choices only; no inference or label changes."
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
