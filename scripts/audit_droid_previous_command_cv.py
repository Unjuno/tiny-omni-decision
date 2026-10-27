"""Train-only cross-fold transition-table baseline on fixed DROID episodes."""

import argparse
import json
import runpy
from pathlib import Path

import numpy as np

from scripts.train_source_unimodal_cv import digest

ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")
HISTORY = ROOT / "droid-train-temporal-baselines-20261027-attempt-01/train-predictions.jsonl"
CV = ROOT / "droid-train-temporal-cv-20261027-attempt-01"
PREDICTIONS = CV / "outer-train-predictions.jsonl"
RESULT = CV / "result.json"
READOUT = CV / "pinned-readout.py"
PINNED = {
    HISTORY: "6cc30b04d1d10d880a157885102848ac882a88ad8580bd17fb9411744cb309ae",
    PREDICTIONS: "f020b2e3bcd0eb36ccc4117b2946a152cd98b5db27b9efc0e954a21214d7d6aa",
    RESULT: "8b44d08afe358f5df2c37bca93b2a14998ede472acc8d031c2d2e1d11eb12867",
    READOUT: "186fad84f223748bd4670903c7b85e61aa6513eed8e311065bf58d17176d7728",
}
PROTOCOL = Path("docs/pretrained_reuse/R2_DROID_PREVIOUS_COMMAND_CV_PROTOCOL_20261027.md")
REPO = Path(__file__).resolve().parents[1]


def transition_oof(previous, targets, folds):
    previous = np.asarray(previous, dtype=np.int64)
    targets = np.asarray(targets, dtype=np.int64)
    if (previous.ndim != 1 or targets.shape != previous.shape or len(targets) == 0
            or np.any((previous < 0) | (previous >= 7))
            or np.any((targets < 0) | (targets >= 7))):
        raise ValueError("invalid seven-class rows")
    n = len(targets)
    probs = np.zeros((n, 7), dtype=np.float64)
    visits = np.zeros(n, dtype=np.int64)
    all_indices = set(range(n))
    for fold in folds:
        train, held = fold["train_indices"], fold["held_indices"]
        if (len(train) != len(set(train)) or len(held) != len(set(held))
                or set(train) & set(held) or set(train) | set(held) != all_indices):
            raise ValueError("invalid train/held fold partition")
        counts = np.ones((7, 7), dtype=np.float64)
        for idx in train:
            counts[previous[idx], targets[idx]] += 1
        counts /= counts.sum(axis=1, keepdims=True)
        probs[held] = counts[previous[held]]
        visits[held] += 1
    if not np.all(visits == 1):
        raise ValueError("every example must be held out exactly once")
    return probs


def validate_rows(history, predictions):
    if len(history) != 275 or len(predictions) != 275:
        raise ValueError("unexpected train row count")
    if len({row["group_id"] for row in history}) != len(history):
        raise ValueError("train recording groups must be unique")
    for first, second in zip(history, predictions, strict=True):
        if any(first[key] != second[key] for key in ("id", "group_id", "target")):
            raise ValueError("row order or identity mismatch")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() or output.is_relative_to(REPO):
        raise ValueError("new external output directory required")
    if any(digest(path) != value for path, value in PINNED.items()):
        raise ValueError("pinned input mismatch")
    history = [json.loads(line) for line in HISTORY.read_text().splitlines()]
    predictions = [json.loads(line) for line in PREDICTIONS.read_text().splitlines()]
    validate_rows(history, predictions)
    original = json.loads(RESULT.read_bytes())
    folds = original["folds"]
    if len(folds) != 5:
        raise ValueError("unexpected fold count")
    previous = np.array([row["predictions"]["previous_command"] for row in history])
    targets = np.array([row["target"] for row in history])
    transition = transition_oof(previous, targets, folds)
    measure = runpy.run_path(str(READOUT))["multiclass_probability_metrics"]
    probabilities = {name: np.array([row["probabilities"][name] for row in predictions])
                     for name in ("raw", "delta")}
    probabilities["previous_command_transition"] = transition
    metrics = {name: measure(targets, value) for name, value in probabilities.items()}
    subsets = {}
    for name, mask in (("stable", previous == targets), ("changed", previous != targets)):
        subsets[name] = {"count": int(mask.sum()), "metrics": {
            arm: measure(targets[mask], value[mask]) for arm, value in probabilities.items()}}
    result = {
        "status": "TRAIN_ONLY_ADDITIONAL_INPUT_DIAGNOSTIC", "rows": len(targets),
        "fold_sizes": [len(fold["held_indices"]) for fold in folds],
        "previous_command_is_extra_input": True, "add_one_pseudocount": 1,
        "metrics": metrics, "target_defined_subsets": subsets,
        "source_hashes": {str(path): value for path, value in PINNED.items()},
        "protocol_sha256": digest(PROTOCOL), "script_sha256": digest(Path(__file__)),
        "limitations": "nonblind train cohort; group folds do not prove operator independence",
    }
    output.mkdir(parents=True)
    data = [{"id": row["id"], "group_id": row["group_id"], "target": int(targets[i]),
             "previous_command": int(previous[i]), "probabilities": transition[i].tolist()}
            for i, row in enumerate(history)]
    (output / "private-predictions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in data), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
