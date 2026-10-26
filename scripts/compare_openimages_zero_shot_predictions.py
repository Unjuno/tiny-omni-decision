"""Create a paired, target-stratified comparison from saved Open Images predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metrics(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    if not rows:
        raise ValueError("metric groups cannot be empty")
    probabilities = np.asarray([row["probabilities"] for row in rows], dtype=np.float64)
    if probabilities.shape != (len(rows), 4) or not np.isfinite(probabilities).all():
        raise ValueError("expected finite four-choice probabilities")
    if np.any(probabilities < 0) or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("choice probabilities must be nonnegative and sum to one")
    targets = np.asarray([row["options"].index(row["target"]) for row in rows])
    predictions = probabilities.argmax(axis=1)
    correct = predictions == targets
    confidence = probabilities.max(axis=1)
    ece = 0.0
    for bin_index in range(15):
        low = bin_index / 15
        high = (bin_index + 1) / 15
        in_bin = (
            (confidence > low) & (confidence <= high)
            if bin_index
            else ((confidence >= low) & (confidence <= high))
        )
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return {
        "count": len(rows),
        "accuracy": float(correct.mean()),
        "nll": float(
            -np.log(np.maximum(1e-15, probabilities[np.arange(len(rows)), targets])).mean()
        ),
        "brier": float(((probabilities - np.eye(4)[targets]) ** 2).sum(axis=1).mean()),
        "ece_15_bins": float(ece),
    }


def compare_rows(
    pecore: list[dict[str, Any]],
    tinyclip: list[dict[str, Any]],
    task_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    def index(rows, *, condition=None):
        filtered = [row for row in rows if condition is None or row.get("condition") == condition]
        mapping = {str(row["sample_id"]): row for row in filtered}
        if len(mapping) != len(filtered):
            raise ValueError("prediction sample IDs must be unique")
        return mapping

    left = index(pecore, condition="aligned")
    right = index(tinyclip)
    tasks = {str(row["sample_id"]): row for row in task_rows}
    if len(tasks) != len(task_rows):
        raise ValueError("task sample IDs must be unique")
    if len(left) != 168 or set(left) != set(right) or not set(left).issubset(tasks):
        raise ValueError("models and fixed task manifest must have identical 168 IDs")
    left_rows, right_rows = [], []
    per_target: dict[str, dict[str, Any]] = {}
    for sample_id in sorted(left):
        a, b, task = left[sample_id], right[sample_id], tasks[sample_id]
        for key in ("options", "target"):
            if a.get(key) != b.get(key) or a.get(key) != task.get(key):
                raise ValueError(f"model predictions do not share frozen {key} for {sample_id}")
        for row in (a, b):
            _metrics([row])
            expected = row["options"][int(np.argmax(row["probabilities"]))]
            if row.get("prediction") != expected:
                raise ValueError(f"stored prediction disagrees with probabilities for {sample_id}")
        left_rows.append(a)
        right_rows.append(b)
    label_to_name = {str(row["target_label"]): str(row["target"]) for row in task_rows}
    for target_label, name in sorted(label_to_name.items(), key=lambda item: item[1]):
        group_ids = [
            sample_id
            for sample_id in sorted(left)
            if tasks[sample_id]["target_label"] == target_label
        ]
        per_target[name] = {
            "PE-Core": _metrics([left[sample_id] for sample_id in group_ids]),
            "TinyCLIP": _metrics([right[sample_id] for sample_id in group_ids]),
        }
    left_correct = [row["prediction"] == row["target"] for row in left_rows]
    right_correct = [row["prediction"] == row["target"] for row in right_rows]
    return {
        "count": len(left_rows),
        "PE-Core": _metrics(left_rows),
        "TinyCLIP": _metrics(right_rows),
        "paired": {
            "same_top1_prediction": sum(
                a["prediction"] == b["prediction"]
                for a, b in zip(left_rows, right_rows, strict=True)
            ),
            "same_correctness": sum(
                a == b for a, b in zip(left_correct, right_correct, strict=True)
            ),
            "PE_Core_correct_TinyCLIP_wrong": sum(
                a and not b for a, b in zip(left_correct, right_correct, strict=True)
            ),
            "TinyCLIP_correct_PE_Core_wrong": sum(
                b and not a for a, b in zip(left_correct, right_correct, strict=True)
            ),
        },
        "per_target": per_target,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pecore-predictions", type=Path, required=True)
    parser.add_argument("--tinyclip-predictions", type=Path, required=True)
    parser.add_argument("--tasks-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    def read(path: Path) -> list[dict[str, Any]]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    task_rows = json.loads(args.tasks_manifest.read_text(encoding="utf-8"))["examples"]
    result = compare_rows(read(args.pecore_predictions), read(args.tinyclip_predictions), task_rows)
    result["input_sha256"] = {
        "PE-Core_predictions": sha256_file(args.pecore_predictions),
        "TinyCLIP_predictions": sha256_file(args.tinyclip_predictions),
        "task_manifest": sha256_file(args.tasks_manifest),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
