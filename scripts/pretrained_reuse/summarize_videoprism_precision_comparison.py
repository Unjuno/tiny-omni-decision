"""Summarize a paired VideoPrism precision replay without selecting a model."""

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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError(f"prediction file is empty: {path}")
    return rows


def _metrics(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    probabilities = np.asarray([row["probabilities"] for row in rows], dtype=np.float64)
    if (probabilities.ndim != 2 or not np.isfinite(probabilities).all()
            or (probabilities < 0).any()
            or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5, rtol=0)):
        raise ValueError("predictions must be finite normalized distributions")
    target_indices = np.asarray(
        [row["options"].index(row["target"]) for row in rows], dtype=np.int64
    )
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predictions == target_indices
    one_hot = np.eye(probabilities.shape[1], dtype=np.float64)[target_indices]
    ece = 0.0
    for index in range(15):
        lower, upper = index / 15, (index + 1) / 15
        in_bin = (confidence >= lower) & (
            (confidence < upper) if index < 14 else (confidence <= upper)
        )
        if in_bin.any():
            ece += float(in_bin.mean()) * abs(
                float(correct[in_bin].mean()) - float(confidence[in_bin].mean())
            )
    return {
        "count": len(rows),
        "accuracy": float(correct.mean()),
        "nll": float(
            -np.log(
                np.maximum(probabilities[np.arange(len(rows)), target_indices], 1e-300)
            ).mean()
        ),
        "brier": float(((probabilities - one_hot) ** 2).sum(axis=1).mean()),
        "ece_15_equal_width": ece,
        "mean_confidence": float(confidence.mean()),
    }


def paired_summary(
    baseline: list[dict[str, Any]], candidate: list[dict[str, Any]]
) -> dict[str, Any]:
    if len(baseline) != len(candidate):
        raise ValueError("baseline and candidate prediction counts differ")
    def key(row: dict[str, Any]) -> tuple[str, str]:
        return row["image_id"], row["taxonomy"]

    baseline_by_key = {key(row): row for row in baseline}
    candidate_by_key = {key(row): row for row in candidate}
    if len(baseline_by_key) != len(baseline) or len(candidate_by_key) != len(candidate):
        raise ValueError("duplicate image/taxonomy prediction key")
    if baseline_by_key.keys() != candidate_by_key.keys():
        raise ValueError("baseline and candidate image/taxonomy keys differ")
    for sample_key in baseline_by_key:
        left, right = baseline_by_key[sample_key], candidate_by_key[sample_key]
        if left["target"] != right["target"] or left["options"] != right["options"]:
            raise ValueError(f"target/options changed for {sample_key}")

    baseline_metrics = _metrics(baseline)
    candidate_metrics = _metrics(candidate)
    baseline_taxonomies = sorted({row["taxonomy"] for row in baseline})
    candidate_taxonomies = sorted({row["taxonomy"] for row in candidate})
    if baseline_taxonomies != candidate_taxonomies:
        raise ValueError("taxonomy sets differ")
    per_taxonomy = {}
    for taxonomy in baseline_taxonomies:
        left = [row for row in baseline if row["taxonomy"] == taxonomy]
        right = [row for row in candidate if row["taxonomy"] == taxonomy]
        per_taxonomy[taxonomy] = {
            "baseline": _metrics(left),
            "candidate": _metrics(right),
        }

    probability_delta = []
    argmax_changes = 0
    target_correct_changes = 0
    for sample_key in baseline_by_key:
        left, right = baseline_by_key[sample_key], candidate_by_key[sample_key]
        left_probability = np.asarray(left["probabilities"], dtype=np.float64)
        right_probability = np.asarray(right["probabilities"], dtype=np.float64)
        if left_probability.shape != right_probability.shape:
            raise ValueError(f"option count changed for {sample_key}")
        left_argmax, right_argmax = int(left_probability.argmax()), int(right_probability.argmax())
        argmax_changes += int(left_argmax != right_argmax)
        target_index = left["options"].index(left["target"])
        changed_correctness = (right_argmax == target_index) != (left_argmax == target_index)
        target_correct_changes += int(changed_correctness)
        probability_delta.extend(np.abs(left_probability - right_probability).tolist())

    return {
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "macro_taxonomy_accuracy_baseline": float(np.mean([
            per_taxonomy[taxonomy]["baseline"]["accuracy"] for taxonomy in baseline_taxonomies
        ])),
        "macro_taxonomy_accuracy_candidate": float(np.mean([
            per_taxonomy[taxonomy]["candidate"]["accuracy"] for taxonomy in baseline_taxonomies
        ])),
        "per_taxonomy": per_taxonomy,
        "argmax_changed_decisions": argmax_changes,
        "correctness_changed_decisions": target_correct_changes,
        "max_probability_absolute_delta": float(max(probability_delta)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--candidate-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite comparison summary: {args.output}")
    baseline_path = args.baseline_predictions.resolve(strict=True)
    candidate_path = args.candidate_predictions.resolve(strict=True)
    result = {
        "status": "paired_development_precision_comparison",
        "scope": (
            "exposed CLEVR-4 checkpoint-selection validation; "
            "not independent quality evidence"
        ),
        "baseline_sha256": sha256_file(baseline_path),
        "candidate_sha256": sha256_file(candidate_path),
        "metrics": paired_summary(read_jsonl(baseline_path), read_jsonl(candidate_path)),
        "model_selection_or_training_performed": False,
        "sealed_audit_loaded": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
