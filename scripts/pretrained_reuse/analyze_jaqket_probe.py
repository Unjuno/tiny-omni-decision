"""Summarize frozen JAQKET validation runs with answer-entity-group bootstrap."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np

CONDITIONS = {
    "ruri-pair": "ruri-pair-readout",
    "ruri-option-only": "ruri-option-only",
    "distil4-pair": "distil4-pair-readout",
    "distil4-option-only": "distil4-option-only",
    "bert-base-pair": "bert-base-pair-readout",
    "bert-base-option-only": "bert-base-option-only",
}


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip().casefold()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _per_example_metrics(probabilities: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    probabilities = np.clip(probabilities, 1e-12, 1.0)
    predictions = probabilities.argmax(axis=1)
    correct = predictions == targets
    rows = np.arange(len(targets))
    nll = -np.log(probabilities[rows, targets])
    one_hot = np.eye(20, dtype=np.float64)[targets]
    brier = np.square(probabilities - one_hot).sum(axis=1)
    confidence = probabilities.max(axis=1)
    ece = 0.0
    for bin_index in range(15):
        low, high = bin_index / 15, (bin_index + 1) / 15
        mask = (confidence > low) & (confidence <= high)
        if mask.any():
            ece += mask.mean() * abs(confidence[mask].mean() - correct[mask].mean())
    return {
        "accuracy": float(correct.mean()),
        "nll": float(nll.mean()),
        "brier": float(brier.mean()),
        "ece_15": float(ece),
    }


def analyze(
    validation_path: Path,
    artifact_dir: Path,
    *,
    prediction_filename: str = "validation-predictions.jsonl",
    split_name: str = "validation",
    repetitions: int = 10_000,
    seed: int = 1729,
) -> dict[str, Any]:
    rows = _read_jsonl(validation_path)
    targets = np.array(
        [
            next(
                index
                for index, option in enumerate(row["answer_candidates"])
                if _normalize(option) == _normalize(row["answer_entity"])
            )
            for row in rows
        ],
        dtype=np.int64,
    )
    groups_by_id: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        groups_by_id.setdefault(_normalize(row["answer_entity"]), []).append(index)
    groups = list(groups_by_id.values())
    predictions: dict[str, np.ndarray] = {}
    full_metrics: dict[str, dict[str, float]] = {}
    for condition, directory in CONDITIONS.items():
        prediction_rows = _read_jsonl(artifact_dir / directory / prediction_filename)
        if [row["qid"] for row in prediction_rows] != [row["qid"] for row in rows]:
            raise ValueError(f"validation prediction qid order mismatch for {condition}")
        probabilities = np.asarray(
            [row["probabilities"] for row in prediction_rows], dtype=np.float64
        )
        if probabilities.shape != (len(rows), 20) or not np.isfinite(probabilities).all():
            raise ValueError(f"invalid probability matrix for {condition}: {probabilities.shape}")
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
            raise ValueError(f"probabilities do not sum to one for {condition}")
        predictions[condition] = probabilities
        full_metrics[condition] = _per_example_metrics(probabilities, targets)

    comparisons = [
        ("bert-base-pair", "distil4-pair"),
        ("bert-base-pair", "ruri-pair"),
        ("bert-base-pair", "bert-base-option-only"),
        ("distil4-pair", "distil4-option-only"),
        ("ruri-pair", "ruri-option-only"),
    ]
    rng = np.random.default_rng(seed)
    deltas: dict[str, dict[str, list[float]]] = {
        f"{left}-minus-{right}": {metric: [] for metric in ("accuracy", "nll", "brier", "ece_15")}
        for left, right in comparisons
    }
    for _ in range(repetitions):
        sampled_groups = rng.integers(0, len(groups), size=len(groups))
        indices = np.concatenate([groups[index] for index in sampled_groups])
        sample_targets = targets[indices]
        for left, right in comparisons:
            key = f"{left}-minus-{right}"
            left_metrics = _per_example_metrics(predictions[left][indices], sample_targets)
            right_metrics = _per_example_metrics(predictions[right][indices], sample_targets)
            for metric in deltas[key]:
                deltas[key][metric].append(left_metrics[metric] - right_metrics[metric])

    intervals = {
        comparison: {
            metric: {
                "mean_delta": float(np.mean(values)),
                "percentile_95": [float(value) for value in np.quantile(values, [0.025, 0.975])],
            }
            for metric, values in by_metric.items()
        }
        for comparison, by_metric in deltas.items()
    }
    interpretation = (
        "same split used for epoch/model selection; intervals are optimistic"
        if split_name == "validation"
        else "public pseudo-test not used for selection; descriptive group-bootstrap intervals"
    )
    return {
        "split_name": split_name,
        f"{split_name}_rows": len(rows),
        "answer_entity_groups": len(groups),
        "grouped_bootstrap": {
            "group_field": "normalized answer_entity",
            "repetitions": repetitions,
            "seed": seed,
            "interpretation": interpretation,
        },
        "metrics": full_metrics,
        "paired_differences": intervals,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prediction-filename", default="validation-predictions.jsonl")
    parser.add_argument("--split-name", default="validation")
    parser.add_argument("--repetitions", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    result = analyze(
        args.validation,
        args.artifact_dir,
        prediction_filename=args.prediction_filename,
        split_name=args.split_name,
        repetitions=args.repetitions,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
