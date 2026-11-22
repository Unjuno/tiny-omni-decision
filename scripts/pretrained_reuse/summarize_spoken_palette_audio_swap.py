"""Compute multiclass probability metrics and speaker-group intervals for audio swap."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def decision_metrics(records: list[dict[str, Any]], *, bins: int = 15) -> dict[str, float | int]:
    if not records or bins < 1:
        raise ValueError("metrics require records and at least one confidence bin")
    probabilities = []
    targets = []
    predictions = []
    for row in records:
        values = row.get("probabilities_by_option_id")
        target = str(row.get("target_option_id", ""))
        predicted = str(row.get("predicted_option_id", ""))
        if not isinstance(values, dict) or len(values) != 4 or target not in values:
            raise ValueError("each prediction must contain four options including its target")
        probs = [float(value) for value in values.values()]
        if (
            not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in probs)
            or not math.isclose(sum(probs), 1.0, abs_tol=1e-5)
            or predicted not in values
        ):
            raise ValueError("prediction has an invalid probability distribution or option ID")
        probabilities.append(values)
        targets.append(target)
        predictions.append(predicted)

    confidence = np.asarray(
        [max(float(value) for value in row.values()) for row in probabilities], dtype=np.float64
    )
    correct = np.asarray(
        [prediction == target for prediction, target in zip(predictions, targets, strict=True)],
        dtype=np.float64,
    )
    nll = float(
        np.mean(
            [
                -math.log(max(float(values[target]), 1e-300))
                for values, target in zip(probabilities, targets, strict=True)
            ]
        )
    )
    brier = float(
        np.mean(
            [
                sum(
                    (float(value) - (1.0 if option_id == target else 0.0)) ** 2
                    for option_id, value in values.items()
                )
                for values, target in zip(probabilities, targets, strict=True)
            ]
        )
    )
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for index in range(bins):
        selected = (confidence >= edges[index]) & (
            (confidence < edges[index + 1])
            if index < bins - 1
            else (confidence <= edges[index + 1])
        )
        if np.any(selected):
            ece += float(np.mean(selected)) * abs(
                float(np.mean(correct[selected])) - float(np.mean(confidence[selected]))
            )
    return {
        "count": len(records),
        "accuracy": float(np.mean(correct)),
        "nll": nll,
        "brier_multiclass": brier,
        "ece_equal_width": ece,
        "ece_bins": bins,
        "mean_confidence": float(np.mean(confidence)),
    }


def summarize(
    records: list[dict[str, Any]], *, replicates: int = 10_000, seed: int = 17
) -> dict[str, Any]:
    if replicates < 100:
        raise ValueError("at least 100 bootstrap replicates are required")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[str(row.get("speaker_group", ""))].append(row)
    if len(groups) != 24 or any(len(rows) != 4 for rows in groups.values()):
        raise ValueError("expected four audio-swap samples in each of 24 speaker groups")
    point = decision_metrics(records)
    group_ids = sorted(groups)
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {
        name: [] for name in ("accuracy", "nll", "brier_multiclass", "ece_equal_width")
    }
    for _ in range(replicates):
        selected = rng.choice(group_ids, size=len(group_ids), replace=True).tolist()
        metrics = decision_metrics([row for group in selected for row in groups[group]])
        for name in draws:
            draws[name].append(float(metrics[name]))
    return {
        "method": "speaker-group bootstrap; 24 whole groups resampled with replacement",
        "seed": seed,
        "replicates": replicates,
        "point_estimates": point,
        "ci95_percentile": {
            name: [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
            for name, values in draws.items()
        },
        "limit": (
            "Fixed-checkpoint TRAIN-only intervention; not blind or "
            "out-of-speaker quality evidence."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(args.output.suffix + ".sha256").exists():
        raise FileExistsError("refusing to overwrite probability-metrics output")
    prediction_bytes = args.predictions.read_bytes()
    records = [
        json.loads(line)
        for line in prediction_bytes.decode("utf-8").splitlines()
        if line.strip()
    ]
    result = summarize(records, replicates=args.replicates, seed=args.seed)
    result["predictions_sha256"] = hashlib.sha256(prediction_bytes).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )
    with args.output.open("xb") as stream:
        stream.write(encoded)
    checksum_path = args.output.with_suffix(args.output.suffix + ".sha256")
    with checksum_path.open("x", encoding="ascii") as stream:
        stream.write(hashlib.sha256(encoded).hexdigest() + "\n")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
