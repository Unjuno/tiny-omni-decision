"""Compute uploader-cluster bootstrap intervals for saved FSD50K predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

LABELS = ("Animal", "Human_voice", "Music", "Vehicle")
ECE_BINS = 15


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_prediction_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _group_sufficient_statistics(
    rows: list[dict[str, Any]],
) -> tuple[np.ndarray, list[str]]:
    if not rows:
        raise ValueError("prediction rows must not be empty")
    groups = sorted({str(row.get("group_id", "")) for row in rows})
    if not groups or groups[0] == "":
        raise ValueError("every prediction must have a nonempty group_id")
    group_index = {group_id: index for index, group_id in enumerate(groups)}
    # Per label: n, correct, positive support, TP, TN, NLL sum, Brier sum,
    # then (count, correct, confidence sum) for each fixed ECE bin.
    columns_per_label = 7 + ECE_BINS * 3
    stats = np.zeros((len(groups), len(LABELS) * columns_per_label), dtype=np.float64)
    seen: set[tuple[str, str]] = set()

    for row in rows:
        label = row.get("label")
        group_id = str(row.get("group_id", ""))
        sample_id = str(row.get("sample_id", ""))
        if label not in LABELS or not sample_id or group_id == "":
            raise ValueError("predictions need valid label, sample_id, and group_id")
        key = (sample_id, label)
        if key in seen:
            raise ValueError(f"duplicate prediction for sample/label: {key}")
        seen.add(key)

        target = row.get("target")
        if target not in (0, 1):
            raise ValueError("binary target must be 0 or 1")
        probabilities = row.get("option_probabilities", {})
        p_positive = float(probabilities.get("annotated", float("nan")))
        p_negative = float(probabilities.get("no_annotation", float("nan")))
        if (
            not np.isfinite(p_positive)
            or not np.isfinite(p_negative)
            or p_positive < 0.0
            or p_positive > 1.0
            or p_negative < 0.0
            or p_negative > 1.0
            or not np.isclose(p_positive + p_negative, 1.0, atol=1e-5)
        ):
            raise ValueError("option probabilities must be finite and sum to one")

        label_index = LABELS.index(label)
        offset = label_index * columns_per_label
        predicted = int(p_positive >= p_negative)
        correct = int(predicted == target)
        confidence = max(p_positive, p_negative)
        bin_index = min(int(confidence * ECE_BINS), ECE_BINS - 1)
        values = stats[group_index[group_id]]
        values[offset] += 1.0
        values[offset + 1] += correct
        values[offset + 2] += target
        values[offset + 3] += int(predicted == 1 and target == 1)
        values[offset + 4] += int(predicted == 0 and target == 0)
        values[offset + 5] += -np.log(max(p_positive if target else p_negative, 1e-15))
        values[offset + 6] += 2.0 * (p_positive - target) ** 2
        bin_offset = offset + 7 + bin_index * 3
        values[bin_offset] += 1.0
        values[bin_offset + 1] += correct
        values[bin_offset + 2] += confidence

    for sample_id in {str(row["sample_id"]) for row in rows}:
        observed = {label for row_id, label in seen if row_id == sample_id}
        if observed != set(LABELS):
            raise ValueError(f"sample {sample_id} does not have exactly all four labels")
    return stats, groups


def _metrics_from_sufficient_statistics(values: np.ndarray) -> dict[str, dict[str, float]]:
    columns_per_label = 7 + ECE_BINS * 3
    result: dict[str, dict[str, float]] = {}
    for label_index, label in enumerate(LABELS):
        offset = label_index * columns_per_label
        count, correct, positives, true_positive, true_negative, nll, brier = values[
            offset : offset + 7
        ]
        if count <= 0:
            raise ValueError("bootstrap replicate omitted all examples for a label")
        negative_support = count - positives
        ece = 0.0
        for bin_index in range(ECE_BINS):
            bin_offset = offset + 7 + bin_index * 3
            bin_count, bin_correct, confidence_sum = values[bin_offset : bin_offset + 3]
            if bin_count:
                ece += (bin_count / count) * abs(
                    (bin_correct / bin_count) - (confidence_sum / bin_count)
                )
        positive_recall = true_positive / positives if positives else float("nan")
        specificity = true_negative / negative_support if negative_support else float("nan")
        result[label] = {
            "accuracy": correct / count,
            "nll": nll / count,
            "brier": brier / count,
            "ece_15": ece,
            "balanced_accuracy": (positive_recall + specificity) / 2.0,
            "positive_recall": positive_recall,
            "count": count,
            "positive_support": positives,
        }
    return result


def _macro(per_label: dict[str, dict[str, float]]) -> dict[str, float]:
    return {
        metric: float(np.mean([per_label[label][metric] for label in LABELS]))
        for metric in (
            "accuracy",
            "nll",
            "brier",
            "ece_15",
            "balanced_accuracy",
            "positive_recall",
        )
    }


def _intervals(samples: np.ndarray) -> dict[str, float | int]:
    finite = samples[np.isfinite(samples)]
    if not len(finite):
        return {"valid_replicates": 0, "low": float("nan"), "high": float("nan")}
    low, high = np.quantile(finite, [0.025, 0.975])
    return {
        "valid_replicates": int(len(finite)),
        "low": float(low),
        "high": float(high),
    }


def bootstrap_intervals(
    rows: list[dict[str, Any]], *, replicates: int = 10_000, seed: int = 17
) -> dict[str, Any]:
    if replicates < 1:
        raise ValueError("replicates must be positive")
    group_stats, group_ids = _group_sufficient_statistics(rows)
    point_per_label = _metrics_from_sufficient_statistics(group_stats.sum(axis=0))
    point_macro = _macro(point_per_label)

    rng = np.random.default_rng(seed)
    weights = rng.multinomial(
        len(group_ids), np.full(len(group_ids), 1.0 / len(group_ids)), size=replicates
    )
    sampled = weights @ group_stats
    bootstrap_per_label = [_metrics_from_sufficient_statistics(row) for row in sampled]
    bootstrap_macro = [_macro(row) for row in bootstrap_per_label]

    metrics = (
        "accuracy",
        "nll",
        "brier",
        "ece_15",
        "balanced_accuracy",
        "positive_recall",
    )
    macro_intervals = {
        metric: _intervals(np.asarray([row[metric] for row in bootstrap_macro]))
        for metric in metrics
    }
    label_intervals = {
        label: {
            metric: _intervals(
                np.asarray([row[label][metric] for row in bootstrap_per_label])
            )
            for metric in metrics
        }
        for label in LABELS
    }
    return {
        "status": "PASS",
        "method": "paired nonparametric cluster bootstrap over uploader group_id",
        "seed": seed,
        "replicates": replicates,
        "independent_groups": len(group_ids),
        "validation_decisions": len(rows),
        "interpretation": (
            "post-selection validation interval; descriptive, not independent confirmation"
        ),
        "macro_point_estimates": point_macro,
        "macro_percentile_95_ci": macro_intervals,
        "per_label_point_estimates": point_per_label,
        "per_label_percentile_95_ci": label_intervals,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    result = bootstrap_intervals(
        load_prediction_rows(args.predictions), replicates=args.replicates, seed=args.seed
    )
    result["input_predictions_sha256"] = sha256_file(args.predictions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "result": result}, sort_keys=True))


if __name__ == "__main__":
    main()
