"""Metrics for fixed binary Perception visibility heads."""

from __future__ import annotations

import numpy as np


def score_binary_logits(
    logits: np.ndarray, targets: np.ndarray, *, bins: int = 15
) -> dict[str, object]:
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    targets = np.asarray(targets).reshape(-1)
    if logits.size == 0 or targets.size != logits.size:
        raise ValueError("logits and targets must be matching nonempty vectors")
    if not np.isfinite(logits).all():
        raise ValueError("finite logits are required")
    if not np.isin(targets, (0, 1)).all():
        raise ValueError("binary targets are required")
    if type(bins) is not int or bins < 1:
        raise ValueError("bins must be a positive integer")
    targets = targets.astype(np.int64, copy=False)
    predictions = (logits > 0).astype(np.int64)
    probability_visible = 1.0 / (1.0 + np.exp(-np.clip(logits, -709.0, 709.0)))
    confidence = np.maximum(probability_visible, 1.0 - probability_visible)
    correct = predictions == targets
    nll = np.logaddexp(0.0, logits) - targets * logits
    brier = np.square(probability_visible - targets)
    ece = 0.0
    calibration_bins = []
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        mask = (confidence >= lower) & (
            confidence < upper if index < bins - 1 else confidence <= upper
        )
        count = int(mask.sum())
        if not count:
            calibration_bins.append({"count": 0, "mean_confidence": None, "accuracy": None})
            continue
        mean_confidence = float(confidence[mask].mean())
        accuracy = float(correct[mask].mean())
        ece += count / len(targets) * abs(mean_confidence - accuracy)
        calibration_bins.append(
            {"count": count, "mean_confidence": mean_confidence, "accuracy": accuracy}
        )
    return {
        "count": int(len(targets)),
        "accuracy": float(correct.mean()),
        "nll": float(nll.mean()),
        "brier": float(brier.mean()),
        "ece_15_equal_width": float(ece),
        "mean_confidence": float(confidence.mean()),
        "positive_rate": float(targets.mean()),
        "predicted_positive_rate": float(predictions.mean()),
        "calibration_bins": calibration_bins,
        "probability_visible": probability_visible.tolist(),
        "predictions": predictions.tolist(),
    }


def paired_bootstrap_accuracy_delta(
    correct_a: np.ndarray,
    correct_b: np.ndarray,
    *,
    seed: int = 17,
    repetitions: int = 2000,
) -> dict[str, object]:
    a = np.asarray(correct_a, dtype=np.float64).reshape(-1)
    b = np.asarray(correct_b, dtype=np.float64).reshape(-1)
    if (
        a.size == 0
        or b.size != a.size
        or not np.isin(a, (0, 1)).all()
        or not np.isin(b, (0, 1)).all()
    ):
        raise ValueError("paired bootstrap needs matching binary correctness vectors")
    if type(seed) is not int or seed < 0 or type(repetitions) is not int or repetitions < 1:
        raise ValueError("seed and repetitions must be positive integers")
    rng = np.random.default_rng(seed)
    deltas = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        sample = rng.integers(0, len(a), size=len(a))
        deltas[index] = np.mean(b[sample] - a[sample])
    observed = float(np.mean(b - a))
    low, high = np.quantile(deltas, [0.025, 0.975])
    return {
        "delta_b_minus_a": observed,
        "bootstrap_repetitions": repetitions,
        "seed": seed,
        "interval_method": "paired example bootstrap, percentile 95% interval",
        "interval_95": [float(low), float(high)],
    }
