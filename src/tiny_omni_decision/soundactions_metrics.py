"""Metrics and paired recording bootstrap for the SoundActions pilot."""

from __future__ import annotations

import numpy as np


def multiclass_metrics(
    probabilities: np.ndarray, targets: np.ndarray, *, class_count: int = 3, ece_bins: int = 15
) -> dict[str, object]:
    probs = np.asarray(probabilities, dtype=np.float64)
    truth = np.asarray(targets, dtype=np.int64)
    if probs.ndim != 2 or probs.shape != (len(truth), class_count) or not len(truth):
        raise ValueError("probability and target dimensions do not agree")
    if ece_bins < 1 or not np.isfinite(probs).all() or np.any(probs < 0):
        raise ValueError("probabilities or ECE bin count are invalid")
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("class probabilities must sum to one")
    if np.any((truth < 0) | (truth >= class_count)):
        raise ValueError("target class is outside the class range")

    predictions = probs.argmax(axis=1)
    correct = predictions == truth
    confidence = probs.max(axis=1)
    bin_ids = np.minimum((confidence * ece_bins).astype(np.int64), ece_bins - 1)
    ece = 0.0
    for bin_id in range(ece_bins):
        selected = bin_ids == bin_id
        if selected.any():
            ece += float(selected.mean()) * abs(
                float(correct[selected].mean()) - float(confidence[selected].mean())
            )
    per_class = {}
    for class_id in range(class_count):
        selected = truth == class_id
        per_class[str(class_id)] = {
            "count": int(selected.sum()),
            "accuracy": float(correct[selected].mean()) if selected.any() else None,
        }
    return {
        "count": len(truth),
        "accuracy": float(correct.mean()),
        "macro_accuracy": float(
            np.mean([row["accuracy"] for row in per_class.values() if row["accuracy"] is not None])
        ),
        "nll": float(-np.log(np.clip(probs[np.arange(len(truth)), truth], 1e-12, 1.0)).mean()),
        "brier": float(np.mean(np.sum((probs - np.eye(class_count)[truth]) ** 2, axis=1))),
        "ece": ece,
        "mean_confidence": float(confidence.mean()),
        "per_class": per_class,
    }


def paired_bootstrap_difference(
    probabilities_a: np.ndarray,
    probabilities_b: np.ndarray,
    targets: np.ndarray,
    *,
    metric: str,
    seed: int = 17,
    iterations: int = 10_000,
) -> dict[str, object]:
    """Bootstrap paired recording differences; positive is always better."""
    if metric not in {"accuracy", "macro_accuracy", "nll", "brier", "ece"}:
        raise ValueError(f"unsupported paired metric: {metric}")
    first = np.asarray(probabilities_a)
    second = np.asarray(probabilities_b)
    truth = np.asarray(targets)
    if first.shape != second.shape or first.ndim != 2 or len(first) != len(truth):
        raise ValueError("paired bootstrap inputs must align by recording")
    if iterations < 1 or not len(truth):
        raise ValueError("bootstrap needs at least one iteration and recording")
    multiclass_metrics(first, truth)
    multiclass_metrics(second, truth)
    rng = np.random.default_rng(seed)
    class_indices = [np.flatnonzero(truth == class_id) for class_id in np.unique(truth)]
    if any(not len(indices) for indices in class_indices):
        raise ValueError("bootstrap target strata must contain at least one recording")
    samples = np.concatenate(
        [
            rng.choice(indices, size=(iterations, len(indices)), replace=True)
            for indices in class_indices
        ],
        axis=1,
    )

    def bootstrap_scores(probabilities: np.ndarray) -> np.ndarray:
        sampled_probabilities = probabilities[samples]
        sampled_targets = truth[samples]
        predictions = sampled_probabilities.argmax(axis=2)
        if metric == "accuracy":
            return np.mean(predictions == sampled_targets, axis=1)
        if metric == "macro_accuracy":
            offsets = np.cumsum([0, *(len(indices) for indices in class_indices)])
            per_class_accuracy = [
                np.mean(predictions[:, offsets[i] : offsets[i + 1]] == class_id, axis=1)
                for i, class_id in enumerate(np.unique(truth))
            ]
            return np.mean(np.stack(per_class_accuracy), axis=0)
        if metric == "nll":
            chosen = np.take_along_axis(sampled_probabilities, sampled_targets[:, :, None], axis=2)[
                :, :, 0
            ]
            return -np.log(np.clip(chosen, 1e-12, 1.0)).mean(axis=1)
        if metric == "brier":
            one_hot = np.eye(sampled_probabilities.shape[2])[sampled_targets]
            return np.sum((sampled_probabilities - one_hot) ** 2, axis=2).mean(axis=1)
        # ECE is nonlinear in confidence bins; use the same fixed paired draws.
        return np.asarray(
            [
                multiclass_metrics(prob_row, target_row)[metric]
                for prob_row, target_row in zip(sampled_probabilities, sampled_targets, strict=True)
            ],
            dtype=np.float64,
        )

    score_a = bootstrap_scores(first)
    score_b = bootstrap_scores(second)
    direction = 1.0 if metric in {"accuracy", "macro_accuracy"} else -1.0
    differences = direction * (score_b - score_a)
    return {
        "metric": metric,
        "direction": "positive means candidate_b is better",
        "seed": seed,
        "iterations": iterations,
        "recordings": len(truth),
        "resampling": "paired, stratified by target class",
        "mean_improvement": float(differences.mean()),
        "percentile_95_ci": [
            float(np.quantile(differences, 0.025)),
            float(np.quantile(differences, 0.975)),
        ],
    }
