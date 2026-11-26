"""Task-specific metrics for grouped IntPhys anomaly-score diagnostics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def binary_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Compute ROC AUC with average ranks for tied anomaly scores."""
    y = np.asarray(labels, dtype=np.int8)
    values = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or values.shape != y.shape or len(y) == 0:
        raise ValueError("labels and scores must be equal nonempty one-dimensional arrays")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(values).all():
        raise ValueError("labels must be binary and scores finite")
    positives = int(y.sum())
    negatives = len(y) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("ROC AUC requires both labels")
    order = np.argsort(values, kind="mergesort")
    sorted_scores = values[order]
    ranks = np.empty(len(y), dtype=np.float64)
    start = 0
    while start < len(y):
        end = start + 1
        while end < len(y) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    positive_rank_sum = float(ranks[y == 1].sum())
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def threshold_metrics(
    labels: Sequence[int], scores: Sequence[float], threshold: float
) -> dict[str, float]:
    y = np.asarray(labels, dtype=np.int8)
    values = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or values.shape != y.shape or len(y) == 0:
        raise ValueError("labels and scores must be equal nonempty one-dimensional arrays")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(values).all():
        raise ValueError("labels and scores must be finite and binary")
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    predicted = values > threshold
    positive = y == 1
    negative = ~positive
    if not positive.any() or not negative.any():
        raise ValueError("threshold metrics require both labels")
    sensitivity = float(predicted[positive].mean())
    specificity = float((~predicted[negative]).mean())
    return {
        "accuracy": float((predicted == positive).mean()),
        "sensitivity_impossible": sensitivity,
        "specificity_possible": specificity,
        "balanced_accuracy": (sensitivity + specificity) / 2.0,
        "roc_auc": binary_auc(y, values),
        "threshold": float(threshold),
        "count": int(len(y)),
        "possible_count": int(negative.sum()),
        "impossible_count": int(positive.sum()),
    }


def validation_tail_threshold(
    normal_scores: Sequence[float], *, tail_probability: float = 0.05
) -> float:
    """Fix a one-class threshold using only held-out possible validation scores."""
    values = np.asarray(normal_scores, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("at least two finite validation scores are required")
    if not 0 < tail_probability < 1:
        raise ValueError("tail_probability must be between zero and one")
    return float(np.quantile(values, 1.0 - tail_probability, method="higher"))


def grouped_paired_bootstrap(
    labels: Sequence[int],
    first_scores: Sequence[float],
    second_scores: Sequence[float],
    group_ids: Sequence[str],
    *,
    first_threshold: float,
    second_threshold: float,
    replicates: int = 2_000,
    seed: int = 17,
) -> dict[str, object]:
    """Paired scene-group bootstrap for first-minus-second AUC and balanced accuracy."""
    y = np.asarray(labels, dtype=np.int8)
    first = np.asarray(first_scores, dtype=np.float64)
    second = np.asarray(second_scores, dtype=np.float64)
    groups = np.asarray(group_ids, dtype=str)
    if (
        y.ndim != 1
        or first.shape != y.shape
        or second.shape != y.shape
        or groups.shape != y.shape
        or len(y) == 0
        or not np.isin(y, [0, 1]).all()
        or not np.isfinite(first).all()
        or not np.isfinite(second).all()
        or np.any(groups == "")
    ):
        raise ValueError("labels, scores, and nonempty groups must align")
    if replicates < 100:
        raise ValueError("at least 100 bootstrap replicates are required")
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("at least two independent groups are required")
    indices_by_group = [np.flatnonzero(groups == group) for group in unique_groups]
    rng = np.random.default_rng(seed)
    auc_deltas: list[float] = []
    balanced_accuracy_deltas: list[float] = []
    for _ in range(replicates):
        selected_groups = rng.integers(0, len(unique_groups), size=len(unique_groups))
        indices = np.concatenate([indices_by_group[index] for index in selected_groups])
        sample_labels = y[indices]
        if len(np.unique(sample_labels)) < 2:
            continue
        first_metrics = threshold_metrics(sample_labels, first[indices], first_threshold)
        second_metrics = threshold_metrics(sample_labels, second[indices], second_threshold)
        auc_deltas.append(first_metrics["roc_auc"] - second_metrics["roc_auc"])
        balanced_accuracy_deltas.append(
            first_metrics["balanced_accuracy"] - second_metrics["balanced_accuracy"]
        )
    if len(auc_deltas) < replicates * 0.95:
        raise ValueError("too many bootstrap replicates lacked both labels")
    return {
        "method": "paired percentile bootstrap resampling scene groups with replacement",
        "seed": seed,
        "requested_replicates": replicates,
        "valid_replicates": len(auc_deltas),
        "scene_groups": int(len(unique_groups)),
        "first_minus_second_roc_auc": {
            "mean": float(np.mean(auc_deltas)),
            "ci95": [float(item) for item in np.quantile(auc_deltas, [0.025, 0.975])],
        },
        "first_minus_second_balanced_accuracy": {
            "mean": float(np.mean(balanced_accuracy_deltas)),
            "ci95": [float(item) for item in np.quantile(balanced_accuracy_deltas, [0.025, 0.975])],
        },
    }
