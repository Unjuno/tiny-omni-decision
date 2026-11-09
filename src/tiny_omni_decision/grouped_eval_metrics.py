"""Metrics and paired speaker-group bootstrap for fixed evaluation cohorts."""

from __future__ import annotations

import math

import numpy as np

from .decision_math import brier_score, expected_calibration_error, negative_log_likelihood


def classification_metrics(
    probabilities: list[list[float]], targets: list[int], *, ece_bins: int = 15
) -> dict[str, float]:
    if not probabilities or len(probabilities) != len(targets):
        raise ValueError("probabilities and targets must be nonempty and equal length")
    if any(len(row) < 2 or not math.isclose(sum(row), 1.0, abs_tol=1e-5) for row in probabilities):
        raise ValueError("each probability row must be a normalized multiclass distribution")
    predictions = [max(range(len(row)), key=row.__getitem__) for row in probabilities]
    confidence = [
        row[prediction] for row, prediction in zip(probabilities, predictions, strict=True)
    ]
    return {
        "accuracy": sum(
            int(p == y) for p, y in zip(predictions, targets, strict=True)
        ) / len(targets),
        "nll": negative_log_likelihood(probabilities, targets),
        "brier": brier_score(probabilities, targets),
        "ece15": expected_calibration_error(probabilities, targets, ece_bins),
        "mean_confidence": sum(confidence) / len(confidence),
    }


def grouped_bootstrap(
    probabilities: list[list[float]],
    targets: list[int],
    groups: list[str],
    *,
    replicates: int = 10_000,
    seed: int = 17,
    ece_bins: int = 15,
    baseline_probabilities: list[list[float]] | None = None,
) -> dict[str, dict[str, list[float]]]:
    """Resample source groups, preserving all examples in each sampled group."""
    if not (len(probabilities) == len(targets) == len(groups)) or not groups:
        raise ValueError("probabilities, targets, and groups must have equal nonzero lengths")
    if replicates < 1:
        raise ValueError("replicates must be positive")
    group_to_indices: dict[str, list[int]] = {}
    for index, group in enumerate(groups):
        if not group:
            raise ValueError("group identities must be nonempty")
        group_to_indices.setdefault(group, []).append(index)
    unique_groups = sorted(group_to_indices)
    rng = np.random.default_rng(seed)
    draws = {name: [] for name in ("accuracy", "nll", "brier", "ece15", "mean_confidence")}
    deltas = {name: [] for name in ("nll", "brier")} if baseline_probabilities is not None else {}
    if baseline_probabilities is not None and len(baseline_probabilities) != len(targets):
        raise ValueError("baseline probabilities must match the candidate cohort")

    for _ in range(replicates):
        sampled = rng.choice(len(unique_groups), size=len(unique_groups), replace=True)
        indices = [
            i
            for group_index in sampled
            for i in group_to_indices[unique_groups[int(group_index)]]
        ]
        sample_probs = [probabilities[i] for i in indices]
        sample_targets = [targets[i] for i in indices]
        metrics = classification_metrics(sample_probs, sample_targets, ece_bins=ece_bins)
        for name, value in metrics.items():
            draws[name].append(value)
        if baseline_probabilities is not None:
            base = classification_metrics(
                [baseline_probabilities[i] for i in indices], sample_targets, ece_bins=ece_bins
            )
            for name in deltas:
                deltas[name].append(metrics[name] - base[name])
    return {"metrics": draws, "candidate_minus_baseline": deltas}


def percentile_interval(values: list[float]) -> list[float]:
    if not values:
        raise ValueError("cannot calculate an interval from no values")
    low, high = np.quantile(np.asarray(values, dtype=np.float64), [0.025, 0.975])
    return [float(low), float(high)]
