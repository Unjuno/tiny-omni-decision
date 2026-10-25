"""Aggregation helpers for the fixed synthetic AV validation cohort."""

from __future__ import annotations

import numpy as np


def group_accuracy(correct: np.ndarray, group_ids: np.ndarray) -> dict[str, float]:
    correct = np.asarray(correct, dtype=np.float64)
    group_ids = np.asarray(group_ids).astype(str)
    if correct.ndim != 1 or group_ids.ndim != 1 or len(correct) != len(group_ids):
        raise ValueError("correct flags and group IDs must be matching vectors")
    if len(correct) == 0 or not np.isfinite(correct).all():
        raise ValueError("group accuracy requires non-empty finite values")
    values = {}
    for group in sorted(set(group_ids.tolist())):
        selected = correct[group_ids == group]
        values[group] = float(selected.mean())
    return values


def paired_group_bootstrap_difference(
    reference: dict[str, float],
    candidate: dict[str, float],
    *,
    replicates: int = 10_000,
    seed: int = 17,
) -> dict[str, float | int]:
    if replicates < 1 or seed < 0:
        raise ValueError("replicates must be positive and seed non-negative")
    if not reference or reference.keys() != candidate.keys():
        raise ValueError("paired bootstrap inputs must have the same non-empty group IDs")
    groups = sorted(reference)
    ref = np.asarray([reference[group] for group in groups], dtype=np.float64)
    cand = np.asarray([candidate[group] for group in groups], dtype=np.float64)
    if not np.isfinite(ref).all() or not np.isfinite(cand).all():
        raise ValueError("bootstrap values must be finite")
    differences = cand - ref
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, len(groups), size=(replicates, len(groups)))
    boot = differences[draws].mean(axis=1)
    lower, upper = np.percentile(boot, [2.5, 97.5])
    return {
        "group_count": len(groups),
        "replicates": replicates,
        "seed": seed,
        "mean_accuracy_difference": float(differences.mean()),
        "lower_95_percentile": float(lower),
        "upper_95_percentile": float(upper),
    }
