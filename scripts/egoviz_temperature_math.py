"""Deterministic positive temperature fitting and grouped cross-fitting."""

from __future__ import annotations

import numpy as np


def fit_alpha(logits: np.ndarray, targets: np.ndarray) -> float:
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    if (logits.ndim != 2 or logits.shape[1] != 4 or logits.shape[0] == 0
            or targets.shape != (logits.shape[0],) or not np.isfinite(logits).all()
            or np.any(targets < 0) or np.any(targets >= 4)):
        raise ValueError("invalid four-choice logits or targets")

    def derivative(alpha: float) -> float:
        scaled = alpha * logits
        weights = np.exp(scaled - scaled.max(axis=1, keepdims=True))
        weights /= weights.sum(axis=1, keepdims=True)
        expected = (weights * logits).sum(axis=1)
        return float(np.mean(expected - logits[np.arange(len(targets)), targets]))

    low, high = 0.05, 20.0
    if derivative(low) >= 0:
        return low
    if derivative(high) <= 0:
        return high
    for _ in range(80):
        mid = (low + high) / 2
        if derivative(mid) < 0:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def cross_fit(logits: np.ndarray, targets: np.ndarray, groups: np.ndarray):
    groups = np.asarray(groups)
    if groups.shape != (len(logits),):
        raise ValueError("group count differs from logits")
    unique = np.unique(groups)
    if len(unique) != 20:
        raise ValueError("expected exactly 20 development session groups")
    shuffled = np.random.default_rng(17).permutation(unique)
    output = np.empty_like(np.asarray(logits, dtype=np.float64))
    folds = []
    for index, heldout in enumerate(np.array_split(shuffled, 5)):
        mask = np.isin(groups, heldout)
        alpha = fit_alpha(logits[~mask], targets[~mask])
        output[mask] = alpha * logits[mask]
        folds.append({"fold": index, "heldout_groups": sorted(heldout.tolist()),
                      "training_examples": int((~mask).sum()),
                      "heldout_examples": int(mask.sum()), "alpha": alpha})
    return output, folds
