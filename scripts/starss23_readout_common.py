"""Fixed STARSS23 readout metrics and label-independent video controls."""

from __future__ import annotations

import hashlib
from collections import Counter


def different_recording_permutation(recordings, seed=17):
    """Preserve the exact video marginal; reject impossible cross-recording derangements."""
    counts = Counter(recordings)
    if not counts or 2 * max(counts.values()) > len(recordings):
        raise ValueError("cannot permute every window to a different recording")
    order = sorted(
        range(len(recordings)),
        key=lambda i: (hashlib.sha256(f"{seed}:{recordings[i]}".encode()).hexdigest(), i),
    )
    offset = max(counts.values())
    shifted = order[offset:] + order[:offset]
    result = [0] * len(order)
    for original, replacement in zip(order, shifted, strict=True):
        result[original] = replacement
    if any(recordings[i] == recordings[j] for i, j in enumerate(result)):
        raise ValueError("cross-recording permutation invariant failed")
    return result


def binary_metrics(targets, positive_probabilities):
    import numpy as np

    y = np.asarray(targets)
    p = np.asarray(positive_probabilities, dtype=np.float64)
    if y.ndim != 1 or y.shape != p.shape or not y.size:
        raise ValueError("invalid metric shapes")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("invalid targets or probabilities")
    predicted = p > 0.5  # argmax over [no, yes] chooses no on a tie
    correct = predicted == y
    confidence = np.maximum(p, 1 - p)
    clipped = np.clip(p, 1e-7, 1 - 1e-7)
    bins = np.minimum((confidence * 15).astype(int), 14)
    bin_counts = [int((bins == i).sum()) for i in range(15)]
    ece = sum(
        float((bins == i).mean())
        * abs(float(correct[bins == i].mean()) - float(confidence[bins == i].mean()))
        for i in range(15)
        if bin_counts[i]
    )
    positives, negatives = int(y.sum()), int((y == 0).sum())
    sensitivity = float(correct[y == 1].mean()) if positives else None
    specificity = float(correct[y == 0].mean()) if negatives else None
    return {
        "count": int(y.size),
        "positive_count": positives,
        "negative_count": negatives,
        "accuracy": float(correct.mean()),
        "nll": float(-(y * np.log(clipped) + (1 - y) * np.log(1 - clipped)).mean()),
        "brier": float((2 * (p - y) ** 2).mean()),
        "ece": ece,
        "ece_bin_counts": bin_counts,
        "balanced_accuracy": (sensitivity + specificity) / 2 if positives and negatives else None,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }
