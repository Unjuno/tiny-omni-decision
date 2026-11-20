"""Integrity and metric helpers for the R2-M supplied-activity Decision task."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def supported_activity_vocabulary(
    counts_by_split: Mapping[str, Mapping[str, int]],
    *,
    required_splits: Sequence[str] = ("train", "calibration", "validation"),
) -> tuple[str, ...]:
    """Return a stable vocabulary supported by every declared development split.

    Coverage is based only on positive example counts, never model performance.
    Callers must preserve and report labels excluded for missing split coverage.
    """
    if not isinstance(counts_by_split, Mapping) or not required_splits:
        raise ValueError("split counts and required split names are required")
    if len(set(required_splits)) != len(required_splits):
        raise ValueError("required split names must be unique")
    supported: list[set[str]] = []
    for split in required_splits:
        counts = counts_by_split.get(split)
        if not isinstance(counts, Mapping):
            raise ValueError(f"missing activity counts for split {split!r}")
        current = set()
        for label, count in counts.items():
            if not isinstance(label, str) or not label:
                raise ValueError("activity labels must be nonempty strings")
            if type(count) is not int or count < 0:
                raise ValueError("activity counts must be nonnegative integers")
            if count > 0:
                current.add(label)
        if not current:
            raise ValueError(f"split {split!r} has no supported activities")
        supported.append(current)
    vocabulary = tuple(sorted(set.intersection(*supported)))
    if len(vocabulary) < 2:
        raise ValueError("at least two activities must be present in every split")
    return vocabulary


def _expected_calibration_error(confidence: np.ndarray, correct: np.ndarray,
                                bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index, lower in enumerate(edges[:-1]):
        upper = edges[index + 1]
        upper_mask = confidence <= upper if index == bins - 1 else confidence < upper
        mask = (confidence >= lower) & upper_mask
        if mask.any():
            result += float(mask.mean() * abs(confidence[mask].mean() - correct[mask].mean()))
    return result


def activity_metrics(
    probabilities: np.ndarray,
    targets: np.ndarray,
    vocabulary: Sequence[str],
) -> dict[str, object]:
    """Compute standard and class-macro metrics for fixed supplied choices."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    if (probabilities.ndim != 2 or targets.ndim != 1
            or probabilities.shape != (len(targets), len(vocabulary))
            or len(targets) == 0 or len(vocabulary) < 2):
        raise ValueError("probabilities/targets do not match nonempty vocabulary")
    if (not np.isfinite(probabilities).all() or (probabilities < 0).any()
            or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)):
        raise ValueError("probabilities must be finite, nonnegative and normalized")
    if (targets < 0).any() or (targets >= len(vocabulary)).any():
        raise ValueError("target is outside the supplied activity choices")

    predictions = probabilities.argmax(axis=1)
    chosen = np.maximum(probabilities[np.arange(len(targets)), targets], 1e-15)
    one_hot = np.eye(len(vocabulary), dtype=np.float64)[targets]
    confidence = probabilities.max(axis=1)
    correct = (predictions == targets).astype(np.float64)
    per_activity: dict[str, dict[str, float | int]] = {}
    for index, label in enumerate(vocabulary):
        mask = targets == index
        if not mask.any():
            raise ValueError(f"activity {label!r} has no evaluation examples")
        per_activity[label] = {
            "examples": int(mask.sum()),
            "accuracy": float(correct[mask].mean()),
            "nll": float(-np.log(chosen[mask]).mean()),
            "brier": float(((probabilities[mask] - one_hot[mask]) ** 2).sum(axis=1).mean()),
        }
    return {
        "examples": int(len(targets)),
        "choices": int(len(vocabulary)),
        "accuracy": float(correct.mean()),
        "macro_accuracy": float(np.mean([row["accuracy"] for row in per_activity.values()])),
        "nll": float(-np.log(chosen).mean()),
        "macro_nll": float(np.mean([row["nll"] for row in per_activity.values()])),
        "brier": float(((probabilities - one_hot) ** 2).sum(axis=1).mean()),
        "macro_brier": float(np.mean([row["brier"] for row in per_activity.values()])),
        "ece_15bin": _expected_calibration_error(confidence, correct),
        "mean_confidence": float(confidence.mean()),
        "per_activity": per_activity,
    }


def train_sensor_normalization(
    sensor_sequences: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate per-axis standardization from training examples only."""
    if not sensor_sequences:
        raise ValueError("at least one training sensor sequence is required")
    arrays = [np.asarray(sequence, dtype=np.float64) for sequence in sensor_sequences]
    if any(sequence.ndim != 2 or sequence.shape[1] != 3 or not len(sequence)
           for sequence in arrays):
        raise ValueError("sensor sequences must be nonempty [time, 3] arrays")
    if any(not np.isfinite(sequence).all() for sequence in arrays):
        raise ValueError("sensor sequences must contain finite values")
    combined = np.concatenate(arrays, axis=0)
    mean = combined.mean(axis=0)
    standard_deviation = combined.std(axis=0)
    if (not np.isfinite(mean).all() or not np.isfinite(standard_deviation).all()
            or (standard_deviation <= 0).any()):
        raise ValueError("each sensor axis must have finite nonzero training variance")
    return mean.astype(np.float32), standard_deviation.astype(np.float32)
