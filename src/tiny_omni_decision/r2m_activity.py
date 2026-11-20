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
    *,
    allow_missing_classes: bool = False,
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
            if allow_missing_classes:
                continue
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
        "macro_classes_evaluated": len(per_activity),
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


def same_participant_different_activity_sensor_donors(
    rows: Sequence[Mapping[str, object]], *, seed: int = 17,
) -> list[Mapping[str, object]]:
    """Create a deterministic sensor donor derangement within each participant.

    Every donor remains from the recipient's participant but has a different
    activity label. The donor mapping is a permutation, so each source sensor
    is used exactly once. A half-cycle across label-sorted blocks guarantees
    different labels when no label occupies more than half of a participant's
    rows; otherwise the function fails closed.
    """
    if not rows:
        raise ValueError("sensor donor derangement requires at least one row")
    participant_indices: dict[int, list[int]] = {}
    for index, row in enumerate(rows):
        participant = int(row["participant_id"])
        participant_indices.setdefault(participant, []).append(index)

    donors: list[Mapping[str, object] | None] = [None] * len(rows)
    rng = np.random.default_rng(seed)
    for participant, indices in sorted(participant_indices.items()):
        ordered = sorted(
            indices,
            key=lambda index: (str(rows[index]["activity"]), str(rows[index]["sample_id"])),
        )
        counts: dict[str, int] = {}
        for index in ordered:
            label = str(rows[index]["activity"])
            counts[label] = counts.get(label, 0) + 1
        if len(counts) < 2 or max(counts.values()) > len(ordered) // 2:
            raise ValueError(
                f"participant {participant} lacks a full different-activity sensor derangement"
            )

        by_activity: dict[str, list[int]] = {}
        for index in ordered:
            by_activity.setdefault(str(rows[index]["activity"]), []).append(index)
        donor_order: list[int] = []
        for activity in sorted(by_activity):
            class_donors = by_activity[activity].copy()
            rng.shuffle(class_donors)
            donor_order.extend(class_donors)

        offset = len(ordered) // 2
        for position, recipient_index in enumerate(ordered):
            donor_index = donor_order[(position + offset) % len(ordered)]
            recipient = rows[recipient_index]
            donor = rows[donor_index]
            if (int(recipient["participant_id"]) != int(donor["participant_id"])
                    or str(recipient["activity"]) == str(donor["activity"])):
                raise ValueError("constructed sensor mapping violates its group constraints")
            donors[recipient_index] = donor

    if any(donor is None for donor in donors):
        raise AssertionError("sensor derangement did not assign every validation row")
    result = [donor for donor in donors if donor is not None]
    if {str(row["sample_id"]) for row in result} != {
        str(row["sample_id"]) for row in rows
    }:
        raise AssertionError("sensor donor mapping is not a permutation of the input rows")
    return result
