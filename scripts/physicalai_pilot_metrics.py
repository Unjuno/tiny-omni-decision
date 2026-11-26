"""Metric helpers for fixed PhysicalAI action-prototype decision pilots."""

from __future__ import annotations

import numpy as np


def expected_calibration_error(
    probabilities: np.ndarray, targets: np.ndarray, *, bins: int = 10
) -> float:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets)
    if probabilities.ndim != 2 or targets.ndim != 1:
        raise ValueError("probabilities must be 2-D and targets must be 1-D")
    if probabilities.shape[0] != targets.shape[0] or targets.size == 0:
        raise ValueError("probabilities and targets must have the same nonzero row count")
    if bins < 1:
        raise ValueError("bins must be positive")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0.0):
        raise ValueError("probabilities must be finite and nonnegative")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5, rtol=0.0):
        raise ValueError("each probability row must sum to one")
    confidence = probabilities.max(axis=1)
    if np.any(confidence > 1.0):
        raise ValueError("probabilities must not exceed one")
    if targets.min() < 0 or targets.max() >= probabilities.shape[1]:
        raise ValueError("targets contain a class outside the probability columns")

    predictions = probabilities.argmax(axis=1)
    bin_indices = np.minimum((confidence * bins).astype(int), bins - 1)
    ece = 0.0
    for bin_index in range(bins):
        selected = bin_indices == bin_index
        if selected.any():
            ece += float(selected.mean()) * abs(
                float((predictions[selected] == targets[selected]).mean())
                - float(confidence[selected].mean())
            )
    return float(ece)


def episode_counts_by_split(
    episode_ids: np.ndarray, train_mask: np.ndarray, validation_mask: np.ndarray
) -> dict[str, int]:
    episode_ids = np.asarray(episode_ids)
    train_mask = np.asarray(train_mask, dtype=bool)
    validation_mask = np.asarray(validation_mask, dtype=bool)
    if episode_ids.ndim != 1 or train_mask.shape != episode_ids.shape:
        raise ValueError("episode IDs and split masks must be aligned 1-D arrays")
    if validation_mask.shape != episode_ids.shape:
        raise ValueError("episode IDs and split masks must be aligned 1-D arrays")
    train_ids = np.unique(episode_ids[train_mask])
    validation_ids = np.unique(episode_ids[validation_mask])
    if np.intersect1d(train_ids, validation_ids).size:
        raise ValueError("train and validation episode IDs overlap")
    return {"train": int(train_ids.size), "validation": int(validation_ids.size)}


def per_class_accuracy(
    probabilities: np.ndarray, targets: np.ndarray
) -> list[dict[str, float | int | None]]:
    probabilities = np.asarray(probabilities)
    targets = np.asarray(targets)
    if probabilities.ndim != 2 or targets.ndim != 1:
        raise ValueError("probabilities must be 2-D and targets must be 1-D")
    if probabilities.shape[0] != targets.shape[0]:
        raise ValueError("probabilities and targets must have the same number of rows")
    if targets.size and (targets.min() < 0 or targets.max() >= probabilities.shape[1]):
        raise ValueError("targets contain a class outside the probability columns")
    predictions = probabilities.argmax(axis=1)
    result: list[dict[str, float | int | None]] = []
    for class_index in range(probabilities.shape[1]):
        selected = targets == class_index
        support = int(selected.sum())
        result.append(
            {
                "class": class_index,
                "support": support,
                "accuracy": (
                    float((predictions[selected] == targets[selected]).mean())
                    if support
                    else None
                ),
            }
        )
    return result
