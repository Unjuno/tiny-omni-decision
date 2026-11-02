"""Metric helpers for fixed PhysicalAI action-prototype decision pilots."""

from __future__ import annotations

import numpy as np


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
