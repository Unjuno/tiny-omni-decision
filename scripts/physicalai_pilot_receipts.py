"""Helpers for the pinned PhysicalAI pilot receipt schema."""

from __future__ import annotations

from typing import Any


def split_episode_ids(receipt: dict[str, Any]) -> tuple[set[int], set[int]]:
    selection = receipt.get("selection", receipt)
    train = selection.get("train_episode_indices", selection.get("train_ids", []))
    validation = selection.get(
        "validation_episode_indices", selection.get("validation_ids", [])
    )
    train_ids = set(train)
    validation_ids = set(validation)
    if train_ids & validation_ids:
        raise ValueError("receipt has episode IDs assigned to both train and validation")
    return train_ids, validation_ids
