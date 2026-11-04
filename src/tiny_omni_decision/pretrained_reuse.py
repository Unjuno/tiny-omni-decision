"""Dependency-light integrity helpers for pretrained representation probes."""

from __future__ import annotations

from typing import Any


def validate_validation_partition(
    validation_ids: list[str], selection_ids: list[str]
) -> tuple[list[str], list[str]]:
    if len(validation_ids) != len(set(validation_ids)):
        raise ValueError("full validation IDs contain duplicates")
    if len(selection_ids) != len(set(selection_ids)):
        raise ValueError("checkpoint-selection IDs contain duplicates")
    missing = set(selection_ids) - set(validation_ids)
    if missing:
        raise ValueError(
            f"checkpoint-selection IDs are absent from full validation: {len(missing)}"
        )
    selection = set(selection_ids)
    ordered_selection = [image_id for image_id in validation_ids if image_id in selection]
    new_holdout = [image_id for image_id in validation_ids if image_id not in selection]
    if (set(ordered_selection) & set(new_holdout)) or (
        set(ordered_selection) | set(new_holdout) != set(validation_ids)
    ):
        raise AssertionError("validation selection/holdout partition is inconsistent")
    return ordered_selection, new_holdout


def validate_clevr4_splits(
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    train_assets: list[dict[str, Any]],
    validation_assets: list[dict[str, Any]],
) -> None:
    if [row["id"] for row in train_rows] != [row["id"] for row in train_assets]:
        raise ValueError("train labels and image manifest ID order differ")
    if [row["id"] for row in validation_rows] != [row["id"] for row in validation_assets]:
        raise ValueError("validation labels and image manifest ID order differ")
    if {row.get("split") for row in train_rows} != {"train"}:
        raise ValueError("train label file contains non-train rows")
    if {row.get("split") for row in validation_rows} != {"val"}:
        raise ValueError("validation label file contains non-validation rows")
    if {row["id"] for row in train_rows} & {row["id"] for row in validation_rows}:
        raise ValueError("train and validation image IDs overlap")
    if {row["sha256"] for row in train_assets} & {row["sha256"] for row in validation_assets}:
        raise ValueError("train and validation media hashes overlap")
