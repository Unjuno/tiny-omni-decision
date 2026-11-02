from __future__ import annotations

import json

import pytest

from scripts.build_avatar_train_only_group_fold import _group_partition
from scripts.extract_avatar_dinov2_train_frames import expected_video_assets


def test_train_only_group_partition_is_deterministic_and_disjoint():
    groups = {f"source-{index:03d}" for index in range(30)}
    first_train, first_validation = _group_partition(groups)
    second_train, second_validation = _group_partition(groups)

    assert first_train == second_train
    assert first_validation == second_validation
    assert first_train.isdisjoint(first_validation)
    assert first_train | first_validation == groups
    assert len(first_train) == 24
    assert len(first_validation) == 6


def test_dinov2_asset_list_uses_only_frozen_train_and_validation_rows(tmp_path):
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    train.write_text(
        json.dumps(
            {
                "split": "train",
                "options": [
                    {"video_asset_key": "clip-a/00001", "video_timestamp_seconds": 0.5},
                    {"video_asset_key": "clip-b/00002", "video_timestamp_seconds": 1.0},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    validation.write_text(
        json.dumps(
            {
                "split": "validation",
                "options": [{"video_asset_key": "clip-a/00001", "video_timestamp_seconds": 0.5}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assets = expected_video_assets([train, validation])

    assert assets == [
        {"asset_key": "clip-a/00001", "timestamp_seconds": 0.5},
        {"asset_key": "clip-b/00002", "timestamp_seconds": 1.0},
    ]


def test_dinov2_asset_list_rejects_timestamp_conflicts(tmp_path):
    source = tmp_path / "split.jsonl"
    row = {
        "split": "train",
        "options": [
            {"video_asset_key": "clip-a/00001", "video_timestamp_seconds": 0.5},
            {"video_asset_key": "clip-a/00001", "video_timestamp_seconds": 0.75},
        ],
    }
    source.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="conflicting timestamp"):
        expected_video_assets([source])
