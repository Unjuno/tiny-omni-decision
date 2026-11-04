"""Tests for deterministic AVATAR VideoPrism feature input construction."""

from __future__ import annotations

import json

import pytest

from scripts.extract_avatar_videoprism_features import load_choice_assets


def _write_rows(path, split: str, options: list[tuple[str, float]]) -> None:
    row = {
        "split": split,
        "options": [
            {"video_asset_key": key, "video_timestamp_seconds": timestamp}
            for key, timestamp in options
        ],
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")


def test_load_choice_assets_is_sorted_and_deduplicates_shared_frames(tmp_path) -> None:
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    _write_rows(
        train,
        "train",
        [
            ("bbbbbbbbbbb_00001/00002", 1.5),
            ("aaaaaaaaaaa_00002/00003", 2.0),
        ],
    )
    _write_rows(
        validation,
        "validation",
        [("bbbbbbbbbbb_00001/00002", 1.5)],
    )

    assert load_choice_assets(train, validation) == [
        {"asset_key": "aaaaaaaaaaa_00002/00003", "timestamp_seconds": 2.0},
        {"asset_key": "bbbbbbbbbbb_00001/00002", "timestamp_seconds": 1.5},
    ]


def test_load_choice_assets_rejects_timestamp_conflicts(tmp_path) -> None:
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    _write_rows(train, "train", [("aaaaaaaaaaa_00002/00003", 2.0)])
    _write_rows(validation, "validation", [("aaaaaaaaaaa_00002/00003", 2.25)])

    with pytest.raises(ValueError, match="conflicting timestamp"):
        load_choice_assets(train, validation)


def test_load_choice_assets_rejects_split_mismatch_and_unsafe_keys(tmp_path) -> None:
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    _write_rows(train, "validation", [("aaaaaaaaaaa_00002/00003", 2.0)])
    _write_rows(validation, "validation", [("../../escape.mp4", 2.0)])

    with pytest.raises(ValueError, match="unexpected split"):
        load_choice_assets(train, validation)

    _write_rows(train, "train", [("../../escape.mp4", 2.0)])
    with pytest.raises(ValueError, match="invalid video asset key"):
        load_choice_assets(train, validation)
