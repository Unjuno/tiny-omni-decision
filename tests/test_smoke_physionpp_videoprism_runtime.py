from __future__ import annotations

import json

import pytest

from scripts.smoke_physionpp_videoprism_runtime import (
    _read_rows,
    _safe_media_path,
    uniform_indices,
)


def test_video_frame_indices_are_uniform_and_include_endpoints() -> None:
    assert uniform_indices(29) == [0, 4, 8, 12, 16, 20, 24, 28]


def test_video_frame_indices_reject_short_inputs() -> None:
    with pytest.raises(ValueError):
        uniform_indices(1)
    with pytest.raises(ValueError):
        uniform_indices(10, 1)


def test_media_path_rejects_parent_traversal_and_absolute_paths() -> None:
    with pytest.raises(ValueError):
        _safe_media_path("../../outside.mp4")
    with pytest.raises(ValueError):
        _safe_media_path("C:/outside.mp4")


def test_train_reader_rejects_non_train_rows(tmp_path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps({"id": "x", "split": "validation"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="TRAIN-only"):
        _read_rows(path)


def test_train_reader_rejects_empty_manifest(tmp_path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="TRAIN-only"):
        _read_rows(path)
