"""Tests for deterministic AVATAR audio-feature inputs and crop boundaries."""

from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.extract_avatar_efficientat_features import (
    crop_audio_window,
    load_audio_assets,
    trim_to_audio_stream_end,
)


def _write_row(path, split: str, key: str, center: float, left: float, right: float) -> None:
    path.write_text(
        json.dumps(
            {
                "split": split,
                "audio_asset_key": key,
                "audio_window_center_seconds": center,
                "audio_window_duration_seconds": 2.0,
                "audio_window_left_padding_seconds": left,
                "audio_window_right_padding_seconds": right,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_load_audio_assets_sorts_and_deduplicates_identical_windows(tmp_path) -> None:
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    _write_row(train, "train", "bbbbbbbbbbb_00001/00002", 1.5, 0.0, 0.0)
    _write_row(validation, "validation", "aaaaaaaaaaa_00002/00003", 0.5, 0.5, 0.0)
    with train.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "split": "train",
                    "audio_asset_key": "bbbbbbbbbbb_00001/00002",
                    "audio_window_center_seconds": 1.5,
                    "audio_window_duration_seconds": 2.0,
                    "audio_window_left_padding_seconds": 0.0,
                    "audio_window_right_padding_seconds": 0.0,
                }
            )
            + "\n"
        )

    assert load_audio_assets(train, validation) == [
        {
            "asset_key": "aaaaaaaaaaa_00002/00003",
            "center_seconds": 0.5,
            "declared_left_padding_seconds": 0.5,
            "declared_right_padding_seconds": 0.0,
        },
        {
            "asset_key": "bbbbbbbbbbb_00001/00002",
            "center_seconds": 1.5,
            "declared_left_padding_seconds": 0.0,
            "declared_right_padding_seconds": 0.0,
        },
    ]


def test_load_audio_assets_rejects_conflicting_or_noncanonical_windows(tmp_path) -> None:
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    _write_row(train, "train", "aaaaaaaaaaa_00002/00003", 0.5, 0.5, 0.0)
    _write_row(validation, "validation", "aaaaaaaaaaa_00002/00003", 0.75, 0.25, 0.0)

    with pytest.raises(ValueError, match="conflicting audio window"):
        load_audio_assets(train, validation)

    _write_row(train, "validation", "../../escape.mp4", 0.5, 0.0, 0.0)
    with pytest.raises(ValueError, match="unexpected split"):
        load_audio_assets(train, validation)


def test_crop_audio_window_pads_both_boundaries_to_exact_sample_count() -> None:
    waveform = np.arange(100, dtype=np.float32)
    left, left_pad, left_right_pad = crop_audio_window(
        waveform, 0.5, sample_rate=10, duration_seconds=2.0
    )
    assert left.shape == (20,)
    assert left_pad == 5
    assert left_right_pad == 0
    np.testing.assert_array_equal(left, np.concatenate([np.zeros(5), np.arange(15)]))

    right, right_left_pad, right_pad = crop_audio_window(
        waveform,
        9.5,
        sample_rate=10,
        duration_seconds=2.0,
        right_boundary_seconds=10.0,
    )
    assert right.shape == (20,)
    assert right_left_pad == 0
    assert right_pad == 5
    np.testing.assert_array_equal(right, np.concatenate([np.arange(85, 100), np.zeros(5)]))


def test_trim_to_audio_stream_end_removes_resampler_tail() -> None:
    waveform = np.ones(320_250, dtype=np.float32)
    trimmed = trim_to_audio_stream_end(waveform, end_seconds=10.0)
    assert trimmed.shape == (320_000,)
    np.testing.assert_array_equal(trimmed, np.ones(320_000, dtype=np.float32))


def test_trim_to_audio_stream_end_rejects_incomplete_decode() -> None:
    waveform = np.ones(319_999, dtype=np.float32)
    with pytest.raises(ValueError, match="shorter than declared audio stream end"):
        trim_to_audio_stream_end(waveform, end_seconds=10.0)
