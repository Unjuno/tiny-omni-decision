from __future__ import annotations

import hashlib

import numpy as np
import pytest

from scripts.extract_cater_videoprism_temporal_features import (
    attach_media_records,
    select_interval_frame_indices,
    token_tensor,
)


@pytest.mark.parametrize("interval", [(0, 75), (75, 150), (150, 225), (225, 300)])
def test_frame_indices_stay_inside_each_task_interval(interval: tuple[int, int]) -> None:
    indices = select_interval_frame_indices(*interval)

    assert len(indices) == 13
    assert indices[0] == interval[0]
    assert indices[-1] == interval[1]
    assert tuple(sorted(set(indices))) == indices
    assert all(interval[0] <= index <= interval[1] for index in indices)


def test_frame_indices_reject_too_short_interval() -> None:
    with pytest.raises(ValueError, match="too short"):
        select_interval_frame_indices(0, 3)


def test_frame_indices_for_second_interval_never_read_later_video_content() -> None:
    indices = select_interval_frame_indices(75, 150)

    assert min(indices) == 75
    assert max(indices) == 150
    assert 151 not in indices


def test_media_validation_hashes_each_group_asset_once_and_keeps_all_rows(tmp_path) -> None:
    payload = b"same underlying CATER asset"
    media_path = tmp_path / "train" / "clip.avi"
    media_path.parent.mkdir()
    media_path.write_bytes(payload)
    media_hash = hashlib.sha256(payload).hexdigest()

    def row(sample_id: str, interval: list[int]) -> dict:
        return {
            "id": sample_id,
            "video_id": "CATER_new_000001.avi",
            "group_id": "cater-video-000001",
            "split": "train",
            "frame_interval": interval,
        }

    train, validation = attach_media_records(
        [row("q1", [0, 75]), row("q2", [75, 150])],
        [],
        [
            {
                "split": "train",
                "video_id": "CATER_new_000001.avi",
                "relative_path": "train/clip.avi",
                "media_sha256": media_hash,
            }
        ],
        tmp_path,
    )

    assert len(train) == 2
    assert validation == []
    assert train[0][1] == train[1][1] == media_path
    assert train[0][2] == train[1][2] == media_hash


def test_token_tensor_preserves_order_and_spatial_tokens_in_float16() -> None:
    tokens = (
        np.arange(13 * 256 * 768, dtype=np.float32) / (13 * 256 * 768)
    ).reshape(1, -1, 768)

    result = token_tensor(tokens, frame_count=13)

    assert result.shape == (13, 256, 768)
    assert result.dtype == np.float16
    assert result[0, 0, 0] == tokens[0, 0, 0]
    assert result[-1, -1, -1] == pytest.approx(tokens[0, -1, -1], abs=1e-3)


@pytest.mark.parametrize(
    "tokens",
    [
        np.zeros((13 * 256, 768), dtype=np.float32),
        np.zeros((1, 13 * 255, 768), dtype=np.float32),
        np.full((1, 13 * 256, 768), np.nan, dtype=np.float32),
        np.full((1, 13 * 256, 768), 70_000, dtype=np.float32),
    ],
)
def test_token_tensor_rejects_invalid_model_outputs(tokens: np.ndarray) -> None:
    with pytest.raises(ValueError):
        token_tensor(tokens, frame_count=13)
