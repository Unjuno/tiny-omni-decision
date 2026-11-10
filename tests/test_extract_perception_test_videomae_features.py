from __future__ import annotations

import numpy as np
import pytest

from scripts.extract_perception_test_videomae_features import (
    FRAME_COUNT,
    FRAME_SIZE,
    TOKEN_SHAPE,
    preprocess_frames,
    uniform_indices,
)


def test_uniform_frame_indices_keep_endpoints_and_fixed_count():
    values = uniform_indices(651)
    assert len(values) == FRAME_COUNT
    assert values[0] == 0
    assert values[-1] == 650
    assert values == sorted(set(values))
    assert TOKEN_SHAPE == (8, 14, 14, 384)


def test_uniform_frame_indices_reject_short_videos():
    with pytest.raises(ValueError, match="at least"):
        uniform_indices(FRAME_COUNT - 1)


def test_preprocess_frames_matches_fixed_float_range():
    frames = np.zeros((FRAME_COUNT, FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8)
    frames[-1] = 255
    result = preprocess_frames(frames)
    assert result.shape == frames.shape
    assert result.dtype == np.float32
    assert result[0].min() == -1.0
    assert result[-1].max() == 1.0
    assert np.isfinite(result).all()


def test_preprocess_frames_rejects_wrong_shape_or_dtype():
    with pytest.raises(ValueError, match="pinned RGB"):
        preprocess_frames(np.zeros((2, 224, 224, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="pinned RGB"):
        preprocess_frames(np.zeros((FRAME_COUNT, FRAME_SIZE, FRAME_SIZE, 3), dtype=np.float32))
