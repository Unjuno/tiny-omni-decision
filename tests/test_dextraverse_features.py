"""Tests for DextraVerse feature cache transformations."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.pretrained_reuse.dextraverse_features import (
    assemble_sensor_window,
    pool_videoprism_tokens,
)


def test_spatial_token_pooling_preserves_eight_temporal_positions() -> None:
    tokens = np.arange(8 * 4 * 768, dtype=np.float32).reshape(1, 8 * 4, 768)
    pooled = pool_videoprism_tokens(tokens)
    assert pooled.shape == (8, 768)
    np.testing.assert_allclose(pooled[0], tokens.reshape(1, 8, 4, 768)[0, 0].mean(0))
    assert not np.array_equal(pooled[0], pooled[-1])


def test_sensor_window_keeps_left_then_right_channels_and_causal_time() -> None:
    left = np.zeros((5, 5, 3), dtype=np.float32)
    right = np.zeros_like(left)
    left[:, 0, 0] = 1
    right[:, 0, 0] = 2
    times = np.arange(5, dtype=np.float64) * 0.1
    values, relative_ms = assemble_sensor_window(left, right, times, [2, 3, 4], 4)
    assert values.shape == (3, 30)
    assert values[0, 0] == 1
    assert values[0, 15] == 2
    np.testing.assert_array_equal(relative_ms, [-200, -100, 0])


def test_sensor_window_rejects_noncausal_query() -> None:
    force = np.zeros((4, 5, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="causal"):
        assemble_sensor_window(force, force, np.arange(4), [0, 1, 2], 1)


def test_video_pool_rejects_wrong_layout_and_nonfinite_tokens() -> None:
    with pytest.raises(ValueError, match="VideoPrism"):
        pool_videoprism_tokens(np.zeros((1, 7, 768), dtype=np.float32))
    tokens = np.zeros((1, 8, 768), dtype=np.float32)
    tokens[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        pool_videoprism_tokens(tokens)
