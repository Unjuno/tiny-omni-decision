from __future__ import annotations

import numpy as np
import pytest

from scripts.build_egoviz_vs_sync_window_corpus import (
    WINDOW_NS,
    choose_frame_indices,
    make_candidates,
    resample_imu,
)
from scripts.extract_egoviz_vs_sync_sensor_features import CHECKPOINT_SHA256
from scripts.extract_egoviz_vs_sync_video_features import temporal_spatial_pool


def test_video_frame_selection_is_uniform_deterministic_and_bounded():
    times = list(range(100))
    chosen = choose_frame_indices(times, 10, 30)

    assert chosen == choose_frame_indices(times, 10, 30)
    assert len(chosen) == 8
    assert chosen == sorted(set(chosen))
    assert all(10 <= index < 30 for index in chosen)


def test_video_frame_selection_rejects_insufficient_frames():
    assert choose_frame_indices(list(range(7)), 0, 7) == []


def test_imu_resampling_has_expected_shape_and_linear_values():
    timestamps = list(range(0, WINDOW_NS + 1, 10_000_000))
    values = np.repeat(np.asarray(timestamps, dtype=np.float64)[:, None], 6, axis=1)

    result = resample_imu(timestamps, values, 0)

    assert result.shape == (6, 128)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result[:, 0], 0)
    np.testing.assert_allclose(result[:, -1], (127 * WINDOW_NS / 128), rtol=1e-6)


def test_imu_resampling_rejects_bad_shape_order_and_coverage():
    with pytest.raises(ValueError, match="shape"):
        resample_imu([0, 1], np.zeros((2, 5)), 0)
    with pytest.raises(ValueError, match="strictly increasing"):
        resample_imu([0, 0], np.zeros((2, 6)), 0)
    with pytest.raises(ValueError, match="cover"):
        resample_imu([0, 1], np.zeros((2, 6)), 0)


def test_candidate_sampling_is_deterministic_unique_and_contains_match():
    first = make_candidates("chunk", query_index=4, count=12, seed=17)
    second = make_candidates("chunk", query_index=4, count=12, seed=17)
    options, target = first

    assert first == second
    assert len(options) == len(set(options)) == 4
    assert options[target] == 4
    assert all(abs(option - 4) >= 3 for index, option in enumerate(options) if index != target)


def test_candidate_sampling_requires_three_distractors():
    with pytest.raises(ValueError, match="three"):
        make_candidates("chunk", query_index=0, count=3)


def test_temporal_spatial_pool_preserves_frame_order():
    tokens = np.arange(1 * 8 * 2 * 768, dtype=np.float32).reshape(1, 8 * 2, 768)

    result = temporal_spatial_pool(tokens)

    assert result.shape == (8, 768)
    np.testing.assert_array_equal(result[0], tokens.reshape(1, 8, 2, 768).mean(axis=2)[0, 0])
    assert np.all(result[1] > result[0])


def test_temporal_spatial_pool_rejects_unexpected_token_shape():
    with pytest.raises(ValueError, match="shape"):
        temporal_spatial_pool(np.zeros((1, 15, 768), dtype=np.float32))


def test_pinned_sensor_feature_extractor_hash_is_sha256():
    assert len(CHECKPOINT_SHA256) == 64
    assert int(CHECKPOINT_SHA256, 16) >= 0
