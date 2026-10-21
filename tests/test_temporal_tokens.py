import numpy as np
import pytest

from scripts.extract_perception_temporal_match_videoprism import completed_run_metadata
from tiny_omni_decision.temporal_tokens import (
    pool_spatial_tokens_per_frame,
    reshape_videoprism_temporal_tokens,
    videoprism_features_by_layout,
)


def test_video_prism_tokens_reshape_preserves_frame_and_spatial_order():
    tokens = np.arange(1 * 3 * 2 * 1, dtype=np.float32).reshape(1, 6, 1)

    shaped = reshape_videoprism_temporal_tokens(tokens, frame_count=3)

    assert shaped.shape == (1, 3, 2, 1)
    np.testing.assert_array_equal(shaped[0, :, :, 0], [[0, 1], [2, 3], [4, 5]])


def test_frame_spatial_pool_keeps_time_axis():
    tokens = np.arange(1 * 3 * 2 * 2, dtype=np.float32).reshape(1, 6, 2)

    pooled = pool_spatial_tokens_per_frame(tokens, frame_count=3)

    assert pooled.shape == (1, 3, 2)
    np.testing.assert_allclose(pooled[0], [[1, 2], [5, 6], [9, 10]])


def test_feature_layouts_keep_legacy_pool_and_offer_temporal_outputs():
    tokens = np.arange(1 * 6 * 2, dtype=np.float32).reshape(1, 6, 2)

    legacy = videoprism_features_by_layout(tokens, frame_count=3, layout="pooled")
    per_frame = videoprism_features_by_layout(tokens, frame_count=3, layout="per_frame_mean")
    full = videoprism_features_by_layout(tokens, frame_count=3, layout="spatiotemporal_tokens")

    np.testing.assert_array_equal(legacy, tokens.mean(axis=1))
    np.testing.assert_allclose(per_frame[0], [[1, 2], [5, 6], [9, 10]])
    assert full.shape == (1, 3, 2, 2)
    np.testing.assert_array_equal(full.reshape(1, 6, 2), tokens)


@pytest.mark.parametrize(
    ("tokens", "frame_count", "message"),
    [
        (np.zeros((8, 768)), 8, "shape"),
        (np.zeros((1, 7, 768)), 8, "divisible"),
        (np.zeros((1, 8, 768)), 0, "positive"),
        (np.zeros((1, 0, 768)), 8, "nonempty"),
        (np.zeros((0, 8, 768)), 8, "nonempty"),
        (np.full((1, 8, 2), np.nan), 2, "finite"),
    ],
)
def test_video_prism_temporal_tokens_reject_invalid_inputs(tokens, frame_count, message):
    with pytest.raises(ValueError, match=message):
        reshape_videoprism_temporal_tokens(tokens, frame_count)


def test_video_prism_feature_conversion_rejects_unknown_layout():
    with pytest.raises(ValueError, match="unsupported"):
        videoprism_features_by_layout(np.zeros((1, 8, 2)), 2, "unknown")


def test_completed_run_metadata_closes_running_state_with_result_receipt():
    initial = {"status": "RUNNING", "manifest_sha256": "manifest"}
    result = {
        "status": "PASS_TEMPORAL_MATCH_VIDEOPRISM_FEATURES",
        "examples": 1,
        "elapsed_seconds": 8.5,
        "feature_layout": "per_frame_mean",
    }

    completed = completed_run_metadata(initial, result, result_sha256="result-hash")

    assert completed == {
        "status": "PASS_TEMPORAL_MATCH_VIDEOPRISM_FEATURES",
        "manifest_sha256": "manifest",
        "result_sha256": "result-hash",
        "examples": 1,
        "elapsed_seconds": 8.5,
        "feature_layout": "per_frame_mean",
    }
