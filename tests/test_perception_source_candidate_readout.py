import numpy as np
import pytest

from tiny_omni_decision.perception_source_candidate_readout import (
    build_feature_examples,
    pool_audio_features,
    pool_video_candidate_features,
)


def test_audio_pool_deduplicates_same_record_window_and_keeps_variants_separate():
    features = np.asarray([[1.0, 3.0], [5.0, 7.0], [9.0, 11.0]], dtype=np.float32)
    references = [
        {"record_id": "r1", "variant": "source", "feature_index": 0},
        {"record_id": "r1", "variant": "source", "feature_index": 0},
        {"record_id": "r1", "variant": "source", "feature_index": 1},
        {"record_id": "r1", "variant": "offset", "feature_index": 2},
    ]

    source = pool_audio_features(references, features, variant="source")
    offset = pool_audio_features(references, features, variant="offset")

    np.testing.assert_allclose(source["r1"], np.asarray([3.0, 5.0]) / np.sqrt(34.0))
    np.testing.assert_allclose(offset["r1"], np.asarray([9.0, 11.0]) / np.sqrt(202.0))


def test_video_pool_uses_temporal_mean_and_chronological_delta():
    records = [
        {
            "record_id": "r1",
            "candidates": [
                {
                    "candidate_id": "c1",
                    "frames": [{"frame_id": 20}, {"frame_id": 10}],
                }
            ],
        }
    ]
    index = [
        {"record_id": "r1", "candidate_id": "c1", "frame_id": 20, "feature_index": 1},
        {"record_id": "r1", "candidate_id": "c1", "frame_id": 10, "feature_index": 0},
    ]
    features = np.asarray([[1.0, 5.0], [4.0, 9.0]], dtype=np.float32)

    pooled = pool_video_candidate_features(records, index, features)

    np.testing.assert_allclose(
        pooled["r1"]["c1"],
        [2.5 / np.sqrt(55.25), 7.0 / np.sqrt(55.25), 0.6, 0.8],
    )


def test_build_examples_rejects_cross_split_or_missing_candidate_labels():
    row = {
        "record_id": "r1",
        "video_id": "v1",
        "audio_shuffle_donor_record_id": "r2",
        "candidates": [{"candidate_id": "c1"}],
    }
    labels = {
        "r1": {
            "request": {"candidates": [{"candidate_id": "c1"}]},
            "supervision": {"none_visible": False, "target_candidate_ids": ["wrong"]},
        }
    }

    with pytest.raises(ValueError, match="target candidates"):
        build_feature_examples(
            [row],
            labels,
            {"r1": {"c1": np.zeros(1536)}},
            {"r1": np.zeros(384)},
            {"r1": np.zeros(384)},
        )


def test_build_examples_keeps_unavailable_offset_control_missing():
    row = {
        "record_id": "r1",
        "video_id": "v1",
        "audio_shuffle_donor_record_id": "r2",
        "audio_shuffle_donor_video_id": "v2",
        "shifted_audio_windows": None,
        "candidates": [{"candidate_id": "c1"}],
    }
    labels = {
        "r1": {
            "request": {"candidates": [{"candidate_id": "c1"}]},
            "supervision": {"none_visible": False, "target_candidate_ids": ["c1"]},
        }
    }

    result = build_feature_examples(
        [row],
        labels,
        {"r1": {"c1": np.zeros(1536)}},
        {"r1": np.zeros(384)},
        {},
    )

    assert result[0].offset_audio is None
