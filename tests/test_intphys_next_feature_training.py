import numpy as np
import pytest

from scripts.train_intphys_next_feature_predictor import (
    load_feature_split,
    validate_group_disjointness,
)


def test_feature_split_loads_finite_sequences_and_string_identities(tmp_path):
    path = tmp_path / "features.npz"
    np.savez_compressed(
        path,
        features=np.zeros((2, 8, 4), dtype=np.float32),
        sample_ids=np.asarray(["video-1", "video-2"]),
        group_ids=np.asarray(["scene-1", "scene-2"]),
    )

    features, sample_ids, group_ids = load_feature_split(path)

    assert features.shape == (2, 8, 4)
    assert sample_ids == ["video-1", "video-2"]
    assert group_ids == ["scene-1", "scene-2"]


@pytest.mark.parametrize(
    "features,sample_ids,group_ids",
    [
        (np.zeros((2, 8, 4), dtype=np.float32), ["a", "a"], ["g1", "g2"]),
        (np.zeros((2, 8, 4), dtype=np.float32), ["a", "b"], ["g1"]),
        (np.full((2, 8, 4), np.nan, dtype=np.float32), ["a", "b"], ["g1", "g2"]),
        (np.zeros((2, 1, 4), dtype=np.float32), ["a", "b"], ["g1", "g2"]),
    ],
)
def test_feature_split_rejects_invalid_sequences(tmp_path, features, sample_ids, group_ids):
    path = tmp_path / "invalid.npz"
    np.savez_compressed(
        path,
        features=features,
        sample_ids=np.asarray(sample_ids),
        group_ids=np.asarray(group_ids),
    )

    with pytest.raises(ValueError):
        load_feature_split(path)


def test_group_disjointness_rejects_shared_sample_or_underlying_scene():
    validate_group_disjointness(["a"], ["scene-a"], ["b"], ["scene-b"])

    with pytest.raises(ValueError, match="sample IDs overlap"):
        validate_group_disjointness(["a"], ["scene-a"], ["a"], ["scene-b"])
    with pytest.raises(ValueError, match="group IDs overlap"):
        validate_group_disjointness(["a"], ["scene-a"], ["b"], ["scene-a"])
