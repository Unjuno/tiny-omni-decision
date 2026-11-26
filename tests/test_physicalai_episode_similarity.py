from __future__ import annotations

import numpy as np

from scripts.audit_physicalai_episode_similarity import audit_episode_similarity


def test_audit_reports_cross_split_neighbors_without_claiming_identity(tmp_path):
    features_path = tmp_path / "features.npz"
    np.savez_compressed(
        features_path,
        episode=np.array([1] * 4 + [2] * 4 + [3] * 4),
        split=np.array(["train"] * 8 + ["validation"] * 4),
        video=np.array([[1.0, 0.0]] * 4 + [[0.0, 1.0]] * 4 + [[1.0, 0.0]] * 4),
    )

    result = audit_episode_similarity(features_path)

    assert result["cross_split_pair_count"] == 2
    assert result["cross_split_cosine"]["max"] == 1.0
    assert result["cross_split_cosine"]["pairs_at_or_above_threshold"]["0.99"] == 1
    assert result["top_20_cross_split_neighbors"][0]["train_episode"] == 1
    assert "not a scene identity" in result["interpretation_limit"]


def test_audit_rejects_episode_rows_split_across_partitions(tmp_path):
    features_path = tmp_path / "features.npz"
    np.savez_compressed(
        features_path,
        episode=np.array([1] * 4),
        split=np.array(["train", "train", "validation", "validation"]),
        video=np.array([[1.0, 0.0]] * 4),
    )

    try:
        audit_episode_similarity(features_path)
    except ValueError as exc:
        assert "multiple splits" in str(exc)
    else:
        raise AssertionError("an episode crossing splits must be rejected")
