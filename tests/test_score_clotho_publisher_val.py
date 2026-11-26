import numpy as np
import pytest

from scripts.score_clotho_publisher_val import build_features, paired_cluster_bootstrap


def test_question_only_features_zero_audio_and_append_option_identity():
    audio = np.ones((2, 384), dtype=np.float32)
    question = np.arange(768, dtype=np.float32).reshape(2, 384)

    features = build_features(audio, question, "question_only")

    assert features.shape == (2, 2, 770)
    assert np.count_nonzero(features[:, :, :384]) == 0
    assert np.array_equal(features[0, 0, 384:768], features[0, 1, 384:768])
    assert np.array_equal(features[0, :, 768:], np.eye(2))


def test_joint_features_preserve_audio_question_and_option_identity():
    audio = np.ones((1, 384), dtype=np.float32)
    question = np.full((1, 384), 2, dtype=np.float32)

    features = build_features(audio, question, "audio_question")

    assert np.all(features[0, :, :384] == 1)
    assert np.all(features[0, :, 384:768] == 2)
    assert np.array_equal(features[0, :, 768:], np.eye(2))


def test_paired_bootstrap_keeps_clusters_and_is_deterministic():
    targets = np.array([0, 1, 0, 1], dtype=np.int64)
    base = np.array([0.2, 0.8, 0.3, 0.7])
    better = np.array([0.1, 0.9, 0.2, 0.8])
    clusters = ["u1", "u1", "u2", "u2"]

    first = paired_cluster_bootstrap(
        targets, {"base": base, "candidate": better}, clusters, replicates=50, seed=17
    )
    second = paired_cluster_bootstrap(
        targets, {"base": base, "candidate": better}, clusters, replicates=50, seed=17
    )

    assert first == second
    assert first["cluster_count"] == 2
    assert first["delta_order"] == "candidate minus base"
    assert first["deltas"]["nll"]["point"] < 0


def test_paired_bootstrap_rejects_single_cluster():
    with pytest.raises(ValueError, match="at least two"):
        paired_cluster_bootstrap(
            np.array([0, 1]),
            {"base": np.array([0.2, 0.8]), "candidate": np.array([0.3, 0.7])},
            ["same", "same"],
        )
