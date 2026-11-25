import numpy as np

from scripts.fit_robopro_action_readouts import (
    MODE_STREAMS,
    _build_examples,
    _fit_normalizers,
    _group_metrics,
    _metrics,
    _normalizer,
    _paired_bootstrap,
)


def test_multiclass_metrics_include_brier_and_expected_calibration():
    probabilities = np.asarray([[0.8, 0.2], [0.1, 0.9]], dtype=np.float32)
    labels = np.asarray([0, 1])

    metrics = _metrics(probabilities, labels)

    assert metrics["count"] == 2
    assert metrics["accuracy"] == 1.0
    assert metrics["nll"] > 0
    assert metrics["brier"] > 0
    assert metrics["ece_15"] > 0


def test_prompt_macro_metrics_weight_groups_equally():
    probabilities = np.asarray(
        [[0.9, 0.1], [0.9, 0.1], [0.1, 0.9], [0.1, 0.9], [0.1, 0.9]],
        dtype=np.float32,
    )
    labels = np.asarray([0, 0, 0, 0, 0])
    groups = np.asarray(["a", "a", "b", "b", "b"])

    metrics = _group_metrics(probabilities, labels, groups)

    assert metrics["per_prompt"]["a"]["accuracy"] == 1.0
    assert metrics["per_prompt"]["b"]["accuracy"] == 0.0
    assert metrics["macro_by_prompt"]["accuracy"] == 0.5


def test_normalizer_is_train_fittable_and_handles_constant_dimensions():
    values = np.asarray([[1.0, 7.0], [3.0, 7.0]], dtype=np.float32)

    mean, std = _normalizer(values, (0,))

    np.testing.assert_allclose(mean.squeeze(), [2.0, 7.0])
    assert std.squeeze()[0] > 0
    assert std.squeeze()[1] >= 1e-4


def test_group_bootstrap_uses_paired_prompt_clusters_and_fixed_seed():
    left = np.asarray([0, 0, 1, 1], dtype=np.float32)
    right = np.asarray([1, 1, 1, 1], dtype=np.float32)
    groups = np.asarray(["a", "a", "b", "b"])

    first = _paired_bootstrap(left, right, groups, replicates=100, seed=17)
    second = _paired_bootstrap(left, right, groups, replicates=100, seed=17)

    assert first == second
    assert first["groups"] == 2
    assert first["mean_group_difference"] == 0.5
    assert first["percentile_95_ci"] == [0.0, 1.0]


def test_build_examples_supports_sample_sizes_above_legacy_fixture(tmp_path):
    episode_count = 8
    splits = np.asarray(["train"] * 4 + ["development"] * 4)
    prompt_hashes = np.asarray(["train-prompt"] * 4 + ["development-prompt"] * 4)
    candidate_indices = np.tile(np.arange(4), 2)
    action_options = np.zeros((2, 3, 4, 14), dtype=np.float32)
    action_options[0] = np.arange(4, dtype=np.float32)[None, :, None]
    action_options[1] = 10 + np.arange(4, dtype=np.float32)[None, :, None]
    target_actions = np.empty((episode_count, 3, 14), dtype=np.float32)
    for episode_index in range(episode_count):
        split_index = 0 if splits[episode_index] == "train" else 1
        target_actions[episode_index] = action_options[split_index][
            :, candidate_indices[episode_index]
        ]

    feature_path = tmp_path / "features.npz"
    np.savez_compressed(
        feature_path,
        temporal_video_features=np.zeros((episode_count, 3, 8, 768), dtype=np.float32),
        pooled_video_features=np.zeros((episode_count, 3, 768), dtype=np.float32),
        aligned_state_windows=np.zeros((episode_count, 3, 8, 14), dtype=np.float32),
        target_action_vectors=target_actions,
        episode_ids=np.arange(episode_count),
        split=splits,
        group_index=np.zeros(episode_count, dtype=np.int64),
        candidate_index=candidate_indices,
        prompt_sha256=prompt_hashes,
        action_options=action_options,
        relative_progress=np.asarray([0.25, 0.5, 0.75], dtype=np.float32),
    )
    text_path = tmp_path / "text.npz"
    np.savez_compressed(
        text_path,
        features=np.zeros((2, 384), dtype=np.float32),
        prompt_sha256=np.asarray(["train-prompt", "development-prompt"]),
        split=np.asarray(["train", "development"]),
    )

    examples, options = _build_examples(feature_path, text_path)

    assert examples["train"]["labels"].shape == (12,)
    assert examples["development"]["labels"].shape == (12,)
    assert options.shape == (2, 3, 4, 14)
    assert examples["development"]["actions"][0, 0, 0] == 10


def test_action_normalizer_uses_only_train_prompt_groups():
    train = {
        "groups": np.asarray(["train-a"] * 3 + ["train-b"] * 3),
        "text": np.zeros((6, 384), dtype=np.float32),
        "state": np.zeros((6, 8, 14), dtype=np.float32),
        "video_sequence": np.zeros((6, 8, 768), dtype=np.float32),
    }
    options = np.zeros((4, 3, 4, 14), dtype=np.float32)
    options[2:] = 1000

    normalizers = _fit_normalizers(train, options)

    assert np.all(normalizers["actions"][0] == 0)


def test_state_video_temporal_arm_uses_state_and_video_streams():
    assert MODE_STREAMS["state_video_gru"] == ["state", "video"]
