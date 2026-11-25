import numpy as np

from scripts.fit_robopro_action_readouts import (
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
