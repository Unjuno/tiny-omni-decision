import numpy as np
import pytest

from tiny_omni_decision.droid_next_action_readout import (
    ACTION_OPTIONS,
    fit_l2_softmax,
    make_readout_features,
    multiclass_probability_metrics,
    next_cartesian_direction_label,
    predict_softmax,
    select_l2_strength_group_cv,
    stratified_group_folds,
)


@pytest.mark.parametrize(
    ("velocity", "expected"),
    [
        ([0, 0, 0, 0, 0, 0], 0),
        ([0.2, 0, 0, 0, 0, 0], 2),
        ([-0.3, 0, 0, 0, 0, 0], 1),
        ([0, 0.4, 0, 0, 0, 0], 4),
        ([0, -0.5, 0, 0, 0, 0], 3),
        ([0, 0, 0.6, 0, 0, 0], 6),
        ([0, 0, -0.7, 0, 0, 0], 5),
        ([1e-4, 0, 0, 0, 0, 0], 0),
        ([0.1, -0.3, 0, 0, 0, 0], 3),
    ],
)
def test_next_cartesian_direction_label(velocity, expected):
    assert next_cartesian_direction_label(np.asarray(velocity)) == expected


def test_next_cartesian_direction_rejects_malformed_input():
    with pytest.raises(ValueError):
        next_cartesian_direction_label(np.zeros(5))
    with pytest.raises(ValueError):
        next_cartesian_direction_label(np.asarray([0, 0, np.nan, 0, 0, 0]))


def test_feature_summaries_keep_video_and_sensor_paths_separate():
    video = np.arange(2 * 3 * 8 * 2, dtype=np.float64).reshape(2, 3, 8, 2)
    sensor = np.arange(2 * 8 * 14, dtype=np.float64).reshape(2, 8, 14)
    features = make_readout_features(video, sensor)
    assert features["video"].shape == (2, 4)
    assert features["sensor"].shape == (2, 112)
    assert features["video_sensor"].shape == (2, 116)
    np.testing.assert_array_equal(features["video_sensor"][:, :4], features["video"])
    np.testing.assert_array_equal(features["video_sensor"][:, 4:], features["sensor"])


def test_multiclass_readout_fits_and_saves_probabilities():
    pytest.importorskip("scipy")
    rng = np.random.default_rng(17)
    targets = np.tile(np.arange(len(ACTION_OPTIONS)), 8)
    features = rng.normal(size=(len(targets), 4))
    features[:, 0] += targets
    model = fit_l2_softmax(features, targets)
    probabilities = predict_softmax(model, features)
    assert probabilities.shape == (len(targets), len(ACTION_OPTIONS))
    assert np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    metrics = multiclass_probability_metrics(targets, probabilities)
    assert metrics["count"] == len(targets)
    assert set(metrics["per_class"]) == set(ACTION_OPTIONS)
    assert metrics["accuracy"] > 0.5


def test_metrics_reject_non_normalized_probabilities():
    with pytest.raises(ValueError):
        multiclass_probability_metrics(np.asarray([0]), np.asarray([[0.8, 0.8, 0, 0, 0, 0, 0]]))


def test_stratified_group_folds_are_deterministic_and_keep_groups_together():
    targets = np.repeat(np.arange(len(ACTION_OPTIONS)), 8)
    group_ids = [f"g{label}-{index // 2}" for label in range(7) for index in range(8)]
    first = stratified_group_folds(targets, group_ids, folds=4, seed=17)
    second = stratified_group_folds(targets, group_ids, folds=4, seed=17)
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left, right)
    locations = {}
    for fold_id, indices in enumerate(first):
        for index in indices:
            group = group_ids[int(index)]
            assert group not in locations or locations[group] == fold_id
            locations[group] = fold_id
    assert len(locations) == len(set(group_ids))


def test_group_cv_selects_only_from_fixed_candidates():
    pytest.importorskip("scipy")
    rng = np.random.default_rng(17)
    targets = np.repeat(np.arange(len(ACTION_OPTIONS)), 5)
    features = rng.normal(size=(len(targets), 4))
    features[:, 0] += targets
    selection = select_l2_strength_group_cv(
        features,
        targets,
        [f"episode-{index}" for index in range(len(targets))],
        candidates=(0.1, 10.0),
        folds=5,
        seed=17,
    )
    assert selection["selected_l2_strength"] in (0.1, 10.0)
    assert set(selection["candidate_metrics"]) == {"0.1", "10.0"}
    assert selection["selection_metric"] == "out-of-fold macro NLL"
