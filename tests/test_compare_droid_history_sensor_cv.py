import numpy as np
import pytest

from scripts.compare_droid_history_sensor_cv import fit_folds, inputs


def test_inputs_preserve_history_and_expected_dimensions():
    sensor = np.arange(2 * 8 * 14, dtype=float).reshape(2, 8, 14)
    arms = inputs(sensor, [0, 6])
    assert arms["history"].shape == (2, 7)
    assert arms["history_sensor"].shape == (2, 119)
    assert np.array_equal(arms["history"], arms["history_sensor"][:, :7])
    with pytest.raises(ValueError, match="previous-command"):
        inputs(sensor, [0, 7])


def test_fit_folds_excludes_held_groups_and_covers_every_row():
    arms = {"history": np.arange(4, dtype=float).reshape(4, 1)}
    targets = np.array([0, 1, 2, 3])
    groups = np.array(["a", "b", "c", "d"])
    folds = [{"train_indices": [2, 3], "held_indices": [0, 1]},
             {"train_indices": [0, 1], "held_indices": [2, 3]}]
    calls = []

    def select(x, y, group_ids, **kwargs):
        calls.append(set(group_ids))
        return {"selected_l2_strength": 100.0}

    api = {"select_l2_strength_group_cv": select,
           "fit_l2_softmax": lambda x, y, **kwargs: {"converged": True},
           "predict_softmax": lambda model, x: np.full((len(x), 7), 1 / 7)}
    outputs, models, details = fit_folds(arms, targets, groups, folds, api)
    assert calls == [{"c", "d"}, {"a", "b"}]
    assert outputs["history"].shape == (4, 7)
    assert len(models) == len(details) == 2
    folds[0]["train_indices"] = [0, 2, 3]
    with pytest.raises(ValueError, match="fold overlap"):
        fit_folds(arms, targets, groups, folds, api)
