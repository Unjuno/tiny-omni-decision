import numpy as np
import pytest
import torch

import scripts.train_droid_dense_history_sensor_cv as trainer
from scripts.train_droid_dense_history_sensor_cv import group_accuracy, sensor_deltas


def test_sensor_transform_keeps_last_state_and_all_increments():
    sensor = np.arange(2 * 8 * 14, dtype=np.float32).reshape(2, 8, 14)
    result = sensor_deltas(sensor)
    assert result.shape == (2, 112)
    assert np.array_equal(result[:, :14], sensor[:, -1])
    assert np.array_equal(result[:, 14:], np.diff(sensor, axis=1).reshape(2, 98))
    with pytest.raises(ValueError, match="finite"):
        sensor_deltas(np.full((1, 8, 14), np.nan))


def test_recording_accuracy_gives_each_group_equal_weight():
    group = np.arange(275)
    target = np.zeros(275, dtype=int)
    predicted = np.zeros(275, dtype=int)
    predicted[0] = 1
    assert group_accuracy(target, predicted, group) == 274 / 275
    with pytest.raises(ValueError, match="missing group"):
        group_accuracy(target[:-1], predicted[:-1], group[:-1])


def test_training_is_deterministic_and_ignores_held_rows(monkeypatch):
    monkeypatch.setattr(trainer, "EPOCHS", 2)
    torch.set_num_threads(1)
    features = np.eye(7, dtype=np.float32)[:4]
    targets = np.array([0, 1, 2, 3])
    train = np.array([0, 1, 2])
    first, first_curve = trainer.fit_arm(features, targets, train, 0, "history")
    changed = features.copy()
    changed[3] = 42
    second, second_curve = trainer.fit_arm(changed, targets, train, 0, "history")
    assert np.array_equal(first.weight.detach().numpy(), second.weight.detach().numpy())
    assert first_curve == second_curve
