import numpy as np
import pytest

pytest.importorskip("torch")
from scripts.train_droid_dense_change_weighted_cv import (  # noqa: E402
    change_metrics,
    correct_probabilities,
    training_row_weights,
    training_weights,
)


def test_train_weights_balance_stable_and_changed_mass():
    previous = np.array([0, 1, 1, 2])
    targets = np.array([0, 1, 2, 2])
    stable, changed = training_weights(previous, targets)
    assert stable * 3 == changed
    assert (stable * 3 + changed) / 4 == 1.0
    with pytest.raises(ValueError, match="stable and changed"):
        training_weights([0, 1], [0, 1])


def test_training_row_weights_never_read_held_target_values():
    previous = np.array([0, 1, 1, 2])
    targets = np.array([0, 1, 2, 2])
    train = np.array([0, 1, 2])
    first = training_row_weights(previous, targets, train)
    modified = targets.copy()
    modified[3] = 6
    second = training_row_weights(previous, modified, train)
    assert np.array_equal(first[0], second[0])
    assert first[0][3] == 0
    assert first[1:] == second[1:]


def test_probability_correction_recovers_exact_reweighted_distribution():
    original = np.array([[0.5, 0.25, 0.05, 0.05, 0.05, 0.05, 0.05]])
    weights = np.array([[0.5, 5, 5, 5, 5, 5, 5]])
    weighted = original * weights
    weighted /= weighted.sum(axis=1, keepdims=True)
    recovered = correct_probabilities(weighted, [0], 0.5, 5.0)
    assert np.allclose(recovered, original)
    with pytest.raises(ValueError, match="weights"):
        correct_probabilities(weighted, [0], float("nan"), 5.0)


def test_change_detection_distinguishes_detection_from_exact_target():
    previous = np.array([0, 1, 2])
    targets = np.array([0, 2, 3])
    predictions = np.array([1, 3, 3])
    result = change_metrics(previous, targets, predictions)
    assert result["true_change_detected"] == 2
    assert result["changed_exact_correct"] == 1
    assert result["stable_false_change_rate"] == 1.0
    assert result["change_precision"] == 2 / 3
