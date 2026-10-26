import numpy as np
import pytest

from tiny_omni_decision.droid_early_outcome_readout import (
    binary_probability_metrics,
    fit_l2_logistic,
    predict_success_probability,
)


def test_binary_probability_metrics_include_balanced_accuracy_and_class_recall():
    result = binary_probability_metrics(np.asarray([0, 0, 1, 1]), np.asarray([0.1, 0.8, 0.7, 0.9]))
    assert result["count"] == 4
    assert result["accuracy"] == 0.75
    assert result["balanced_accuracy"] == 0.75
    assert result["failure_recall"] == 0.5
    assert result["success_recall"] == 1.0
    assert 0 <= result["ece"] <= 1


def test_binary_probability_metrics_fail_closed_on_invalid_inputs():
    with pytest.raises(ValueError, match="lie in \\[0, 1\\]"):
        binary_probability_metrics(np.asarray([0, 1]), np.asarray([0.2, 1.1]))
    with pytest.raises(ValueError, match="both target classes"):
        binary_probability_metrics(np.asarray([1, 1]), np.asarray([0.2, 0.8]))
    with pytest.raises(ValueError, match="finite"):
        binary_probability_metrics(np.asarray([0, 1]), np.asarray([0.2, np.nan]))


def test_l2_logistic_fit_is_deterministic_and_reloadable():
    pytest.importorskip("scipy")
    features = np.asarray([[-2.0, 0.1], [-1.0, 0.2], [1.0, -0.1], [2.0, -0.2]])
    targets = np.asarray([0, 0, 1, 1])
    first = fit_l2_logistic(features, targets)
    second = fit_l2_logistic(features, targets)
    np.testing.assert_array_equal(first["weight"], second["weight"])
    np.testing.assert_array_equal(first["mean"], second["mean"])
    probabilities = predict_success_probability(first, features)
    np.testing.assert_allclose(probabilities, predict_success_probability(second, features))
    assert binary_probability_metrics(targets, probabilities)["accuracy"] == 1.0


def test_l2_logistic_rejects_one_class_training_data():
    with pytest.raises(ValueError, match="both binary classes"):
        fit_l2_logistic(np.ones((3, 2)), np.ones(3))
