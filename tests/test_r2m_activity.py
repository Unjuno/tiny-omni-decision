import numpy as np
import pytest

from tiny_omni_decision.r2m_activity import (
    activity_metrics,
    supported_activity_vocabulary,
    train_sensor_normalization,
)


def test_vocabulary_requires_presence_in_every_declared_split():
    counts = {
        "train": {"walk": 10, "open": 4, "typing": 2},
        "calibration": {"walk": 2, "open": 1, "typing": 0},
        "validation": {"walk": 3, "open": 2, "typing": 0},
    }

    assert supported_activity_vocabulary(counts) == ("open", "walk")


def test_vocabulary_fails_when_fewer_than_two_classes_are_shared():
    counts = {
        "train": {"walk": 10, "open": 4},
        "calibration": {"walk": 2, "open": 0},
        "validation": {"walk": 3, "open": 2},
    }

    with pytest.raises(ValueError, match="at least two activities"):
        supported_activity_vocabulary(counts)


def test_activity_metrics_include_macro_scores_and_per_class_counts():
    vocabulary = ("close", "open", "walk")
    targets = np.array([0, 1, 2], dtype=np.int64)
    probabilities = np.eye(3, dtype=np.float64) * 0.8 + 0.2 / 3

    metrics = activity_metrics(probabilities, targets, vocabulary)

    assert metrics["examples"] == 3
    assert metrics["choices"] == 3
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_accuracy"] == 1.0
    assert set(metrics["per_activity"]) == set(vocabulary)
    assert all(row["examples"] == 1 for row in metrics["per_activity"].values())


def test_activity_metrics_reject_missing_validation_class():
    with pytest.raises(ValueError, match="has no evaluation examples"):
        activity_metrics(
            np.array([[0.7, 0.2, 0.1], [0.2, 0.7, 0.1]]),
            np.array([0, 1]),
            ("a", "b", "c"),
        )


def test_sensor_normalization_is_per_axis_and_derived_from_training_values():
    mean, standard_deviation = train_sensor_normalization([
        np.array([[1.0, 9.0, -2.0], [3.0, 11.0, 2.0]]),
    ])

    np.testing.assert_allclose(mean, [2.0, 10.0, 0.0])
    np.testing.assert_allclose(standard_deviation, [1.0, 1.0, 2.0])
