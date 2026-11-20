import numpy as np
import pytest

from tiny_omni_decision.r2m_activity import (
    activity_metrics,
    same_participant_different_activity_sensor_donors,
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


def test_sensor_donors_are_deterministic_same_participant_and_wrong_activity():
    rows = [
        {"sample_id": f"p{person}-{label}-{index}", "participant_id": person,
         "activity": label}
        for person in (9, 15)
        for label, count in (("a", 3), ("b", 2), ("c", 1))
        for index in range(count)
    ]

    donors = same_participant_different_activity_sensor_donors(rows, seed=17)
    repeated = same_participant_different_activity_sensor_donors(rows, seed=17)

    assert [row["sample_id"] for row in donors] == [
        row["sample_id"] for row in repeated
    ]
    assert {row["sample_id"] for row in donors} == {row["sample_id"] for row in rows}
    for recipient, donor in zip(rows, donors, strict=True):
        assert donor["participant_id"] == recipient["participant_id"]
        assert donor["activity"] != recipient["activity"]


def test_sensor_donor_derangement_fails_when_activity_dominates_group():
    rows = [
        {"sample_id": f"{label}-{index}", "participant_id": 9, "activity": label}
        for label, count in (("common", 3), ("rare", 1))
        for index in range(count)
    ]

    with pytest.raises(ValueError, match="lacks a full different-activity"):
        same_participant_different_activity_sensor_donors(rows)
