from __future__ import annotations

import numpy as np

from scripts.physicalai_pilot_metrics import (
    episode_counts_by_split,
    expected_calibration_error,
    per_class_accuracy,
)


def test_per_class_accuracy_reports_support_and_missing_classes() -> None:
    probabilities = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.6, 0.3],
            [0.2, 0.3, 0.5],
        ]
    )
    targets = np.array([0, 1, 1])

    assert per_class_accuracy(probabilities, targets) == [
        {"class": 0, "support": 1, "accuracy": 1.0},
        {"class": 1, "support": 2, "accuracy": 0.5},
        {"class": 2, "support": 0, "accuracy": None},
    ]


def test_per_class_accuracy_rejects_probability_target_shape_mismatch() -> None:
    probabilities = np.ones((2, 3), dtype=np.float64) / 3

    try:
        per_class_accuracy(probabilities, np.array([0]))
    except ValueError as exc:
        assert "same number of rows" in str(exc)
    else:
        raise AssertionError("mismatched targets must be rejected")


def test_episode_counts_are_derived_from_ids_and_reject_group_overlap() -> None:
    counts = episode_counts_by_split(
        np.array([10, 10, 11, 11, 20, 20]),
        np.array([True, True, True, True, False, False]),
        np.array([False, False, False, False, True, True]),
    )
    assert counts == {"train": 2, "validation": 1}

    try:
        episode_counts_by_split(
            np.array([10, 10]), np.array([True, False]), np.array([False, True])
        )
    except ValueError as exc:
        assert "episode IDs overlap" in str(exc)
    else:
        raise AssertionError("group overlap must be rejected")


def test_ece_includes_confidence_exactly_one_in_final_bin() -> None:
    probabilities = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    targets = np.array([1, 1, 0])

    assert np.isclose(expected_calibration_error(probabilities, targets), 1 / 3)


def test_ece_uses_expected_confidence_bins() -> None:
    probabilities = np.array([[0.91, 0.09], [0.89, 0.11]])
    targets = np.array([0, 1])

    assert np.isclose(expected_calibration_error(probabilities, targets), 0.49)
