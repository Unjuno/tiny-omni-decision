from __future__ import annotations

import numpy as np
import pytest

from scripts.evaluate_physical_concepts_probe import (
    anomaly_probability,
    binary_metrics,
    bootstrap_group_accuracy,
    expected_calibration_error,
    fixed_validation_threshold,
    group_ranking_accuracy,
)


def test_fixed_anomaly_threshold_and_scale_use_validation_distribution() -> None:
    scores = np.arange(100, dtype=np.float64)
    threshold, scale = fixed_validation_threshold(scores)
    assert threshold == pytest.approx(94.05)
    assert scale > 0
    probabilities = anomaly_probability(
        np.asarray([threshold - scale, threshold, threshold + scale]), threshold, scale
    )
    assert probabilities[0] == pytest.approx(1 / (1 + np.e))
    assert probabilities[1] == pytest.approx(0.5)
    assert probabilities[2] == pytest.approx(1 / (1 + np.exp(-1)))


def test_ece_and_binary_metrics_report_separated_balanced_classes() -> None:
    labels = np.asarray([0, 0, 1, 1], dtype=np.int8)
    scores = np.asarray([0.1, 0.2, 0.8, 0.9])
    probabilities = scores.copy()
    metrics = binary_metrics(labels, scores, probabilities)
    assert metrics["accuracy"] == 1.0
    assert metrics["sensitivity"] == 1.0
    assert metrics["specificity"] == 1.0
    assert metrics["auc"] == 1.0
    assert metrics["nll"] < 0.3
    assert expected_calibration_error(labels, probabilities) > 0


def test_group_ranking_accuracy_credits_ties_half() -> None:
    scores = np.asarray(
        [
            [0.1, 0.2, 0.4, 0.5],
            [0.4, 0.5, 0.1, 0.2],
            [0.0, 0.5, 0.125, 0.375],
        ]
    )
    assert group_ranking_accuracy(scores).tolist() == [1.0, 0.0, 0.5]


def test_group_bootstrap_is_deterministic_and_requires_all_categories() -> None:
    data = {
        name: np.tile([0.1, 0.2, 0.3, 0.4], (12, 1))
        for name in (
            "continuity",
            "solidity",
            "object_persistence",
            "unchangeableness",
            "directional_inertia",
        )
    }
    first = bootstrap_group_accuracy(data, seed=17, iterations=200)
    second = bootstrap_group_accuracy(data, seed=17, iterations=200)
    assert first == second
    assert first["macro_group_accuracy"] == 1.0
    with pytest.raises(ValueError, match="all five"):
        bootstrap_group_accuracy({"continuity": data["continuity"]}, iterations=200)


def test_threshold_and_metrics_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="finite one-dimensional"):
        fixed_validation_threshold(np.asarray([np.nan] * 25))
    with pytest.raises(ValueError, match="scale positive"):
        anomaly_probability(np.asarray([1.0]), 0.0, 0.0)
    with pytest.raises(ValueError, match="both binary classes"):
        binary_metrics(np.asarray([1, 1]), np.asarray([0.1, 0.2]), np.asarray([0.8, 0.9]))
