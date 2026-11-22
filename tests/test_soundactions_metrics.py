import numpy as np
import pytest

from tiny_omni_decision.soundactions_metrics import (
    multiclass_metrics,
    paired_bootstrap_difference,
)


def test_multiclass_metrics_report_calibration_and_class_counts():
    probabilities = np.asarray([[0.8, 0.1, 0.1], [0.1, 0.7, 0.2], [0.1, 0.2, 0.7]])
    result = multiclass_metrics(probabilities, np.asarray([0, 1, 2]))
    assert result["accuracy"] == 1.0
    assert result["macro_accuracy"] == 1.0
    assert result["nll"] == pytest.approx(-(np.log(0.8) + 2 * np.log(0.7)) / 3)
    assert result["per_class"]["1"]["count"] == 1
    assert result["ece"] == pytest.approx((0.2 + 0.3 + 0.3) / 3)


def test_multiclass_metrics_reject_invalid_probabilities():
    with pytest.raises(ValueError, match="sum to one"):
        multiclass_metrics(np.asarray([[0.2, 0.2, 0.2]]), np.asarray([0]))


def test_recording_bootstrap_is_seeded_and_directional():
    targets = np.asarray([0, 0, 1, 1, 2, 2])
    weaker = np.asarray([[0.4, 0.3, 0.3], [0.4, 0.3, 0.3]] * 3)
    stronger = np.eye(3)[targets] * 0.8 + 0.2 / 3
    first = paired_bootstrap_difference(
        weaker, stronger, targets, metric="accuracy", iterations=200
    )
    second = paired_bootstrap_difference(
        weaker, stronger, targets, metric="accuracy", iterations=200
    )
    assert first == second
    assert first["mean_improvement"] > 0
    assert first["resampling"] == "paired, stratified by target class"
