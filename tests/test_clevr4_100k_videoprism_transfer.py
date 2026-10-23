from __future__ import annotations

import math

import numpy as np
import pytest

from scripts.evaluate_clevr4_100k_videoprism_transfer import ece, metrics


def test_metrics_reports_accuracy_nll_brier_ece_and_confidence() -> None:
    probabilities = np.asarray([[0.8, 0.2], [0.2, 0.8]], dtype=np.float64)
    targets = np.asarray([0, 1], dtype=np.int64)

    result = metrics(probabilities, targets)

    assert result["count"] == 2
    assert result["accuracy"] == 1.0
    assert result["nll"] == pytest.approx(-math.log(0.8))
    assert result["brier"] == pytest.approx(0.08)
    assert result["ece_15_equal_width"] == pytest.approx(0.2)
    assert result["mean_confidence"] == pytest.approx(0.8)


def test_ece_is_zero_for_calibrated_confidence_bin() -> None:
    probabilities = np.asarray([[0.8, 0.2]] * 5, dtype=np.float64)
    targets = np.asarray([0, 0, 0, 0, 1], dtype=np.int64)

    assert ece(probabilities, targets) == pytest.approx(0.0)
