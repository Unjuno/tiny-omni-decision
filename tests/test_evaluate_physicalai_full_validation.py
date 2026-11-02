from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from scripts.evaluate_physicalai_full_validation import evaluate, metrics


def test_metrics_calculates_multiclass_probability_scores():
    probabilities = np.array([[0.9, 0.1], [0.2, 0.8]], dtype=np.float64)
    targets = np.array([0, 1])

    result = metrics(probabilities, targets)

    assert result["accuracy"] == 1.0
    assert result["macro_accuracy_present_classes"] == 1.0
    assert result["nll"] == pytest.approx(-(math.log(0.9) + math.log(0.8)) / 2)
    assert result["brier"] == pytest.approx(0.05)
    assert result["n"] == 2


def test_full_validation_evaluator_fails_closed_below_200_groups(tmp_path: Path):
    validation_dir = tmp_path / "validation"
    validation_dir.mkdir()
    np.savez_compressed(
        validation_dir / "features.npz",
        episode=np.arange(199, dtype=np.int64),
        split=np.array(["validation"] * 199),
    )

    with pytest.raises(ValueError, match="at least 200 episode groups"):
        evaluate(tmp_path / "missing-pilot", validation_dir)
