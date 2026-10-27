import numpy as np
import pytest

pytest.importorskip("torch")

from scripts.fit_perception_temporal_match_readouts import categorical_metrics as binary_metrics
from scripts.fit_perception_temporal_match_sequence_readouts import (
    categorical_metrics as summed_metrics,
)


def test_binary_and_categorical_brier_have_fixed_factor_two():
    scores = np.array([[0.0, 0.0], [1.0, -1.0], [-0.7, 0.2]])
    targets = np.array([0, 1, 1])
    binary = binary_metrics(scores, targets)["brier"]
    summed = summed_metrics(scores, targets)["brier"]
    assert binary * 2 == pytest.approx(summed, abs=1e-15)
    assert binary_metrics(scores[:1], targets[:1])["brier"] == pytest.approx(0.25)
    assert summed_metrics(scores[:1], targets[:1])["brier"] == pytest.approx(0.5)
