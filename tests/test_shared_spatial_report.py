import math

import pytest

from scripts.report_shared_spatial_images import metrics


def test_saved_probability_metrics():
    rows = [dict(options=["a", "b"], target="a", prediction="a", probabilities=[0.75, 0.25]),
            dict(options=["a", "b"], target="b", prediction="a", probabilities=[0.75, 0.25])]
    result = metrics(rows)
    assert result["accuracy"] == 0.5
    assert result["nll"] == pytest.approx((-math.log(0.75)-math.log(0.25))/2)
    assert result["brier"] == 0.625
    assert result["ece_15_bins"] == 0.25


def test_inconsistent_saved_predictions_rejected():
    with pytest.raises(ValueError, match="prediction differs"):
        metrics([dict(options=["a", "b"], target="a", prediction="b", probabilities=[0.8, 0.2])])
