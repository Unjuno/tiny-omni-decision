import numpy as np
import pytest

from scripts.compare_dinov2_spatial_readouts import paired_image_bootstrap


def test_paired_image_bootstrap_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="equal"):
        paired_image_bootstrap(
            {"accuracy": np.zeros((3, 4))},
            {"accuracy": np.zeros((2, 4))},
            repeats=100,
        )


def test_paired_image_bootstrap_reports_zero_delta_for_equal_predictions():
    metrics = {
        "accuracy": np.asarray([[1, 0, 1, 1], [0, 1, 0, 1], [1, 1, 0, 0]], dtype=float),
        "nll": np.asarray(
            [[0.2, 0.4, 0.3, 0.2], [0.6, 0.5, 0.1, 0.4], [0.3, 0.1, 0.5, 0.1]],
            dtype=float,
        ),
    }
    result = paired_image_bootstrap(metrics, metrics, repeats=100, seed=17)
    for metric in metrics:
        for task in ("macro", "color", "texture", "count", "shape"):
            assert result[metric][task]["point_delta"] == 0
            assert result[metric][task]["ci95"] == [0, 0]


def test_paired_image_bootstrap_requires_enough_replicates():
    values = {"accuracy": np.ones((3, 4))}
    with pytest.raises(ValueError, match="bootstrap input"):
        paired_image_bootstrap(values, values, repeats=99)
