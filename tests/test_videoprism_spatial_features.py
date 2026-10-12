import numpy as np
import pytest

from tiny_omni_decision.videoprism_spatial_features import readout_features, spatial_cells


def test_grid_pooling_and_global_control():
    x = np.arange(256, dtype=np.float32)[None, :, None] * np.ones((1, 1, 768), np.float32)
    cells = spatial_cells(x)
    assert cells.shape == (1, 17, 768)
    np.testing.assert_allclose(cells[:, 0], x.mean(1))
    assert cells[0, 1, 0] == 25.5 and cells[0, -1, 0] == 229.5
    assert readout_features(cells, "spatial").shape == (1, 13056)
    global_features = readout_features(cells, "global")
    assert np.unique(global_features[0, 768:]).size == 1


def test_invalid_features_rejected():
    with pytest.raises(ValueError):
        spatial_cells(np.zeros((1, 255, 768)))
    with pytest.raises(ValueError):
        spatial_cells(np.full((1, 256, 768), np.nan))
