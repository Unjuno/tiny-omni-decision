import numpy as np
import pytest

from tiny_omni_decision.video_token_grid import token_grid


def test_grid_preserves_temporal_spatial_order_and_every_value():
    values = np.zeros((1, 2048, 768), dtype=np.float32)
    values[0, 3 * 256 + 4 * 16 + 5, 7] = 123
    grid = token_grid(values)
    assert grid.shape == (8, 16, 16, 768)
    assert grid[3, 4, 5, 7] == 123
    np.testing.assert_array_equal(grid.reshape(1, 2048, 768), values)


def test_grid_rejects_wrong_shape_and_nonfinite():
    with pytest.raises(ValueError):
        token_grid(np.zeros((1, 1024, 768), dtype=np.float32))
    values = np.zeros((1, 2048, 768), dtype=np.float32)
    values[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        token_grid(values)
