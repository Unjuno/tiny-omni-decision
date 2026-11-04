import numpy as np
import pytest

from tiny_omni_decision.spatial_context_features import (
    pooled_spatial_context,
    spatial_context_positions,
)


def test_every_cell_contributes_in_row_major_order_and_time_is_retained():
    low = np.arange(2 * 4 * 4 * 2, dtype=np.float32).reshape(2, 4, 4, 2)
    high = np.arange(2 * 8 * 8 * 3, dtype=np.float32).reshape(2, 8, 8, 3)
    expected = []
    for t in range(2):
        cells = []
        for y in range(2):
            for x in range(2):
                cells.append(
                    np.concatenate(
                        [
                            low[t, y * 2 : (y + 1) * 2, x * 2 : (x + 1) * 2].mean((0, 1)),
                            high[t, y * 4 : (y + 1) * 4, x * 4 : (x + 1) * 4].mean((0, 1)),
                        ]
                    )
                )
        expected.append(cells)
    actual = pooled_spatial_context(low, high, grid_size=2)
    np.testing.assert_array_equal(actual, np.array(expected))
    np.testing.assert_array_equal(
        pooled_spatial_context(low[::-1], high[::-1], grid_size=2), actual[::-1]
    )
    assert actual.dtype == np.float32
    changed = high.copy()
    changed[1, -1, -1, -1] += 16
    delta = pooled_spatial_context(low, changed, grid_size=2) - actual
    assert np.count_nonzero(delta) == 1 and delta[1, -1, -1] == 1


def test_pooling_does_not_overflow_large_finite_features_or_mutate_inputs():
    low = np.full((1, 4, 4, 1), 1e38, dtype=np.float32)
    high = low.copy()
    result = pooled_spatial_context(low, high, grid_size=2)
    assert np.isfinite(result).all()
    assert (result == low[0, 0, 0, 0]).all()
    assert (low == high).all()


@pytest.mark.parametrize("kind", ["empty", "time", "grid", "nan", "complex", "integer"])
def test_invalid_spatial_grids_fail_closed(kind):
    low = np.zeros((2, 4, 4, 1), dtype=np.float32)
    high = np.zeros((2, 8, 8, 1), dtype=np.float32)
    grid = 2
    if kind == "empty":
        low = low[:0]
    elif kind == "time":
        high = high[:1]
    elif kind == "grid":
        grid = 3
    elif kind == "nan":
        low[0, 0, 0, 0] = np.nan
    elif kind == "complex":
        high = high.astype(np.complex64)
    else:
        low = low.astype(np.int64)
    with pytest.raises(ValueError):
        pooled_spatial_context(low, high, grid_size=grid)


def test_actual_frame_positions_are_aligned_to_cells_and_keep_original_time():
    positions = spatial_context_positions(np.array([0, 2, 4]), source_frames=5, grid_size=2)
    assert positions.shape == (3, 4, 3) and positions.dtype == np.float32
    np.testing.assert_array_equal(
        positions[0, :, :2], [[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]]
    )
    np.testing.assert_array_equal(positions[:, :, 2], [[0.0] * 4, [0.5] * 4, [1.0] * 4])


@pytest.mark.parametrize(
    "frames,source", [([], 5), ([0, 0], 5), ([1, 0], 5), ([0, 5], 5), ([0.0, 2.0], 5), ([0, 2], 1)]
)
def test_invalid_time_coordinates_fail_closed(frames, source):
    with pytest.raises(ValueError):
        spatial_context_positions(np.array(frames), source_frames=source, grid_size=2)
