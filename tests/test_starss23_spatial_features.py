import numpy as np
import pytest

from tiny_omni_decision.starss23_spatial_features import panorama_sector_features


def test_panorama_sectors_follow_starss_front_referenced_azimuth_order():
    # Column values make the back seam and positive-left ordering observable.
    grid = np.zeros((1, 2, 16, 16, 768), dtype=np.float32)
    for column in range(16):
        grid[:, :, :, column, :] = column**2
    pooled = grid.reshape(1, 2 * 256, 768)

    sectors = panorama_sector_features(pooled, frame_count=2)

    assert sectors.shape == (1, 8, 768)
    np.testing.assert_allclose(
        sectors[0, :, 0], [56.5, 30.5, 12.5, 2.5, 112.5, 182.5, 132.5, 90.5]
    )


def test_panorama_sectors_average_time_rows_and_sector_columns():
    grid = np.broadcast_to(
        np.arange(2 * 16 * 16, dtype=np.float32).reshape(1, 2, 16, 16, 1),
        (1, 2, 16, 16, 768),
    ).copy()
    sectors = panorama_sector_features(grid.reshape(1, 2 * 256, 768), frame_count=2)

    assert np.isfinite(sectors).all()
    assert sectors.shape == (1, 8, 768)


@pytest.mark.parametrize(
    ("tokens", "frames"),
    [
        (np.zeros((1, 255, 768), dtype=np.float32), 1),
        (np.zeros((1, 256, 767), dtype=np.float32), 1),
        (np.zeros((1, 512), dtype=np.float32), 2),
        (np.full((1, 256, 768), np.nan, dtype=np.float32), 1),
    ],
)
def test_panorama_bands_reject_malformed_or_nonfinite_tokens(tokens, frames):
        with pytest.raises(ValueError):
            panorama_sector_features(tokens, frame_count=frames)
