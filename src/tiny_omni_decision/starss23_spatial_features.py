"""Spatially retain STARSS23 azimuth sectors from VideoPrism video tokens."""

from __future__ import annotations

import numpy as np

TOKEN_WIDTH = 768
SPATIAL_GRID = 16
TOKENS_PER_FRAME = SPATIAL_GRID**2
SECTOR_COLUMNS = (
    (7, 8),
    (5, 6),
    (3, 4),
    (1, 2),
    (15, 0),
    (13, 14),
    (11, 12),
    (9, 10),
)


def panorama_sector_features(tokens: np.ndarray, frame_count: int) -> np.ndarray:
    """Pool tokens to eight STARSS azimuth sectors in front-referenced order.

    The pinned VideoPrism v1 base output is `[batch, frame_count*256, 768]`.
    The 16x16 patches are row-major. STARSS23 maps azimuth 0 degrees to the
    panorama center, positive angles to the left, and negative angles to the
    right. Thus front-referenced 45-degree sectors each cover two columns; the
    back sector wraps across the panorama seam and pools columns 15 and 0.
    Each output averages its two columns over all frames and patch rows.
    """
    values = np.asarray(tokens, dtype=np.float32)
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if values.ndim != 3 or values.shape[1:] != (
        frame_count * TOKENS_PER_FRAME,
        TOKEN_WIDTH,
    ):
        raise ValueError("expected finite [batch, frames*256, 768] VideoPrism tokens")
    if not np.isfinite(values).all():
        raise ValueError("VideoPrism tokens must be finite")
    grid = values.reshape(
        values.shape[0], frame_count, SPATIAL_GRID, SPATIAL_GRID, TOKEN_WIDTH
    )
    horizontal_columns = grid.mean(axis=(1, 2))
    sectors = np.stack(
        [horizontal_columns[:, columns, :].mean(axis=1) for columns in SECTOR_COLUMNS],
        axis=1,
    )
    return np.asarray(sectors, dtype=np.float32)
