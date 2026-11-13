"""Letterbox geometry and token-grid pooling for proposal-conditioned video features."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def map_bbox_to_token_grid(
    bbox_xyxy_normalized: Sequence[float],
    source_width: int,
    source_height: int,
    *,
    frame_size: int = 288,
    grid_size: int = 16,
) -> tuple[float, float, float, float]:
    """Map a normalized source-frame box through square letterbox to token xyxy."""
    if len(bbox_xyxy_normalized) != 4:
        raise ValueError("bbox must be a finite normalized xyxy quadruple")
    x1, y1, x2, y2 = (float(value) for value in bbox_xyxy_normalized)
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)) or not (
        0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1
    ):
        raise ValueError("bbox must be a finite normalized xyxy quadruple")
    if any(
        type(value) is not int or value <= 0
        for value in (source_width, source_height, frame_size, grid_size)
    ):
        raise ValueError("frame and token dimensions must be positive integers")

    scale = min(frame_size / source_width, frame_size / source_height)
    pad_x = (frame_size - source_width * scale) / 2
    pad_y = (frame_size - source_height * scale) / 2
    factor = grid_size / frame_size
    return (
        (x1 * source_width * scale + pad_x) * factor,
        (y1 * source_height * scale + pad_y) * factor,
        (x2 * source_width * scale + pad_x) * factor,
        (y2 * source_height * scale + pad_y) * factor,
    )


def roi_average(tokens: np.ndarray, bbox_xyxy_grid: Sequence[float]) -> np.ndarray:
    """Area-weight token vectors intersecting an xyxy ROI in grid coordinates."""
    values = np.asarray(tokens)
    if values.ndim != 3 or not np.issubdtype(values.dtype, np.number):
        raise ValueError("tokens must be a numeric [height, width, channels] array")
    if len(bbox_xyxy_grid) != 4:
        raise ValueError("ROI must have positive finite xyxy coordinates")
    x1, y1, x2, y2 = (float(value) for value in bbox_xyxy_grid)
    height, width = values.shape[:2]
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)) or not (
        0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height
    ):
        raise ValueError("ROI must have positive finite xyxy coordinates inside the token grid")

    x_start, x_stop = math.floor(x1), math.ceil(x2)
    y_start, y_stop = math.floor(y1), math.ceil(y2)
    x_cells = np.arange(x_start, x_stop)
    y_cells = np.arange(y_start, y_stop)
    x_overlap = np.maximum(0, np.minimum(x_cells + 1, x2) - np.maximum(x_cells, x1))
    y_overlap = np.maximum(0, np.minimum(y_cells + 1, y2) - np.maximum(y_cells, y1))
    weights = y_overlap[:, None] * x_overlap[None, :]
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("ROI must overlap at least one spatial token")
    selected = values[y_start:y_stop, x_start:x_stop, :]
    if not np.isfinite(selected).all():
        raise ValueError("ROI tokens must be finite")
    pooled = np.einsum("hw,hwd->d", weights, selected, optimize=True) / total
    return pooled.astype(values.dtype, copy=False)
