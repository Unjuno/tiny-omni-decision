"""Foreground-area statistics of frozen feature cells; no labels or models."""

import numpy as np


def mask_feature_statistics(masks, feature_grids):
    """Keep each mask; treat each feature cell as constant over its pixel area.

    Foreground pixel counts weight cells, not bounding boxes or seed points.
    Concatenate grids in supplied order. Float64 two-pass population variance
    avoids second-moment cancellation; saved mean/std remain float32. Empty
    masks keep zero descriptors and explicit counts instead of being dropped.
    """
    if (not isinstance(masks, np.ndarray) or masks.ndim != 3 or min(masks.shape) < 1
            or masks.dtype not in (np.dtype('uint8'), np.dtype('bool'))
            or np.any(masks > 1) or not isinstance(feature_grids, (list, tuple))
            or not feature_grids):
        raise ValueError('nonempty binary mask inventory and ordered feature grids required')
    q, h, w = masks.shape
    for grid in feature_grids:
        if (not isinstance(grid, np.ndarray) or grid.ndim != 3 or min(grid.shape) < 1
                or grid.dtype != np.float32 or not np.isfinite(grid).all()
                or h % grid.shape[0] or w % grid.shape[1]):
            raise ValueError('finite FP32 feature cells must partition the mask raster exactly')
    counts = masks.sum((1, 2), dtype=np.int64)
    means, stds = [], []
    for grid in feature_grids:
        gh, gw, c = grid.shape
        weights = masks.reshape(q, gh, h // gh, gw, w // gw).sum((2, 4), dtype=np.float64)
        weights = weights.reshape(q, gh * gw)
        values = grid.reshape(gh * gw, c).astype(np.float64)
        mean = np.zeros((q, c), np.float64)
        std = np.zeros((q, c), np.float64)
        for i in np.flatnonzero(counts):
            mean[i] = (weights[i, :, None] * values).sum(0) / counts[i]
            centered = values - mean[i]
            variance = (weights[i, :, None] * (centered * centered)).sum(0) / counts[i]
            std[i] = np.sqrt(variance)
        means.append(mean.astype(np.float32))
        stds.append(std.astype(np.float32))
    return np.concatenate(means, 1), np.concatenate(stds, 1), counts
