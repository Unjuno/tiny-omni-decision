"""Read fine-grid appearance along every model-produced track without labels."""

import numpy as np


def track_appearance_features(lowres, hires, normalized_xy):
    """Border bilinear sampling using TAPIR's half-cell convention.

    Return [points,time,channels] FP32 and a closed-raster support flag. Original
    coordinates are never changed. Border features outside the frame are retained
    but do not imply valid visible correspondence. No descriptor renormalization.
    """
    grids = [np.asarray(lowres), np.asarray(hires)]
    xy = np.asarray(normalized_xy)
    if (
        xy.ndim != 3
        or min(xy.shape) == 0
        or xy.shape[-1] != 2
        or xy.dtype.kind != "f"
        or not np.isfinite(xy).all()
    ):
        raise ValueError("nonempty finite floating [points,time,xy] coordinates required")
    for grid in grids:
        if (
            grid.ndim != 4
            or min(grid.shape) == 0
            or grid.shape[0] != xy.shape[1]
            or grid.dtype.kind != "f"
            or not np.isfinite(grid).all()
        ):
            raise ValueError("aligned nonempty finite floating time/spatial grids required")
    supported = ((xy >= 0) & (xy <= 1)).all(-1)
    # Bound only lookup indices for source-compatible border padding, not inputs.
    lookup = np.clip(xy.astype(np.float64), 0, 1)
    time = np.arange(xy.shape[1])[None]
    output = []
    for grid in grids:
        _, h, w, _ = grid.shape
        x = np.clip(lookup[..., 0] * w - 0.5, 0, w - 1)
        y = np.clip(lookup[..., 1] * h - 0.5, 0, h - 1)
        x0, y0 = np.floor(x).astype(np.intp), np.floor(y).astype(np.intp)
        x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
        fx, fy = (x - x0)[..., None], (y - y0)[..., None]
        with np.errstate(over="ignore", invalid="ignore"):
            value = (
                grid[time, y0, x0] * ((1 - fx) * (1 - fy))
                + grid[time, y0, x1] * (fx * (1 - fy))
                + grid[time, y1, x0] * ((1 - fx) * fy)
                + grid[time, y1, x1] * (fx * fy)
            ).astype(np.float32)
        if not np.isfinite(value).all():
            raise ValueError("sampled features exceed finite FP32 range")
        output.append(value)
    return np.concatenate(output, axis=-1), supported
