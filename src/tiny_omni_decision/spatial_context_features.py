"""Whole-grid temporal appearance; no labels, regions or semantic selection."""

import numpy as np


def pooled_spatial_context(lowres, hires, *, grid_size=16):
    """Mean every spatial cell into ordered grid tokens, retaining each time.

    Pooling can dilute a small object's features; it does not establish semantic
    identification. FP64 accumulation prevents intermediate FP32 sum overflow.
    Stored features remain FP32, matching the frozen encoder's precision.
    """
    if type(grid_size) is not int or grid_size < 1:
        raise ValueError("positive integer grid size required")
    arrays = [np.asarray(lowres), np.asarray(hires)]
    for a in arrays:
        if (
            a.ndim != 4
            or min(a.shape) == 0
            or a.dtype.kind != "f"
            or not np.isfinite(a).all()
            or a.shape[1] % grid_size
            or a.shape[2] % grid_size
        ):
            raise ValueError(
                "nonempty finite floating time/spatial grids divisible by grid required"
            )
    if arrays[0].shape[0] != arrays[1].shape[0]:
        raise ValueError("aligned time grids required")
    pooled = []
    for a in arrays:
        t, h, w, c = a.shape
        blocks = a.reshape(t, grid_size, h // grid_size, grid_size, w // grid_size, c)
        with np.errstate(over="ignore", invalid="ignore"):
            value = blocks.mean(axis=(2, 4), dtype=np.float64).astype(np.float32)
        if not np.isfinite(value).all():
            raise ValueError("pooled feature exceeds finite FP32 range")
        pooled.append(value)
    result = np.concatenate(pooled, axis=-1)
    return result.reshape(result.shape[0], grid_size * grid_size, result.shape[-1])


def spatial_context_positions(frame_indices, *, source_frames, grid_size=16):
    """Normalized source x/y/time for every retained appearance token."""
    frames = np.asarray(frame_indices)
    if (
        type(source_frames) is not int
        or source_frames < 2
        or type(grid_size) is not int
        or grid_size < 1
        or frames.ndim != 1
        or len(frames) == 0
        or frames.dtype.kind not in "iu"
        or (frames < 0).any()
        or (frames >= source_frames).any()
        or not (frames[1:] > frames[:-1]).all()
    ):
        raise ValueError("strictly ordered integer source frame IDs and valid raster required")
    xy = np.array(
        [
            [(x + 0.5) / grid_size, (y + 0.5) / grid_size]
            for y in range(grid_size)
            for x in range(grid_size)
        ],
        dtype=np.float32,
    )
    result = np.empty((len(frames), len(xy), 3), dtype=np.float32)
    result[..., :2] = xy[None]
    result[..., 2] = (frames.astype(np.float64) / (source_frames - 1))[:, None]
    return result
