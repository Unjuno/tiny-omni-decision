"""Exact native visual-bank intervention; independent of Torch/model loading."""

import numpy as np


def visual_condition(frames, pooled, condition):
    if (
        frames.ndim != 3
        or frames.shape[1:] != (16, 768)
        or pooled.shape != (len(frames), 768)
        or not len(frames)
        or frames.dtype != np.float32
        or pooled.dtype != np.float32
        or not np.isfinite(frames).all()
        or not np.isfinite(pooled).all()
        or not np.array_equal(frames.mean(1), pooled)
        or condition not in {"ordered", "broadcast"}
    ):
        raise ValueError("complete finite aligned native16-frame FP32 bank required")
    if condition == "ordered":
        return frames
    return np.repeat(pooled[:, None, :], 16, axis=1)
