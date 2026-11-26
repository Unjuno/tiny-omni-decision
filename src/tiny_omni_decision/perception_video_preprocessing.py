"""Box-preserving RGB preprocessing for frozen VideoPrism token grids."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


def letterbox_videoprism_frames(
    frames: Sequence[np.ndarray],
    resize_bilinear: Callable[[np.ndarray, tuple[int, int]], np.ndarray],
    *,
    frame_size: int = 288,
    pad_value: float = 0.0,
) -> np.ndarray:
    """Resize RGB frames without cropping and return normalized square inputs.

    `resize_bilinear` is supplied by the pinned TensorFlow preprocessing runtime.
    The geometry uses the same centered, shorter-side-preserving letterbox rule
    as `map_bbox_to_token_grid`.
    """
    values = np.asarray(frames)
    if (
        values.ndim != 4
        or values.shape[0] < 1
        or values.shape[-1] != 3
        or values.shape[1] < 1
        or values.shape[2] < 1
        or not np.issubdtype(values.dtype, np.number)
        or not np.isfinite(values).all()
    ):
        raise ValueError("frames must be a finite nonempty [T,H,W,3] RGB batch")
    if type(frame_size) is not int or frame_size < 1:
        raise ValueError("frame_size must be a positive integer")
    if not np.isfinite(pad_value) or not 0 <= pad_value <= 1:
        raise ValueError("pad_value must be in [0,1]")

    result = np.full(
        (values.shape[0], frame_size, frame_size, 3),
        fill_value=pad_value,
        dtype=np.float32,
    )
    for index, frame in enumerate(values):
        height, width = frame.shape[:2]
        scale = min(frame_size / height, frame_size / width)
        resized_height = max(1, round(height * scale))
        resized_width = max(1, round(width * scale))
        resized = np.asarray(resize_bilinear(frame, (resized_height, resized_width)))
        if resized.shape != (resized_height, resized_width, 3):
            raise ValueError("bilinear resize returned an unexpected frame shape")
        if not np.isfinite(resized).all():
            raise ValueError("bilinear resize returned nonfinite pixels")
        top = (frame_size - resized_height) // 2
        left = (frame_size - resized_width) // 2
        result[index, top : top + resized_height, left : left + resized_width] = (
            resized.astype(np.float32, copy=False) / 255.0
        )
    return result
