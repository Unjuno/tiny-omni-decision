"""Pure transformations shared by the DextraVerse frozen-feature worker."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def pool_videoprism_tokens(
    tokens: NDArray[np.floating], frame_count: int = 8
) -> NDArray[np.float32]:
    """Reduce spatial tokens while retaining temporal order as ``[T, 768]``."""
    values = np.asarray(tokens, dtype=np.float32)
    if (values.ndim != 3 or values.shape[0] != 1 or values.shape[2] != 768
            or frame_count < 1 or values.shape[1] % frame_count):
        raise ValueError("expected VideoPrism [1, T*spatial, 768] tokens")
    pooled = values.reshape(1, frame_count, -1, 768).mean(axis=2)[0]
    if pooled.shape != (frame_count, 768) or not np.isfinite(pooled).all():
        raise ValueError("VideoPrism temporal features must be finite [T, 768]")
    return pooled


def assemble_sensor_window(
    left_force: NDArray[np.floating],
    right_force: NDArray[np.floating],
    timestamps_s: NDArray[np.floating],
    sensor_indices: list[int] | tuple[int, ...],
    query_index: int,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Return left-then-right 30-D force and query-relative milliseconds."""
    left = np.asarray(left_force)
    right = np.asarray(right_force)
    times = np.asarray(timestamps_s, dtype=np.float64)
    indices = np.asarray(sensor_indices, dtype=np.int64)
    if (left.ndim != 3 or right.shape != left.shape or left.shape[1:] != (5, 3)
            or times.ndim != 1 or left.shape[0] != times.size
            or not 0 <= query_index < times.size or indices.ndim != 1 or indices.size < 2
            or np.any(indices < 0) or np.any(indices >= times.size)
            or np.any(np.diff(indices) <= 0) or query_index not in indices):
        raise ValueError("invalid force arrays, timestamps, or sensor indices")
    values = np.concatenate((left[indices], right[indices]), axis=1).reshape(-1, 30)
    relative_ms = ((times[indices] - times[query_index]) * 1000).astype(np.float32)
    if not np.isfinite(values).all() or not np.isfinite(relative_ms).all():
        raise ValueError("sensor features and timestamps must be finite")
    if np.any(np.diff(relative_ms) <= 0) or relative_ms[-1] > 1e-4:
        raise ValueError("sensor history must be strictly ordered and causal")
    return values.astype(np.float32, copy=False), relative_ms
