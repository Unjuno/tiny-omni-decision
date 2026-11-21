"""Pure transformations shared by the DextraVerse frozen-feature worker."""

from __future__ import annotations

import struct
from typing import BinaryIO

import numpy as np
from numpy.typing import NDArray

VIDEO_BATCH_SHAPE = (8, 240, 320, 3)
VIDEO_FEATURE_SHAPE = (8, 768)


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = stream.read(size - len(chunks))
        if not chunk:
            raise EOFError(f"video feature pipe closed after {len(chunks)}/{size} bytes")
        chunks.extend(chunk)
    return bytes(chunks)


def write_video_batch(stream: BinaryIO, frames: NDArray[np.uint8]) -> int:
    """Send a little-endian count and raw RGB batch to the pinned encoder worker."""
    values = np.asarray(frames)
    if (values.dtype != np.uint8 or values.ndim != 5 or values.shape[0] < 1
            or tuple(values.shape[1:]) != VIDEO_BATCH_SHAPE):
        raise ValueError("encoder input must be uint8 [batch, 8, 240, 320, 3]")
    stream.write(struct.pack("<I", values.shape[0]))
    stream.write(np.ascontiguousarray(values).tobytes(order="C"))
    stream.flush()
    return values.shape[0]


def read_video_batch(stream: BinaryIO) -> NDArray[np.uint8] | None:
    """Read one encoder request; a zero count is the graceful-stop marker."""
    header = stream.read(4)
    if len(header) != 4:
        raise EOFError("encoder input closed before a complete batch header")
    count = struct.unpack("<I", header)[0]
    if count == 0:
        return None
    size = count * int(np.prod(VIDEO_BATCH_SHAPE))
    return np.frombuffer(_read_exact(stream, size), dtype=np.uint8).reshape(
        count, *VIDEO_BATCH_SHAPE
    )


def write_video_features(stream: BinaryIO, features: NDArray[np.floating]) -> None:
    """Write one `[batch, 8, 768]` float32 response to the parent process."""
    values = np.asarray(features, dtype=np.float32)
    if (values.ndim != 3 or values.shape[0] < 1
            or tuple(values.shape[1:]) != VIDEO_FEATURE_SHAPE
            or not np.isfinite(values).all()):
        raise ValueError("encoder response must be finite [batch, 8, 768] float32")
    stream.write(np.ascontiguousarray(values).tobytes(order="C"))
    stream.flush()


def read_video_features(stream: BinaryIO, count: int) -> NDArray[np.float32]:
    """Read the fixed-size response for one previously submitted batch."""
    if count < 1:
        raise ValueError("response count must be positive")
    size = count * int(np.prod(VIDEO_FEATURE_SHAPE)) * np.dtype(np.float32).itemsize
    result = np.frombuffer(_read_exact(stream, size), dtype=np.float32).reshape(
        count, *VIDEO_FEATURE_SHAPE
    )
    if not np.isfinite(result).all():
        raise ValueError("encoder returned non-finite temporal features")
    return result


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
