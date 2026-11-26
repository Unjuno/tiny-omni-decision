"""Utilities for preserving VideoPrism's time-major token sequence."""

from __future__ import annotations

import numpy as np


def reshape_videoprism_temporal_tokens(
    tokens: np.ndarray, frame_count: int
) -> np.ndarray:
    """Reshape `[batch, frames * spatial_tokens, width]` into `[B,T,N,D]`."""
    values = np.asarray(tokens)
    if values.ndim != 3:
        raise ValueError("VideoPrism tokens must have shape [batch, sequence, width]")
    if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count < 1:
        raise ValueError("frame_count must be a positive integer")
    batch_size, sequence_length, width = values.shape
    if min(batch_size, sequence_length, width) < 1:
        raise ValueError("VideoPrism token dimensions must be nonempty")
    if sequence_length % frame_count:
        raise ValueError("VideoPrism token sequence length must be divisible by frame_count")
    if width < 1 or not np.isfinite(values).all():
        raise ValueError("VideoPrism tokens must be finite and have a nonempty width")
    return values.reshape(batch_size, frame_count, sequence_length // frame_count, width)


def pool_spatial_tokens_per_frame(tokens: np.ndarray, frame_count: int) -> np.ndarray:
    """Mean-pool spatial tokens while retaining the ordered frame dimension."""
    return reshape_videoprism_temporal_tokens(tokens, frame_count).mean(axis=2)


def videoprism_features_by_layout(
    tokens: np.ndarray, frame_count: int, layout: str
) -> np.ndarray:
    """Convert VideoPrism output tokens to a pooled or time-preserving layout."""
    shaped = reshape_videoprism_temporal_tokens(tokens, frame_count)
    if layout == "pooled":
        batch_size, frames, spatial_tokens, width = shaped.shape
        return shaped.reshape(batch_size, frames * spatial_tokens, width).mean(axis=1)
    if layout == "per_frame_mean":
        return shaped.mean(axis=2)
    if layout == "spatiotemporal_tokens":
        return shaped
    raise ValueError(f"unsupported VideoPrism feature layout: {layout}")
