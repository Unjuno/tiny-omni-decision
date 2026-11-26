"""Deterministic media transforms for the DCASE 2025 stereo SELD task."""

from __future__ import annotations

import math

import numpy as np


def uniform_clip_frame_indices(
    *,
    start_frame: int,
    fps: float,
    clip_frames: int = 50,
    output_frames: int = 8,
    event_frame: int | None = None,
) -> tuple[int, ...]:
    """Return source indices for the full clip or a fixed 0.7s event-centered view."""
    if start_frame < 0 or clip_frames < 1 or output_frames < 1:
        raise ValueError("frame offsets and counts must be nonnegative and positive")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("video frame rate must be finite and positive")
    start_index = int(np.round(start_frame * 0.1 * fps))
    clip_length = int(np.round(clip_frames * 0.1 * fps))
    if clip_length < 1 or output_frames > clip_length:
        raise ValueError("requested frame count does not fit in the DCASE clip")
    if event_frame is None:
        first, last = 0, clip_length - 1
    else:
        if not 0 <= event_frame < clip_frames:
            raise ValueError("event frame is outside the DCASE clip")
        window_length = min(clip_length, max(output_frames, int(np.round(0.7 * fps))))
        center = int(np.round(event_frame * 0.1 * fps))
        first = min(max(center - window_length // 2, 0), clip_length - window_length)
        last = first + window_length - 1
    offsets = np.linspace(first, last, output_frames).round().astype(int)
    return tuple(int(start_index + offset) for offset in offsets)


def stereo_from_foa(
    foa_clip: np.ndarray, *, view_degrees: int, clip_limit: float = 0.9999
) -> np.ndarray:
    """Apply the official FOA-to-stereo orientation transform to [samples,4] audio."""
    values = np.asarray(foa_clip, dtype=np.float32)
    if (
        values.ndim != 2
        or values.shape[1] != 4
        or values.shape[0] < 1
        or not np.isfinite(values).all()
    ):
        raise ValueError("FOA audio must be a finite [samples,4] array")
    if isinstance(view_degrees, bool) or not isinstance(view_degrees, int):
        raise ValueError("view angle must be an integer number of degrees")
    if not 0 <= view_degrees < 360 or not 0 < clip_limit <= 1:
        raise ValueError("view angle or clipping limit is outside the supported range")

    angle = np.deg2rad(view_degrees)
    w, y, _z, x = values.T
    rotated_y = np.sin(angle) * x + np.cos(angle) * y
    stereo = np.stack((w + rotated_y, w - rotated_y), axis=1).astype(np.float32)
    if float(np.abs(stereo).max()) > clip_limit:
        raise ValueError("official DCASE generator would skip this clipped stereo sample")
    return stereo
