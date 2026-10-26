"""Deterministic fixed-duration audio windows for event-conditioned features."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def split_audio_interval(
    start_us: int, end_us: int, *, window_us: int = 2_000_000
) -> list[dict[str, int]]:
    """Partition an event interval into nonoverlapping windows with centered zero padding.

    Every source microsecond belongs to exactly one returned segment. Segments
    shorter than `window_us` are centered in the model window using zero padding;
    no samples outside the annotated event interval are introduced.
    """
    if type(start_us) is not int or type(end_us) is not int or start_us < 0 or end_us <= start_us:
        raise ValueError("audio interval must be a positive half-open integer range")
    if type(window_us) is not int or window_us <= 0:
        raise ValueError("window size must be a positive integer")

    duration = end_us - start_us
    if duration <= window_us:
        missing = window_us - duration
        pad_left = missing // 2
        return [
            {
                "source_start_us": start_us,
                "source_end_us": end_us,
                "pad_left_us": pad_left,
                "pad_right_us": missing - pad_left,
            }
        ]

    windows = []
    cursor = start_us
    while cursor < end_us:
        next_cursor = min(cursor + window_us, end_us)
        missing = window_us - (next_cursor - cursor)
        pad_left = missing // 2
        windows.append(
            {
                "source_start_us": cursor,
                "source_end_us": next_cursor,
                "pad_left_us": pad_left,
                "pad_right_us": missing - pad_left,
            }
        )
        cursor = next_cursor
    return windows


def materialize_audio_window(
    waveform: Sequence[float] | np.ndarray,
    window: Mapping[str, int],
    *,
    sample_rate_hz: int,
    window_us: int = 2_000_000,
) -> np.ndarray:
    """Crop a resampled mono waveform and zero-pad it to exactly `window_us`."""
    if type(sample_rate_hz) is not int or sample_rate_hz <= 0:
        raise ValueError("sample rate must be a positive integer")
    if type(window_us) is not int or window_us <= 0:
        raise ValueError("window size must be a positive integer")
    try:
        start = window["source_start_us"]
        end = window["source_end_us"]
        pad_left = window["pad_left_us"]
        pad_right = window["pad_right_us"]
    except (KeyError, TypeError) as error:
        raise ValueError("audio window metadata is incomplete") from error
    if (
        any(type(value) is not int for value in (start, end, pad_left, pad_right))
        or start < 0
        or end <= start
        or pad_left < 0
        or pad_right < 0
        or end - start + pad_left + pad_right != window_us
    ):
        raise ValueError("audio window metadata must cover one fixed positive window")

    values = np.asarray(waveform)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.number):
        raise ValueError("waveform must be a numeric mono vector")
    if not np.isfinite(values).all():
        raise ValueError("waveform must contain only finite samples")
    first = round(start * sample_rate_hz / 1_000_000)
    last = round(end * sample_rate_hz / 1_000_000)
    if first < 0 or last > len(values) or last <= first:
        raise ValueError("audio source interval is outside waveform")

    target_samples = round(window_us * sample_rate_hz / 1_000_000)
    left_samples = round(pad_left * sample_rate_hz / 1_000_000)
    source_samples = values[first:last]
    right_samples = target_samples - left_samples - len(source_samples)
    if right_samples < 0:
        raise ValueError("rounded audio samples exceed fixed model window")
    result = np.pad(source_samples, (left_samples, right_samples), mode="constant")
    if len(result) != target_samples:
        raise AssertionError("materialized audio window has unexpected sample count")
    return result
