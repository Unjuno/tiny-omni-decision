"""Causal video-window selection for the HapTile Decision pilot."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def select_causal_frame_indices(
    timestamps_s: Sequence[float],
    query_time_s: float,
    *,
    context_seconds: float = 1.0,
    frame_count: int = 8,
    max_time_error_seconds: float = 0.1,
) -> tuple[int, ...]:
    """Select unique frames nearest to evenly spaced past-only target times."""
    if (
        not math.isfinite(query_time_s)
        or not math.isfinite(context_seconds)
        or context_seconds <= 0
        or frame_count < 2
        or not math.isfinite(max_time_error_seconds)
        or max_time_error_seconds < 0
    ):
        raise ValueError("invalid query, context, frame count, or timing tolerance")
    times = tuple(float(value) for value in timestamps_s)
    if (
        not times
        or any(not math.isfinite(value) for value in times)
        or any(later <= earlier for earlier, later in zip(times, times[1:], strict=False))
    ):
        raise ValueError("video timestamps must be finite and strictly increasing")

    targets = np.linspace(query_time_s - context_seconds, query_time_s, frame_count)
    # Assign distinct frames jointly. Independent nearest-neighbor selection can
    # map adjacent targets to the same final timestamp when a clip ends just
    # before the query cutoff, despite having enough unique frames overall.
    # Dynamic programming minimizes total squared timing error while preserving
    # chronological order and the fixed per-frame tolerance.
    states: dict[int, tuple[float, tuple[int, ...]]] = {}
    for target_index, target in enumerate(targets):
        current: dict[int, tuple[float, tuple[int, ...]]] = {}
        for index, timestamp in enumerate(times):
            error = abs(timestamp - float(target))
            if timestamp > query_time_s or error > max_time_error_seconds:
                continue
            if target_index == 0:
                current[index] = (error * error, (index,))
                continue
            previous = [
                value for previous_index, value in states.items()
                if previous_index < index
            ]
            if previous:
                previous_cost, previous_path = min(previous, key=lambda value: value)
                current[index] = (
                    previous_cost + error * error,
                    (*previous_path, index),
                )
        if not current:
            raise ValueError(
                "no distinct causal frames satisfy the fixed temporal tolerance"
            )
        states = current
    return min(states.values(), key=lambda value: value)[1]
