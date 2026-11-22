"""Causal task and robot-state features for the HapTile pilot."""

from __future__ import annotations

import bisect
import math
from collections.abc import Mapping, Sequence

import numpy as np

STATE_CHANNEL_ORDER = (
    "pose_x", "pose_y", "pose_z",
    "velocity_vx", "velocity_vy", "velocity_vz",
    "velocity_wx", "velocity_wy", "velocity_wz",
    "joint_base", "joint_shoulder", "joint_elbow", "joint_wrist_1",
    "joint_wrist_2", "joint_wrist_3",
    "joint_velocity_base", "joint_velocity_shoulder", "joint_velocity_elbow",
    "joint_velocity_wrist_1", "joint_velocity_wrist_2", "joint_velocity_wrist_3",
    "gripper_position", "tactile_left", "tactile_right", "tactile_normalized",
)

STREAM_WIDTHS = {
    "pose": 3,
    "end_effector_velocity": 6,
    "joints": 6,
    "joint_velocities": 6,
    "gripper": 1,
    "tactile": 3,
}


def causal_state_sequence(
    streams: Mapping[str, tuple[Sequence[float], Sequence[Sequence[float]]]],
    query_time_s: float,
    *,
    context_seconds: float = 1.0,
    steps: int = 8,
    max_staleness_seconds: float = 0.2,
) -> np.ndarray:
    """Align ordered sensor streams as-of each fixed past-only query time."""
    if (not math.isfinite(query_time_s) or not math.isfinite(context_seconds)
            or context_seconds <= 0 or steps < 2
            or not math.isfinite(max_staleness_seconds)
            or max_staleness_seconds < 0):
        raise ValueError("invalid causal sequence timing configuration")
    if set(streams) != set(STREAM_WIDTHS):
        raise ValueError("all declared HapTile state streams are required")

    targets = np.linspace(query_time_s - context_seconds, query_time_s, steps)
    aligned: list[list[float]] = [[] for _ in range(steps)]
    for name, width in STREAM_WIDTHS.items():
        raw_times, raw_values = streams[name]
        times = tuple(float(value) for value in raw_times)
        values = tuple(tuple(float(value) for value in row) for row in raw_values)
        if (not times or len(times) != len(values)
                or any(not math.isfinite(value) for value in times)
                or any(later <= earlier for earlier, later in zip(
                    times, times[1:], strict=False
                ))
                or any(len(row) != width or not all(map(math.isfinite, row))
                       for row in values)):
            raise ValueError(f"{name} stream is empty, malformed or non-monotonic")
        for index, target in enumerate(targets):
            observation_index = bisect.bisect_right(times, float(target)) - 1
            if observation_index < 0:
                raise ValueError(f"{name} has no observation at or before context start")
            age = float(target) - times[observation_index]
            if age > max_staleness_seconds:
                raise ValueError(f"{name} observation exceeds fixed staleness limit")
            aligned[index].extend(values[observation_index])

    result = np.asarray(aligned, dtype=np.float32)
    if result.shape != (steps, len(STATE_CHANNEL_ORDER)) or not np.isfinite(result).all():
        raise ValueError("aligned HapTile state sequence has invalid shape or values")
    return result
