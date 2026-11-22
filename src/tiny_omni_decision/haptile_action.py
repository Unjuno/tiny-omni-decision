"""Causal coarse-action labels for the pinned HapTile pilot."""

from __future__ import annotations

import math
from collections.abc import Sequence

ACTION_LABELS = ("-x", "+x", "-y", "+y", "-z", "+z", "hold")
ACTION_OPTION_TEXTS = (
    "move the end effector in the negative X direction",
    "move the end effector in the positive X direction",
    "move the end effector in the negative Y direction",
    "move the end effector in the positive Y direction",
    "move the end effector in the negative Z direction",
    "move the end effector in the positive Z direction",
    "hold the end effector still",
)
ACTION_QUESTION = (
    "Given the instruction and the past one second of robot observations, "
    "which direction will the end effector move during the next 250 to 500 milliseconds?"
)
DEFAULT_MOTION_DEADBAND_M = 0.005


def directional_action_label(
    current_xyz: Sequence[float],
    future_xyz: Sequence[float],
    *,
    deadband_m: float = DEFAULT_MOTION_DEADBAND_M,
) -> str:
    """Choose the dominant Cartesian displacement direction or ``hold``.

    The label is computed from future-minus-current end-effector position. The
    caller is responsible for keeping all observations used as input at or
    before the query cutoff; future positions are target-only.
    """
    if len(current_xyz) != 3 or len(future_xyz) != 3:
        raise ValueError("current and future positions must each have three axes")
    if not math.isfinite(deadband_m) or deadband_m <= 0:
        raise ValueError("deadband_m must be finite and positive")
    current = tuple(float(value) for value in current_xyz)
    future = tuple(float(value) for value in future_xyz)
    if not all(math.isfinite(value) for value in (*current, *future)):
        raise ValueError("positions must contain only finite values")
    delta = tuple(end - start for start, end in zip(current, future, strict=True))
    if math.sqrt(sum(value * value for value in delta)) < deadband_m:
        return "hold"
    axis = max(range(3), key=lambda index: abs(delta[index]))
    return ("+" if delta[axis] > 0 else "-") + "xyz"[axis]
