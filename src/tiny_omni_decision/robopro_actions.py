"""Small, auditable utilities for RoboPro demonstrator-action probes."""

from __future__ import annotations

import math
from typing import Any


def uniform_indices(frame_count: int, requested: int) -> list[int]:
    if frame_count < 1 or requested < 1 or requested > frame_count:
        raise ValueError("requested frame count must fit within the source clip")
    if requested == 1:
        return [0]
    step = (frame_count - 1) / (requested - 1)
    return [round(index * step) for index in range(requested)]


def causal_window_indices(
    frame_count: int,
    query_index: int,
    *,
    history_seconds: float = 2.0,
    fps: int = 50,
    requested: int = 8,
) -> list[int]:
    """Sample an endpoint-inclusive window ending at the observation time."""
    if (
        frame_count < 1
        or not 0 <= query_index < frame_count
        or not math.isfinite(history_seconds)
        or history_seconds <= 0
        or fps < 1
        or requested < 2
    ):
        raise ValueError("invalid causal window dimensions or time parameters")
    start_index = max(0, query_index - round(history_seconds * fps))
    available = query_index - start_index + 1
    if available < requested:
        raise ValueError("query has too little past context for distinct video frames")
    return [index + start_index for index in uniform_indices(available, requested)]


def numeric_option_separation(
    options: list[list[float]],
    *,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Summarize exact/near duplicate numeric options without judging validity."""
    if len(options) < 2 or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("at least two options and a finite non-negative tolerance are required")
    dimension = len(options[0])
    if dimension < 1 or any(len(option) != dimension for option in options):
        raise ValueError("numeric options must have one shared non-empty dimension")
    if any(not math.isfinite(value) for option in options for value in option):
        raise ValueError("numeric options must contain only finite values")

    distances: list[float] = []
    exact_duplicate_pairs: list[list[int]] = []
    near_duplicate_pairs: list[list[int]] = []
    for left in range(len(options)):
        for right in range(left + 1, len(options)):
            distance = max(abs(a - b) for a, b in zip(options[left], options[right], strict=True))
            distances.append(distance)
            if distance == 0:
                exact_duplicate_pairs.append([left, right])
            if distance <= tolerance:
                near_duplicate_pairs.append([left, right])
    ordered = sorted(distances)
    middle = len(ordered) // 2
    median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2
    return {
        "dimension": dimension,
        "pair_count": len(distances),
        "exact_duplicate_pairs": exact_duplicate_pairs,
        "within_tolerance_pairs": near_duplicate_pairs,
        "min_linf_distance": min(distances),
        "median_linf_distance": median,
        "max_linf_distance": max(distances),
        "comparison_tolerance": tolerance,
    }


def validate_probe_feature_shapes(
    temporal_shape: tuple[int, ...],
    state_shape: tuple[int, ...],
    target_shape: tuple[int, ...],
    options_shape: tuple[int, ...],
    *,
    episode_count: int,
    prompt_group_count: int,
    query_count: int = 3,
    frame_count: int = 8,
    feature_width: int = 768,
    state_width: int = 14,
    action_width: int = 14,
) -> None:
    """Validate feature cache dimensions against the selected sample size."""
    expected_temporal = (episode_count, query_count, frame_count, feature_width)
    expected_state = (episode_count, query_count, frame_count, state_width)
    expected_target = (episode_count, query_count, action_width)
    expected_options = (prompt_group_count, query_count, 4, action_width)
    if temporal_shape != expected_temporal:
        raise ValueError(f"unexpected temporal feature shape: {temporal_shape}")
    if state_shape != expected_state or target_shape != expected_target:
        raise ValueError("sensor/action feature cache has an unexpected shape")
    if options_shape != expected_options:
        raise ValueError(f"unexpected candidate action shape: {options_shape}")
