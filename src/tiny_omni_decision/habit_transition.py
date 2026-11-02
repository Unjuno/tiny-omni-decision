"""Leakage-safe transition windows and option mapping for HABIT probes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class TransitionWindow:
    boundary: int
    history_indices: tuple[int, ...]
    target_index: int
    target_option_id: int


def resolve_episode_options(
    episode_low_level_tasks: Sequence[str], global_subtasks: Mapping[int, str]
) -> tuple[tuple[int, str], ...]:
    """Resolve per-episode option text through the pinned global subtask table.

    Exact text matching is intentional: fuzzy or positional matching can silently
    associate a target label with the wrong action description.
    """
    if isinstance(episode_low_level_tasks, (str, bytes)) or not episode_low_level_tasks:
        raise ValueError("episode must provide a non-empty low-level option list")

    reverse: dict[str, int] = {}
    for raw_id, raw_text in global_subtasks.items():
        if isinstance(raw_id, bool) or not isinstance(raw_id, int) or raw_id < 0:
            raise ValueError("global subtask IDs must be non-negative integers")
        if not isinstance(raw_text, str) or not raw_text.strip():
            raise ValueError("global subtask text must be non-empty")
        text = raw_text.strip()
        if text in reverse:
            raise ValueError(f"duplicate global subtask text is ambiguous: {text!r}")
        reverse[text] = raw_id

    options: list[tuple[int, str]] = []
    seen_texts: set[str] = set()
    for raw_text in episode_low_level_tasks:
        if not isinstance(raw_text, str) or not raw_text.strip():
            raise ValueError("episode option text must be a non-empty string")
        text = raw_text.strip()
        if text in seen_texts:
            raise ValueError(f"duplicate episode option text: {text!r}")
        seen_texts.add(text)
        if text not in reverse:
            raise ValueError(f"episode option text is not present in global subtasks: {text!r}")
        options.append((reverse[text], text))

    if len(options) < 2:
        raise ValueError("decision task must provide at least two candidate options")
    return tuple(sorted(options))


def transition_window_for_boundary(
    labels: Sequence[int],
    *,
    boundary: int,
    option_ids: Sequence[int],
    frame_count: int = 8,
    observation_gap: int = 5,
    target_horizon: int = 5,
) -> TransitionWindow | None:
    """Build an 8-frame history ending before a label change and future target.

    With defaults, inputs are rows ``boundary-12..boundary-5`` and the target is
    the low-level task label at ``boundary+5`` (one second later at 10 Hz).
    ``None`` means the episode edge prevents the fixed context/horizon.
    """
    if isinstance(boundary, bool) or not isinstance(boundary, int):
        raise ValueError("boundary must be an integer")
    if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count < 1:
        raise ValueError("frame_count must be a positive integer")
    if (
        isinstance(observation_gap, bool)
        or not isinstance(observation_gap, int)
        or observation_gap < 1
    ):
        raise ValueError("observation_gap must be a positive integer")
    if (
        isinstance(target_horizon, bool)
        or not isinstance(target_horizon, int)
        or target_horizon < 1
    ):
        raise ValueError("target_horizon must be a positive integer")
    if len(set(option_ids)) != len(option_ids) or len(option_ids) < 2:
        raise ValueError("candidate option IDs must be unique and contain at least two options")
    if any(
        isinstance(option_id, bool) or not isinstance(option_id, int) or option_id < 0
        for option_id in option_ids
    ):
        raise ValueError("candidate option IDs must be non-negative integers")
    if boundary <= 0 or boundary >= len(labels):
        return None
    if labels[boundary] == labels[boundary - 1]:
        raise ValueError("boundary does not identify a low-level task transition")

    history_end = boundary - observation_gap
    history_start = history_end - frame_count + 1
    target_index = boundary + target_horizon
    if history_start < 0 or target_index >= len(labels):
        return None
    target_option_id = labels[target_index]
    if target_option_id not in option_ids:
        raise ValueError("future target is not among the supplied candidate options")
    return TransitionWindow(
        boundary=boundary,
        history_indices=tuple(range(history_start, history_end + 1)),
        target_index=target_index,
        target_option_id=target_option_id,
    )
