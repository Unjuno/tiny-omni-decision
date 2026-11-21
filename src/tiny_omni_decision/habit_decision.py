"""Map HABIT frame labels to unique text options for Decision examples."""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


def normalize_habit_task_text(value: str) -> str:
    """Normalize formatting for exact candidate matching, without paraphrasing."""
    if not isinstance(value, str):
        raise TypeError("HABIT task text must be a string")
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


@dataclass(frozen=True)
class HabitChoiceTarget:
    """A unique candidate list and target index derived from a global task ID."""

    options: tuple[str, ...]
    target_index: int
    collapsed_duplicate_count: int


@dataclass(frozen=True)
class HabitTransitionPoint:
    """A target subtask onset and the latest admissible pre-onset input time."""

    target_subtask_id: int
    onset_ms: float
    context_end_ms: float


def habit_transition_points(
    timestamps_ms: Sequence[float],
    subtask_ids: Sequence[int],
    *,
    pre_transition_guard_ms: float = 100.0,
) -> tuple[HabitTransitionPoint, ...]:
    """Return subtask onsets after the initial segment with a causal cutoff.

    Each label is treated as the task active at its timestamp. The first run
    has no prior task boundary and is omitted. Inputs must end at least the
    guard interval before each target onset; future timestamps are never read
    or returned as input times.
    """
    if len(timestamps_ms) != len(subtask_ids) or not timestamps_ms:
        raise ValueError("timestamps and subtask IDs must be nonempty and aligned")
    if (not math.isfinite(pre_transition_guard_ms) or pre_transition_guard_ms <= 0
            or any(not math.isfinite(time) for time in timestamps_ms)
            or any(later <= earlier for earlier, later in zip(
                timestamps_ms, timestamps_ms[1:], strict=False
            ))):
        raise ValueError(
            "timestamps must be finite and strictly increasing; guard must be positive"
        )
    if any(isinstance(label, bool) or not isinstance(label, int) or label < 0
           for label in subtask_ids):
        raise ValueError("subtask IDs must be nonnegative integers")

    return tuple(
        HabitTransitionPoint(
            target_subtask_id=label,
            onset_ms=float(timestamps_ms[index]),
            context_end_ms=float(timestamps_ms[index]) - pre_transition_guard_ms,
        )
        for index, label in enumerate(subtask_ids)
        if index > 0 and label != subtask_ids[index - 1]
    )


def map_habit_choice_target(
    episode_options: Sequence[str],
    target_subtask_id: int,
    global_subtasks: Mapping[int, str],
) -> HabitChoiceTarget:
    """Resolve a global Parquet label against an episode's candidate texts.

    Repeated action text within a task plan denotes the same semantic action
    for this Decision target, so retain its first occurrence and collapse later
    duplicates. The returned target index is over this unique, stable-order list.
    """
    if isinstance(target_subtask_id, bool) or not isinstance(target_subtask_id, int):
        raise TypeError("target_subtask_id must be an integer")
    if target_subtask_id not in global_subtasks:
        raise ValueError("target_subtask_id is outside the global subtask catalog")
    if not episode_options:
        raise ValueError("episode options must not be empty")

    unique_options: list[str] = []
    normalized_to_index: dict[str, int] = {}
    for option in episode_options:
        normalized = normalize_habit_task_text(option)
        if not normalized:
            raise ValueError("episode options must contain nonempty task text")
        if normalized not in normalized_to_index:
            normalized_to_index[normalized] = len(unique_options)
            unique_options.append(option.strip())

    if len(unique_options) < 2:
        raise ValueError("at least two unique semantic options are required")

    target_text = global_subtasks[target_subtask_id]
    normalized_target = normalize_habit_task_text(target_text)
    if not normalized_target or normalized_target not in normalized_to_index:
        raise ValueError("global target task text is absent from episode options")

    return HabitChoiceTarget(
        options=tuple(unique_options),
        target_index=normalized_to_index[normalized_target],
        collapsed_duplicate_count=len(episode_options) - len(unique_options),
    )
