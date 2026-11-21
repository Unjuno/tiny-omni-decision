"""Causal episode-window construction for the pinned DextraVerse contact task."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FINGER_NAMES = ("thumb", "index", "middle", "ring", "little")
LABELS = (
    *(f"left_{name}" for name in FINGER_NAMES),
    *(f"right_{name}" for name in FINGER_NAMES),
    "simultaneous",
    "no_new_contact_within_1s",
)
FINGER_COUNT = 10
FORCE_AXES = 3
SPLIT_PREFIX = "dextraverse-contact-decision-v1|17"


@dataclass(frozen=True)
class ContactWindow:
    """One valid query and its strictly causal inputs."""

    query_index: int
    query_time_s: float
    video_indices: tuple[int, ...]
    sensor_indices: tuple[int, ...]
    target: int


@dataclass(frozen=True)
class EpisodePartitions:
    """Disjoint group roles inside one outer training fold."""

    fit: tuple[int, ...]
    calibration: tuple[int, ...]
    selection: tuple[int, ...]


def outer_episode_folds(episode_indices: list[int] | tuple[int, ...],
                        fold_count: int = 5) -> dict[int, int]:
    """Assign complete episode IDs to deterministic SHA-256 OOF folds."""
    ids = tuple(episode_indices)
    if (fold_count < 2 or len(ids) < fold_count or len(set(ids)) != len(ids)
            or any(type(index) is not int or index < 0 for index in ids)):
        raise ValueError("unique nonnegative episode IDs and at least two folds are required")
    ordered = sorted(
        ids,
        key=lambda index: (
            hashlib.sha256(f"{SPLIT_PREFIX}|outer|{index}".encode()).hexdigest(),
            index,
        ),
    )
    return {episode_id: position % fold_count for position, episode_id in enumerate(ordered)}


def inner_episode_partitions(episode_indices: list[int] | tuple[int, ...],
                             outer_fold: int) -> EpisodePartitions:
    """Reserve disjoint whole episodes for calibration and checkpoint selection."""
    ids = tuple(episode_indices)
    if (len(set(ids)) != len(ids) or len(ids) < 10
            or any(type(index) is not int or index < 0 for index in ids)
            or type(outer_fold) is not int or outer_fold < 0):
        raise ValueError("at least ten unique nonnegative episode IDs are required")
    ordered = sorted(
        ids,
        key=lambda index: (
            hashlib.sha256(
                f"{SPLIT_PREFIX}|inner|{outer_fold}|{index}".encode()
            ).hexdigest(),
            index,
        ),
    )
    role_size = math.ceil(0.10 * len(ordered))
    calibration = tuple(ordered[:role_size])
    selection = tuple(ordered[role_size:2 * role_size])
    fit = tuple(ordered[2 * role_size:])
    if not calibration or not selection or not fit:
        raise ValueError(
            "nested episode groups cannot form nonempty fit/calibration/selection sets"
        )
    return EpisodePartitions(fit=fit, calibration=calibration, selection=selection)


def _window_has_gap(timestamps_s: NDArray[np.float64], start_s: float, end_s: float,
                    max_gap_s: float) -> bool:
    starts_before_end = timestamps_s[:-1] < end_s
    ends_after_start = timestamps_s[1:] > start_s
    return bool(np.any(
        (np.diff(timestamps_s) > max_gap_s) & starts_before_end & ends_after_start
    ))


def _nearest_index(timestamps_s: NDArray[np.float64], target_s: float,
                   valid: NDArray[np.bool_]) -> int | None:
    candidates = np.flatnonzero(valid)
    if candidates.size == 0:
        return None
    offset = int(np.argmin(np.abs(timestamps_s[candidates] - target_s)))
    return int(candidates[offset])


def build_contact_windows(
    timestamps_s: NDArray[np.float64],
    fingertip_force_n: NDArray[np.float32] | NDArray[np.float64],
    *,
    video_valid: NDArray[np.bool_] | None = None,
    threshold_n: float = 0.5,
    query_step_s: float = 0.5,
    history_s: float = 1.0,
    horizon_s: float = 1.0,
    video_lead_s: float = 0.25,
    video_duration_s: float = 1.0,
    video_frame_count: int = 8,
    query_tolerance_s: float = 0.025,
    video_tolerance_s: float = 0.050,
    max_gap_s: float = 0.200,
) -> list[ContactWindow]:
    """Create fixed-grid next-contact examples from one episode.

    Force has shape ``[time, 10 left-then-right fingertips, 3 axes]`` in
    calibrated newtons.
    Invalid windows are skipped. Contact labels are first rising crossings of
    the force-vector magnitude; no label or input uses data after the query
    time. The returned sensor indices lie in ``[query-history, query]`` and
    video indices end no later than ``query-video_lead``.
    """
    timestamps = np.asarray(timestamps_s, dtype=np.float64)
    force = np.asarray(fingertip_force_n, dtype=np.float64)
    if timestamps.ndim != 1 or timestamps.size < 2:
        raise ValueError("timestamps_s must contain at least two episode samples")
    if force.shape != (timestamps.size, FINGER_COUNT, FORCE_AXES):
        raise ValueError("force must have shape [time, 10 left-then-right fingertips, 3 axes]")
    if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
        raise ValueError("episode timestamps must be finite and strictly increasing")
    numeric_settings = (
        threshold_n, query_step_s, history_s, horizon_s, video_lead_s,
        video_duration_s, query_tolerance_s, video_tolerance_s, max_gap_s,
    )
    if (not np.isfinite(numeric_settings).all() or threshold_n <= 0 or query_step_s <= 0
            or history_s <= 0 or horizon_s <= 0 or video_lead_s < 0
            or video_duration_s <= 0 or video_frame_count < 1
            or query_tolerance_s < 0 or video_tolerance_s < 0 or max_gap_s <= 0):
        raise ValueError("window timings and threshold must be finite and in range")
    if video_valid is None:
        frame_valid = np.ones(timestamps.size, dtype=bool)
    else:
        frame_valid = np.asarray(video_valid, dtype=bool)
        if frame_valid.shape != timestamps.shape:
            raise ValueError("video_valid must match timestamps_s")

    # A crossing is defined by the two sampled states. The vector norm is
    # rotation-invariant despite each fingertip sensor having its own frame.
    force_valid = np.isfinite(force).all(axis=(1, 2))
    magnitudes = np.linalg.norm(force, axis=2)
    in_contact = magnitudes >= threshold_n
    rising = (~in_contact[:-1]) & in_contact[1:]
    crossing_times = timestamps[1:]
    sample_interval = float(np.median(np.diff(timestamps)))
    if not np.isfinite(sample_interval) or sample_interval <= 0:
        raise ValueError("episode has no valid tactile sample interval")

    windows: list[ContactWindow] = []
    first_query = timestamps[0] + history_s
    last_query = timestamps[-1] - horizon_s
    grid_count = int(np.floor((last_query - first_query) / query_step_s)) + 1
    for grid_index in range(max(0, grid_count)):
        intended_query = first_query + grid_index * query_step_s
        query_index = _nearest_index(
            timestamps, intended_query, np.ones(timestamps.size, dtype=bool)
        )
        if query_index is None:
            continue
        query_time = float(timestamps[query_index])
        if abs(query_time - intended_query) > query_tolerance_s:
            continue

        sensor_start = query_time - history_s
        video_end = query_time - video_lead_s
        video_start = video_end - video_duration_s
        target_end = query_time + horizon_s
        if any(_window_has_gap(timestamps, start, end, max_gap_s) for start, end in (
            (sensor_start, query_time),
            (video_start, video_end),
            (query_time, target_end),
        )):
            continue

        sensor_indices = np.flatnonzero(
            (timestamps >= sensor_start) & (timestamps <= query_time)
        )
        if sensor_indices.size < 2 or not force_valid[sensor_indices].all():
            continue
        target_indices = np.flatnonzero(
            (timestamps >= query_time) & (timestamps <= target_end)
        )
        if target_indices.size < 2 or not force_valid[target_indices].all():
            continue
        if not force_valid[query_index]:
            continue

        target_times = np.linspace(
            video_start, video_end, num=video_frame_count, dtype=np.float64
        )
        selected_frames: list[int] = []
        for target_time in target_times:
            # Never use a frame later than the declared 250-ms guard, even if
            # it is the nearest frame to the endpoint target.
            causal_valid = frame_valid & (timestamps <= video_end)
            frame_index = _nearest_index(timestamps, float(target_time), causal_valid)
            if (frame_index is None
                    or abs(timestamps[frame_index] - target_time) > video_tolerance_s):
                break
            selected_frames.append(frame_index)
        if (len(selected_frames) != video_frame_count
                or len(set(selected_frames)) != video_frame_count):
            continue

        event_mask = rising & (crossing_times[:, None] > query_time)
        event_mask &= crossing_times[:, None] <= target_end
        event_rows, event_fingers = np.nonzero(event_mask)
        if event_rows.size == 0:
            target = LABELS.index("no_new_contact_within_1s")
        else:
            event_times = crossing_times[event_rows]
            earliest = float(event_times.min())
            simultaneous_fingers = np.unique(
                event_fingers[event_times <= earliest + sample_interval + 1e-12]
            )
            if simultaneous_fingers.size > 1:
                target = LABELS.index("simultaneous")
            else:
                target = int(simultaneous_fingers[0])

        windows.append(ContactWindow(
            query_index=int(query_index),
            query_time_s=query_time,
            video_indices=tuple(selected_frames),
            sensor_indices=tuple(int(index) for index in sensor_indices),
            target=target,
        ))
    return windows
