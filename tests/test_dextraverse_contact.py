from __future__ import annotations

import numpy as np
import pytest

from scripts.pretrained_reuse.dextraverse_contact import (
    LABELS,
    build_contact_windows,
    inner_episode_partitions,
    outer_episode_folds,
)


def _episode(seconds: int = 6) -> tuple[np.ndarray, np.ndarray]:
    timestamps = np.arange(seconds * 20 + 1, dtype=np.float64) / 20
    forces = np.zeros((timestamps.size, 5, 3), dtype=np.float32)
    return timestamps, forces


def _window_at(windows, query_time: float):
    return next(window for window in windows if np.isclose(window.query_time_s, query_time))


def test_first_future_fingertip_crossing_is_labelled_and_inputs_are_causal():
    timestamps, forces = _episode()
    forces[timestamps >= 1.8, 1, 0] = 0.6

    windows = build_contact_windows(timestamps, forces)
    window = _window_at(windows, 1.5)

    assert LABELS[window.target] == "index"
    assert max(window.sensor_indices) == window.query_index
    assert max(window.video_indices) < window.query_index
    assert timestamps[max(window.video_indices)] <= window.query_time_s - 0.25
    assert min(window.video_indices) >= 0


def test_two_fingertips_crossing_within_one_sample_are_simultaneous():
    timestamps, forces = _episode()
    forces[timestamps >= 1.8, 0, 0] = 0.6
    forces[timestamps >= 1.85, 4, 0] = 0.7

    window = _window_at(build_contact_windows(timestamps, forces), 1.5)

    assert LABELS[window.target] == "simultaneous"


def test_contact_already_present_at_query_is_not_a_new_event():
    timestamps, forces = _episode()
    forces[timestamps < 1.5, 2, 0] = 0.6
    forces[timestamps >= 1.5, 2, 0] = 0.8

    window = _window_at(build_contact_windows(timestamps, forces), 1.5)

    assert LABELS[window.target] == "no_new_contact_within_1s"


def test_no_future_crossing_is_no_new_contact():
    timestamps, forces = _episode()

    window = _window_at(build_contact_windows(timestamps, forces), 1.5)

    assert LABELS[window.target] == "no_new_contact_within_1s"


def test_query_must_match_tactile_grid_within_tolerance():
    timestamps, forces = _episode()
    timestamps[30:] += 0.04

    windows = build_contact_windows(timestamps, forces)

    assert all(
        min(window.query_time_s % 0.5, 0.5 - window.query_time_s % 0.5) <= 0.025
        for window in windows
    )


def test_gap_in_input_or_future_target_window_rejects_query():
    timestamps, forces = _episode()
    timestamps[37:] += 0.25

    windows = build_contact_windows(timestamps, forces)

    assert all(not (1.5 <= window.query_time_s <= 2.0) for window in windows)


def test_missing_force_rejects_affected_window():
    timestamps, forces = _episode()
    forces[30, 0, 0] = np.nan

    windows = build_contact_windows(timestamps, forces)

    assert all(window.query_time_s != 1.5 for window in windows)


def test_missing_video_frames_reject_affected_window():
    timestamps, forces = _episode()
    video_valid = np.ones(timestamps.size, dtype=bool)
    video_valid[(timestamps >= 0.3) & (timestamps <= 0.45)] = False

    windows = build_contact_windows(timestamps, forces, video_valid=video_valid)

    assert all(window.query_time_s != 1.5 for window in windows)


def test_invalid_shapes_and_nonmonotonic_time_fail_closed():
    timestamps, forces = _episode()

    with pytest.raises(ValueError, match="shape"):
        build_contact_windows(timestamps, forces[:, :4])
    timestamps[10] = timestamps[9]
    with pytest.raises(ValueError, match="strictly increasing"):
        build_contact_windows(timestamps, forces)


def test_outer_episode_folds_are_deterministic_and_balanced():
    episode_ids = list(range(258))

    first = outer_episode_folds(episode_ids)
    second = outer_episode_folds(episode_ids)

    assert first == second
    fold_sizes = [sum(fold == value for value in first.values()) for fold in range(5)]
    assert max(fold_sizes) - min(fold_sizes) <= 1
    assert set(first) == set(episode_ids)


def test_nested_episode_partitions_are_disjoint_and_complete():
    episode_ids = list(range(40))

    partitions = inner_episode_partitions(episode_ids, outer_fold=2)

    fit, calibration, selection = map(
        set, (partitions.fit, partitions.calibration, partitions.selection)
    )
    assert not fit & calibration
    assert not fit & selection
    assert not calibration & selection
    assert fit | calibration | selection == set(episode_ids)
    assert len(calibration) == 4
    assert len(selection) == 4
