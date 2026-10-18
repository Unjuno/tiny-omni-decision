from __future__ import annotations

import pytest

from scripts.build_holoassist_temporal_windows import build_temporal_windows


def row(
    split: str,
    session: str,
    event_id: str,
    start: float,
    end: float,
    target: int,
    duration: float = 65.0,
) -> dict:
    return {
        "id": event_id,
        "split": split,
        "session_id": session,
        "task_type": "assemble stool",
        "media_path": f"{split}/{session}.mp4",
        "media_sha256": f"hash-{session}",
        "action_start_seconds": start,
        "action_end_seconds": end,
        "video_duration_seconds": duration,
        "target": target,
        "future_context_seconds": 0.0,
    }


def test_window_builder_assigns_each_action_once_and_masks_empty_context() -> None:
    rows = {
        "train": [
            row("train", "tr-1", "e1", 1.0, 10.0, 0),
            row("train", "tr-1", "e2", 12.0, 20.0, 1),
            row("train", "tr-1", "e3", 31.0, 35.0, 0),
        ],
        "validation": [row("validation", "va-1", "e4", 40.0, 60.0, 0)],
    }

    windows, report = build_temporal_windows(rows, window_seconds=30.0)

    assert len(windows["train"]) == 3
    assert [window["event_ids"] for window in windows["train"]] == [["e1", "e2"], ["e3"], []]
    assert [window["target"] for window in windows["train"]] == [1, 0, None]
    assert [window["loss_mask"] for window in windows["train"]] == [True, True, False]
    assert report["train"]["source_action_event_count"] == 3
    assert report["train"]["positive_window_count"] == 1
    assert report["validation"]["window_count"] == 3


def test_window_builder_rejects_future_context_and_cross_split_sessions() -> None:
    rows = {
        "train": [row("train", "same", "e1", 1.0, 10.0, 0)],
        "validation": [row("validation", "same", "e2", 1.0, 10.0, 1)],
    }
    with pytest.raises(ValueError, match="sessions overlap"):
        build_temporal_windows(rows)

    rows["validation"][0]["session_id"] = "other"
    rows["validation"][0]["future_context_seconds"] = 0.1
    with pytest.raises(ValueError, match="post-action context"):
        build_temporal_windows(rows)


def test_window_builder_rejects_actions_outside_video_duration() -> None:
    rows = {
        "train": [row("train", "tr-1", "e1", 60.0, 70.0, 0)],
        "validation": [row("validation", "va-1", "e2", 1.0, 2.0, 0)],
    }
    with pytest.raises(ValueError, match="fell outside time windows"):
        build_temporal_windows(rows)
