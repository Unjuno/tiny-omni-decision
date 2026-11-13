from __future__ import annotations

import pytest

from scripts.filter_cater_temporal_manifest_by_media import filter_by_media_frames


def _row(split: str, video: str, start: int, end: int) -> dict:
    return {
        "id": f"{video}-{start}-{end}",
        "group_id": video,
        "video_id": video,
        "split": split,
        "question": f"interval {start}-{end}",
        "frame_interval": [start, end],
        "options": [{"id": "a"}, {"id": "b"}],
        "num_options": 2,
        "target_option_id": "a",
    }


def test_media_filter_keeps_observable_windows_and_preserves_groups() -> None:
    train = [
        _row("train", "full", 0, 75),
        _row("train", "full", 225, 300),
        _row("train", "short", 0, 75),
        _row("train", "short", 75, 150),
    ]
    validation = [_row("validation", "valid", 150, 225)]
    media = [
        {"split": "train", "video_id": "full", "stream_frames": 301},
        {"split": "train", "video_id": "short", "stream_frames": 100, "decoded_frames": 100},
        {"split": "validation", "video_id": "valid", "stream_frames": 301},
    ]

    filtered_train, filtered_validation, summary = filter_by_media_frames(train, validation, media)

    assert [row["id"] for row in filtered_train] == [
        "full-0-75",
        "full-225-300",
        "short-0-75",
    ]
    assert [row["id"] for row in filtered_validation] == ["valid-150-225"]
    assert summary["removed_rows_by_split"] == {"train": 1}
    assert summary["train_validation_video_overlap"] == 0
    assert summary["option_count_unchanged"] is True


def test_media_filter_fails_closed_when_a_video_is_missing_frame_audit() -> None:
    row = _row("train", "missing", 0, 75)

    with pytest.raises(ValueError, match="missing frame-count record"):
        filter_by_media_frames([row], [_row("validation", "valid", 0, 75)], [])


def test_media_filter_rejects_duplicate_frame_records() -> None:
    train = [_row("train", "duplicate", 0, 75)]
    frame_record = {"split": "train", "video_id": "duplicate", "stream_frames": 301}

    with pytest.raises(ValueError, match="duplicate media frame-count record"):
        filter_by_media_frames(train, [], [frame_record, frame_record])
