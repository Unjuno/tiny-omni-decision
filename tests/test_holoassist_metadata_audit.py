from __future__ import annotations

import pytest

from scripts.audit_holoassist_metadata import audit_metadata, duration_seconds


def record(session_id: str, *, start: float = 1.0, end: float = 2.0) -> dict:
    return {
        "video_name": session_id,
        "taskId": f"task-{session_id}",
        "taskType": "example-task",
        "videoMetadata": {"duration": {"raw": "00:00:03.25", "seconds": 3}},
        "events": [
            {
                "label": "Fine grained action",
                "start": start,
                "end": end,
                "attributes": {"Action Correctness": "Correct Action"},
            }
        ],
    }


def splits() -> dict[str, set[str]]:
    return {"train": {"train-1"}, "validation": {"val-1"}, "test": {"test-1"}}


def test_audit_keeps_test_split_metadata_only_and_hashes_disjoint_ids() -> None:
    result = audit_metadata([record("train-1"), record("val-1")], splits())

    assert result["split_overlap_counts"] == {
        "train_x_validation": 0,
        "train_x_test": 0,
        "validation_x_test": 0,
    }
    assert result["splits"]["train"]["annotation_coverage_count"] == 1
    assert result["splits"]["validation"]["annotation_coverage_count"] == 1
    assert result["splits"]["test"]["annotation_coverage_count"] == 0
    assert "fine_grained_action_count" not in result["splits"]["test"]


def test_audit_rejects_cross_split_session_overlap() -> None:
    overlapping = {**splits(), "validation": {"train-1"}}

    with pytest.raises(ValueError, match="overlap"):
        audit_metadata([record("train-1")], overlapping)


def test_audit_rejects_missing_train_or_validation_annotation() -> None:
    with pytest.raises(ValueError, match="lack trainval annotations"):
        audit_metadata([record("train-1")], splits())


def test_audit_counts_reversed_action_interval_without_clamping_it() -> None:
    result = audit_metadata(
        [record("train-1", start=2.1, end=2.0), record("val-1")], splits()
    )

    assert result["splits"]["train"]["reversed_fine_action_intervals"] == 1
    assert result["splits"]["train"]["out_of_range_fine_action_intervals"] == 0


def test_duration_parser_uses_fractional_raw_duration() -> None:
    assert duration_seconds("00:01:02.345") == pytest.approx(62.345)
    assert duration_seconds("not-a-time") is None
