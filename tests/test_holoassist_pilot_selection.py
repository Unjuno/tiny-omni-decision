from __future__ import annotations

import pytest

from scripts.select_holoassist_pilot_sessions import build_selection


def row(session_id: str, task_type: str, correctness: str = "Correct Action") -> dict:
    return {
        "video_name": session_id,
        "taskId": f"task-{session_id}",
        "taskType": task_type,
        "events": [
            {
                "label": "Fine grained action",
                "attributes": {"Action Correctness": correctness},
            }
        ],
    }


def member(session_id: str, index: int) -> dict:
    return {
        "name": f"{session_id}/Export_py/Video_compress.mp4",
        "session_id": session_id,
        "size": index + 1,
        "offset_data": index * 100,
    }


def fixture():
    records = [
        row("tr-a1", "task-a", "Correct Action"),
        row("tr-a2", "task-a", "Wrong Action, corrected by performer"),
        row("tr-b1", "task-b"),
        row("tr-b2", "task-b"),
        row("va-a1", "task-a"),
        row("va-a2", "task-a"),
        row("va-b1", "task-b"),
        row("te-1", "task-b"),
    ]
    splits = {
        "train": {"tr-a1", "tr-a2", "tr-b1", "tr-b2"},
        "validation": {"va-a1", "va-a2", "va-b1"},
        "test": {"te-1"},
    }
    selected_ids = splits["train"] | splits["validation"]
    members = [member(sid, index) for index, sid in enumerate(selected_ids)]
    return records[:-1], splits, members


def test_selection_is_deterministic_stratified_and_excludes_test() -> None:
    records, splits, members = fixture()

    first = build_selection(records, splits, members, train_count=2, validation_count=2, seed=17)
    second = build_selection(records, splits, members, train_count=2, validation_count=2, seed=17)

    assert first == second
    assert set(first["train"]["task_type_counts"]) == {"task-a", "task-b"}
    assert set(first["validation"]["task_type_counts"]) == {"task-a", "task-b"}
    selected = {
        video["session_id"]
        for split in ("train", "validation")
        for video in first[split]["videos"]
    }
    assert not selected & splits["test"]
    assert first["all_selected_video_bytes"] > 0


def test_selection_requires_a_video_for_every_train_and_validation_session() -> None:
    records, splits, members = fixture()

    with pytest.raises(ValueError, match="differ"):
        build_selection(records, splits, members[:-1], train_count=2, validation_count=2)


def test_selection_rejects_any_test_media_member() -> None:
    records, splits, members = fixture()
    members.append(member("te-1", 99))

    with pytest.raises(ValueError, match="test media"):
        build_selection(records, splits, members, train_count=2, validation_count=2)


def test_selection_never_accepts_test_annotation_rows() -> None:
    records, splits, members = fixture()
    records.append(row("te-1", "task-b"))

    with pytest.raises(ValueError, match="test annotation"):
        build_selection(records, splits, members, train_count=2, validation_count=2)


def test_selection_rejects_duplicate_video_entries() -> None:
    records, splits, members = fixture()
    members.append(members[0])

    with pytest.raises(ValueError, match="duplicate compressed video"):
        build_selection(records, splits, members, train_count=2, validation_count=2)
