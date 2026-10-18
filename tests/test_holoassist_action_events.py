from __future__ import annotations

import hashlib

import pytest

from scripts.build_holoassist_action_events import build_event_rows


def event(event_id: int, label: str, start: float = 1.0, end: float = 2.0) -> dict:
    return {
        "id": event_id,
        "label": "Fine grained action",
        "start": start,
        "end": end,
        "attributes": {"Action Correctness": label},
    }


def fixture(tmp_path):
    files = {}
    selected = {"train": [], "validation": []}
    records = []
    for split, session in (("train", "tr-1"), ("validation", "va-1")):
        path = tmp_path / split / f"{session}.mp4"
        path.parent.mkdir(parents=True)
        path.write_bytes(f"media-{session}".encode())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        files[(split, session)] = {"size_bytes": path.stat().st_size, "sha256": digest}
        selected[split].append({"session_id": session})
        records.append(
            {
                "video_name": session,
                "taskId": f"task-{session}",
                "taskType": "assemble stool",
                "videoMetadata": {"duration": {"raw": "00:00:03.25"}},
                "events": [
                    event(1, "Correct Action", 0.25, 1.25),
                    event(2, "Wrong Action, corrected by instructor verbally", 1.5, 2.0),
                    event(3, "otherwise", 2.0, 2.2),
                    event(4, "Correct Action", 2.5, 2.4),
                ],
            }
        )
    splits = {"train": {"tr-1"}, "validation": {"va-1"}, "test": {"te-1"}}
    return records, splits, selected, files, tmp_path


def test_builder_keeps_modalities_aligned_and_excludes_future_context(tmp_path) -> None:
    records, splits, selected, files, media_root = fixture(tmp_path)

    rows, report = build_event_rows(
        records=records,
        split_ids=splits,
        selected_sessions=selected,
        media_files=files,
        media_root=media_root,
    )

    assert len(rows["train"]) == 2
    assert len(rows["validation"]) == 2
    assert rows["train"][0]["clip_start_seconds"] == 0.0
    assert rows["train"][0]["clip_end_seconds"] == 1.25
    assert rows["train"][0]["future_context_seconds"] == 0.0
    assert rows["train"][0]["target"] == 0
    assert rows["train"][1]["target"] == 1
    assert rows["train"][0]["options"] == rows["train"][1]["options"]
    assert report["validation"]["incorrect_count"] == 1
    assert report["validation"]["excluded_event_counts"] == {
        "otherwise_excluded": 1,
        "reversed_interval_excluded": 1,
    }


def test_builder_rejects_cross_split_session_leakage(tmp_path) -> None:
    records, splits, selected, files, media_root = fixture(tmp_path)
    selected["validation"][0]["session_id"] = "tr-1"
    files[("validation", "tr-1")] = files[("train", "tr-1")]

    with pytest.raises(ValueError, match="escape official validation"):
        build_event_rows(
            records=records,
            split_ids=splits,
            selected_sessions=selected,
            media_files=files,
            media_root=media_root,
        )


def test_builder_rejects_test_annotation_rows(tmp_path) -> None:
    records, splits, selected, files, media_root = fixture(tmp_path)
    records.append({"video_name": "te-1"})

    with pytest.raises(ValueError, match="test annotation"):
        build_event_rows(
            records=records,
            split_ids=splits,
            selected_sessions=selected,
            media_files=files,
            media_root=media_root,
        )


def test_builder_verifies_media_sha_before_emitting_rows(tmp_path) -> None:
    records, splits, selected, files, media_root = fixture(tmp_path)
    files[("train", "tr-1")]["sha256"] = "0" * 64

    with pytest.raises(ValueError, match="SHA-256 changed"):
        build_event_rows(
            records=records,
            split_ids=splits,
            selected_sessions=selected,
            media_files=files,
            media_root=media_root,
        )
