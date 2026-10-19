from __future__ import annotations

import hashlib
import zipfile

import pytest

from scripts.build_holoassist_action_events import (
    build_event_rows,
    load_official_train_only,
    validate_av_validation_report,
)


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


def test_internal_validation_split_reader_does_not_read_other_split_payloads(tmp_path):
    path = tmp_path / "splits.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("train-v1_2.txt", "train-a\ntrain-b\n")
        archive.writestr("val-v1_2.txt", b"\xff\xfe")
        archive.writestr("test-v1_2.txt", b"\xff\xfe")

    assert load_official_train_only(path) == {"train-a", "train-b"}


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


def test_builder_uses_verified_video_duration_to_exclude_tail_event(tmp_path) -> None:
    records, splits, selected, files, media_root = fixture(tmp_path)
    records[0]["events"].append(event(5, "Correct Action", 2.1, 2.3))

    rows, report = build_event_rows(
        records=records,
        split_ids=splits,
        selected_sessions=selected,
        media_files=files,
        media_root=media_root,
        media_video_durations={("train", "tr-1"): 2.2, ("validation", "va-1"): 3.25},
    )

    assert len(rows["train"]) == 2
    assert rows["train"][0]["video_duration_seconds"] == 2.2
    assert report["train"]["excluded_event_counts"]["out_of_media_interval_excluded"] == 1


def test_builder_disambiguates_reused_upstream_event_ids_by_record_position(tmp_path) -> None:
    records, splits, selected, files, media_root = fixture(tmp_path)
    records[0]["events"].append(
        event(1, "Wrong Action, corrected by instructor verbally", 2.1, 2.3)
    )

    rows, _ = build_event_rows(
        records=records,
        split_ids=splits,
        selected_sessions=selected,
        media_files=files,
        media_root=media_root,
    )

    assert [row["id"] for row in rows["train"] if row["id"].startswith("task-tr-1:1:")] == [
        "task-tr-1:1:0",
        "task-tr-1:1:4",
    ]


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


def test_av_validation_gate_requires_exact_selected_media_hashes() -> None:
    report = {
        "status": "PASS_HOLOASSIST_TRAINVAL_AV_MEDIA",
        "selection_sha256": "selection",
        "media_manifest_sha256": "manifest",
        "media": [
            {"split": "train", "session_id": "tr-1", "media_sha256": "sha-tr"},
            {
                "split": "validation",
                "session_id": "va-1",
                "media_sha256": "sha-va",
                "video_duration_seconds": 3.25,
            },
        ],
    }
    report["media"][0]["video_duration_seconds"] = 3.25
    validate_av_validation_report(
        report,
        selection_sha256="selection",
        media_manifest_sha256="manifest",
        expected_media={
            ("train", "tr-1", "sha-tr"),
            ("validation", "va-1", "sha-va"),
        },
    )

    report["media"].pop()
    with pytest.raises(ValueError, match="missing or duplicate|exactly cover"):
        validate_av_validation_report(
            report,
            selection_sha256="selection",
            media_manifest_sha256="manifest",
            expected_media={
                ("train", "tr-1", "sha-tr"),
                ("validation", "va-1", "sha-va"),
            },
        )

    report["media"].append(report["media"][0])
    with pytest.raises(ValueError, match="missing or duplicate"):
        validate_av_validation_report(
            report,
            selection_sha256="selection",
            media_manifest_sha256="manifest",
            expected_media={
                ("train", "tr-1", "sha-tr"),
                ("validation", "va-1", "sha-va"),
            },
        )


def test_av_validation_gate_requires_verified_video_durations() -> None:
    report = {
        "status": "PASS_HOLOASSIST_TRAINVAL_AV_MEDIA",
        "selection_sha256": "selection",
        "media_manifest_sha256": "manifest",
        "media": [
            {
                "split": "train",
                "session_id": "tr-1",
                "media_sha256": "sha-tr",
                "video_duration_seconds": float("nan"),
            }
        ],
    }

    with pytest.raises(ValueError, match="missing or invalid video durations"):
        validate_av_validation_report(
            report,
            selection_sha256="selection",
            media_manifest_sha256="manifest",
            expected_media={("train", "tr-1", "sha-tr")},
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
