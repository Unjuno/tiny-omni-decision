from __future__ import annotations

import json
import zipfile

import pytest

from scripts.audit_avatar_task_support import audit_task_support


def _frame(root, source_group: str, task: str, categories: list[str]) -> None:
    video_id = f"{source_group}_72"
    folder = root / f"{source_group}_00072"
    folder.mkdir(parents=True, exist_ok=True)
    annotations = [
        {
            "task": task,
            "audio_visual_category": category,
            "bbox": [1, 1, 2, 2] if task != "Off-Screen" else None,
        }
        for category in categories
    ]
    (folder / "00001.json").write_text(
        json.dumps({"video_id": video_id, "annotations": annotations}),
        encoding="utf-8",
    )


def test_task_support_is_grouped_deterministic_and_aggregate_only(tmp_path):
    train = tmp_path / "train"
    groups = ["abcdefghijk", "bcdefghijkl", "cdefghijklm", "defghijklmn", "efghijklmno"]
    for group in groups:
        _frame(train, group, "Mixed-Sound", ["Guitar", "Piano"])
    _frame(train, groups[0], "Multi-Entity", ["Drum"])

    first = audit_task_support(train, previously_exposed_groups={groups[0]}, seed=17)
    second = audit_task_support(train, previously_exposed_groups={groups[0]}, seed=17)

    assert first == second
    assert first["source_group_ids_emitted"] is False
    assert first["official_validation_or_sealed_audit_accessed"] is False
    mixed = first["tasks"]["Mixed-Sound"]
    assert mixed["previously_unexposed_source_group_count"] == 4
    split = mixed["illustrative_group_split_not_frozen"]
    assert split["train_group_count"] == 3
    assert split["validation_group_count"] == 1
    assert split["overlap_count"] == 0
    assert split["frames_with_multiple_categories"] == 4
    assert "abcdefghijk" not in json.dumps(first)


def test_task_support_requires_isolated_train_directory(tmp_path):
    audit = tmp_path / "audit"
    audit.mkdir()
    with pytest.raises(ValueError, match="isolated TRAIN"):
        audit_task_support(audit, previously_exposed_groups=set())


def test_global_group_assignment_is_shared_across_tasks(tmp_path):
    train = tmp_path / "train"
    shared = "abcdefghijk"
    for group, task in (
        (shared, "Mixed-Sound"),
        (shared, "Multi-Entity"),
        ("bcdefghijkl", "Mixed-Sound"),
        ("cdefghijklm", "Multi-Entity"),
        ("defghijklmn", "Mixed-Sound"),
    ):
        _frame(train, group, task, ["Guitar"])

    report = audit_task_support(
        train,
        previously_exposed_groups=set(),
        seed=17,
        validation_fraction=0.4,
    )
    mixed = report["tasks"]["Mixed-Sound"]["illustrative_group_split_not_frozen"]
    multi = report["tasks"]["Multi-Entity"]["illustrative_group_split_not_frozen"]

    assert mixed["assignment_unit"] == "source_group_global_across_tasks"
    assert mixed["global_train_group_set_sha256"] == multi["global_train_group_set_sha256"]
    assert mixed["global_validation_group_set_sha256"] == multi[
        "global_validation_group_set_sha256"
    ]
    assert mixed["global_train_group_count"] + mixed["global_validation_group_count"] == 4
    assert mixed["overlap_count"] == multi["overlap_count"] == 0


def test_media_zip_inventory_uses_central_directory_and_reports_missing(tmp_path):
    train = tmp_path / "train"
    groups = ["abcdefghijk", "bcdefghijkl"]
    for group in groups:
        _frame(train, group, "Mixed-Sound", ["Guitar"])
    media_zip = tmp_path / "video.zip"
    with zipfile.ZipFile(media_zip, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(f"clips/{groups[0]}_72.mp4", b"video-payload-not-read")

    report = audit_task_support(
        train,
        previously_exposed_groups=set(),
        seed=17,
        validation_fraction=0.5,
        media_zip=media_zip,
    )
    media = report["media_zip_inventory"]
    mixed = media["task_media"]["Mixed-Sound"]

    assert media["zip_member_payloads_opened"] is False
    assert mixed["expected_annotated_clip_count"] == 2
    assert mixed["matched_clip_count"] == 1
    assert mixed["missing_clip_count"] == 1
    assert mixed["uncompressed_bytes"] == len(b"video-payload-not-read")
