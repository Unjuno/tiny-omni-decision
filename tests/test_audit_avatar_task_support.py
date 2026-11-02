from __future__ import annotations

import json

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
