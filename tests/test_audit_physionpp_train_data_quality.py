import hashlib

import pytest

from scripts.audit_physionpp_train_data_quality import (
    audit_train_rows,
    select_visual_review_rows,
)


def _row(root, scenario, number, target, *, split="train"):
    row_id = f"readout_data_v1/{scenario}/trial#{number:04d}"
    relative_path = f"raw/physionpp/{scenario}/{number:04d}_img.mp4"
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = f"{scenario}-{number}".encode()
    path.write_bytes(payload)
    is_positive = target == "Yes"
    return {
        "id": row_id,
        "modality": "video",
        "source_record_id": row_id,
        "source_asset_group_id": f"{scenario}:seed={number}",
        "target": target,
        "source_target": is_positive,
        "split": split,
        "media": [
            {
                "kind": "video",
                "path": relative_path,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "num_frames": 120,
            }
        ],
    }


def test_train_audit_verifies_media_and_reports_source_group_quality(tmp_path):
    rows = [
        _row(tmp_path, "collision", 0, "No"),
        _row(tmp_path, "collision", 1, "Yes"),
    ]

    report = audit_train_rows(rows, tmp_path)

    assert report["train_rows"] == 2
    assert report["unique_video_sha256"] == 2
    assert report["unique_source_asset_groups"] == 2
    assert report["target_counts"] == {"No": 1, "Yes": 1}
    assert report["source_target_mismatches"] == 0
    assert report["scenario_class_counts"] == {"collision": {"No": 1, "Yes": 1}}


def test_train_audit_rejects_non_train_rows(tmp_path):
    row = _row(tmp_path, "collision", 0, "No", split="validation")

    with pytest.raises(ValueError, match="only non-empty train rows"):
        audit_train_rows([row], tmp_path)


def test_train_audit_rejects_media_hash_mismatch(tmp_path):
    row = _row(tmp_path, "collision", 0, "No")
    row["media"][0]["sha256"] = "0" * 64

    with pytest.raises(ValueError, match="hash mismatch"):
        audit_train_rows([row], tmp_path)


def test_visual_selection_is_deterministic_and_stratified_by_scenario_object_and_label(tmp_path):
    rows = [
        _row(tmp_path, "collision", i, target)
        for i, target in enumerate(["No", "No", "Yes", "Yes"])
    ] + [
        _row(tmp_path, "cloth", i + 10, target)
        for i, target in enumerate(["No", "Yes"])
    ]

    first = select_visual_review_rows(rows)
    second = select_visual_review_rows(list(reversed(rows)))

    assert [row["id"] for row in first] == [row["id"] for row in second]
    assert {(row["source_record_id"].split("/")[1], row["target"]) for row in first} == {
        ("collision", "No"),
        ("collision", "Yes"),
        ("cloth", "No"),
        ("cloth", "Yes"),
    }


def test_visual_selection_matches_target_object_and_skips_unmatched_strata(tmp_path):
    rows = []
    for number, (target_object, target) in enumerate(
        [
            ("cube", "No"),
            ("cube", "Yes"),
            ("sphere", "No"),
            ("sphere", "Yes"),
            ("cone", "No"),
        ]
    ):
        row = _row(tmp_path, "collision", number, target)
        row["source_record_id"] = (
            f"readout_data_v1/collision/collision-target={target_object}#"
            f"{number:04d}"
        )
        rows.append(row)

    selected = select_visual_review_rows(rows)

    assert {
        (_row["source_record_id"].split("target=")[1].split("#")[0], _row["target"])
        for _row in selected
    } == {("cube", "No"), ("cube", "Yes"), ("sphere", "No"), ("sphere", "Yes")}


def test_visual_selection_rejects_validation_rows():
    with pytest.raises(ValueError, match="only train rows"):
        select_visual_review_rows([{"split": "validation"}])
