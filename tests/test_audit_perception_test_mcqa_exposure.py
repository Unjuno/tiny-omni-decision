from __future__ import annotations

import json
import zipfile

import pytest

from scripts.audit_perception_test_mcqa_exposure import (
    build_exposure_report,
    load_question_video_groups,
    scan_artifact_roots,
)


def _question(text="Which action is shown?", options=None, reasoning="descriptive"):
    return {
        "question": text,
        "options": options or ["running", "walking", "sitting"],
        "reasoning": reasoning,
    }


def _write_annotation_zip(path, split, rows):
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"all_{split}.json", json.dumps(rows))


def test_scan_collects_id_from_path_and_json_but_does_not_read_payload(tmp_path):
    root = tmp_path / "artifacts"
    run = root / "perception-test-run-01"
    feature = run / "features" / "validation" / "video_17.json"
    feature.parent.mkdir(parents=True)
    feature.write_text(json.dumps({"video_id": 22}), encoding="utf-8")
    media = run / "media" / "video_31.mp4"
    media.parent.mkdir()
    media.write_bytes(b"opaque media payload")
    readme = run / "README.md"
    readme.write_text("video_99 is only a documentation example", encoding="utf-8")
    cut_map = run / "cut_frame_mapping_valid.json"
    cut_map.write_text('{"video_id": "video_55"}', encoding="utf-8")
    sealed = run / "test" / "video_77.json"
    sealed.parent.mkdir()
    sealed.write_text('{"video_id": "video_77"}', encoding="utf-8")
    unrelated = root / "unrelated-run" / "video_44.json"
    unrelated.parent.mkdir()
    unrelated.write_text("{}", encoding="utf-8")
    prior_audit = root / "perception-test-mcqa-exposure-audit-attempt-01" / "report.json"
    prior_audit.parent.mkdir()
    prior_audit.write_text('{"video_id": "video_88"}', encoding="utf-8")

    result = scan_artifact_roots([root])

    assert result["unique_video_ids"] == ["video_17", "video_22", "video_31", "video_55"]
    assert result["media_and_model_payload_bytes_opened"] is False
    assert "video_31" in result["evidence_by_video_id"]
    assert "video_44" not in result["evidence_by_video_id"]
    assert "video_99" not in result["evidence_by_video_id"]
    assert "video_77" not in result["evidence_by_video_id"]
    assert "video_88" not in result["evidence_by_video_id"]


def test_load_groups_and_build_report_quarantines_all_artifact_mentions(tmp_path):
    train_path = tmp_path / "train.zip"
    valid_path = tmp_path / "valid.zip"
    _write_annotation_zip(
        train_path,
        "train",
        {
            "video_1": {
                "metadata": {"split": "train", "video_id": "video_1"},
                "mc_question": [_question(), _question("What happens next?")],
            },
            "video_2": {
                "metadata": {"split": "train", "video_id": "video_2"},
                "mc_question": [_question("What object is visible?")],
            },
        },
    )
    _write_annotation_zip(
        valid_path,
        "valid",
        {
            "video_3": {
                "metadata": {"split": "valid", "video_id": "video_3"},
                "mc_question": [
                    _question(),
                    _question("Which action is shown?", ["SITTING", "RUNNING", "WALKING"]),
                    _question("What happens next?", ["running", "walking", "sitting"]),
                ],
            },
            "video_4": {
                "metadata": {"split": "valid", "video_id": "video_4"},
                "mc_question": [
                    _question(),
                    _question("Why does this happen?", reasoning="explanatory"),
                ],
            },
        },
    )
    train = load_question_video_groups(train_path, split="train")
    valid = load_question_video_groups(valid_path, split="valid")
    artifact_scan = {
        "unique_video_ids": ["video_1", "video_3"],
        "evidence_by_video_id": {
            "video_1": ["run/features/train/video_1.npy"],
            "video_3": ["run/valid/video_3_predictions.json"],
            "video_4": ["run/cut_frame_mapping_valid.json"],
        },
        "artifact_roots": [],
        "media_and_model_payload_bytes_opened": False,
    }

    report = build_exposure_report(train, valid, artifact_scan)

    assert report["official_train_valid_video_id_overlap"] == 0
    assert report["prior_artifact_video_id_overlap"] == {
        "train_video_groups": 1,
        "valid_video_groups": 1,
    }
    assert report["unexposed_by_artifact_inventory"]["train"] == {
        "video_groups": 1,
        "questions": 1,
    }
    assert report["unexposed_by_artifact_inventory"]["valid"] == {
        "video_groups": 1,
        "questions": 2,
    }
    assert report["direct_artifact_video_id_overlap"] == {
        "train_video_groups": 1,
        "valid_video_groups": 1,
    }
    assert report["validation_candidates_without_direct_artifact_match"] == {
        "video_groups": 1,
        "questions": 2,
        "video_ids": ["video_4"],
        "interpretation": (
            "Candidate pool only: absence of a matching local media/feature/prediction/cohort path "
            "is not proof of prior non-exposure, semantic novelty, or split suitability."
        ),
    }
    deduplicated = report["validation_candidates_after_train_content_deduplication"]
    assert deduplicated["video_groups_with_at_least_one_question"] == 1
    assert deduplicated["questions"] == 1
    assert deduplicated["cross_split_normalized_content_overlap_removed"] == 1
    assert deduplicated["video_ids"] == ["video_4"]
    assert deduplicated["question_types"] == {
        "explanatory": {"questions": 1, "video_groups": 1}
    }


def test_group_loader_rejects_official_test_split(tmp_path):
    with pytest.raises(ValueError, match="only train and valid"):
        load_question_video_groups(tmp_path / "unused.zip", split="test")


def test_group_report_fails_on_cross_split_video_overlap():
    groups = {"video_ids": ["video_1"], "question_count": 1, "questions_by_video": {"video_1": 1}}
    scan = {
        "unique_video_ids": [],
        "evidence_by_video_id": {},
        "artifact_roots": [],
        "media_and_model_payload_bytes_opened": False,
    }
    with pytest.raises(ValueError, match="overlap"):
        build_exposure_report(groups, groups, scan)
