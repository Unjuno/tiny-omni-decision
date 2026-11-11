from __future__ import annotations

import json
import zipfile

import pytest

from scripts.build_perception_test_mcqa_corpus import build_corpus, sha256_file


def _question(question_id, text, *, answer_id=0, reasoning="descriptive"):
    return {
        "id": question_id,
        "question": text,
        "options": ["first", "second", "third"],
        "answer_id": answer_id,
        "reasoning": reasoning,
        "area": "semantics",
    }


def _write_archive(path, split, videos):
    rows = {
        video_id: {
            "metadata": {"split": split, "video_id": video_id},
            "mc_question": questions,
        }
        for video_id, questions in videos.items()
    }
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"all_{split}.json", json.dumps(rows))


def _make_inputs(tmp_path):
    train_path = tmp_path / "train.zip"
    valid_path = tmp_path / "valid.zip"
    _write_archive(
        train_path,
        "train",
        {
            "video_train_a": [
                _question(0, "Shared prompt", answer_id=1),
                _question(1, "Training-only prompt"),
            ],
            "video_train_b": [_question(0, "Another training prompt")],
        },
    )
    _write_archive(
        valid_path,
        "valid",
        {
            "video_valid_a": [
                _question(0, "SHARED   PROMPT", answer_id=2),
                _question(1, "Validation-only prompt", reasoning="predictive"),
            ],
            "video_valid_b": [_question(0, "Shared prompt")],
        },
    )
    report = {
        "protocol": "test-exposure-audit",
        "official_train": {"annotation_archive_sha256": sha256_file(train_path)},
        "official_valid": {"annotation_archive_sha256": sha256_file(valid_path)},
        "validation_candidates_after_train_content_deduplication": {
            "video_ids": ["video_valid_a"]
        },
    }
    return train_path, valid_path, report


def test_builds_video_grouped_content_disjoint_corpus(tmp_path):
    train_path, valid_path, report = _make_inputs(tmp_path)

    train, valid, manifest = build_corpus(
        train_annotations=train_path,
        valid_annotations=valid_path,
        exposure_report=report,
        seed=17,
    )

    assert [row["id"] for row in train] == [
        "perception-test/train/video_train_a/0",
        "perception-test/train/video_train_a/1",
        "perception-test/train/video_train_b/0",
    ]
    assert [row["id"] for row in valid] == [
        "perception-test/valid/video_valid_a/1"
    ]
    assert valid[0]["reasoning_type"] == "predictive"
    assert valid[0]["target_option_index"] == 0
    assert manifest["group_integrity"]["train_validation_video_id_overlap"] == 0
    assert manifest["validation"]["normalized_content_overlap_with_train"] == 0
    assert manifest["validation"]["video_group_count"] == 1
    assert manifest["validation"]["reasoning_type_counts"] == {
        "predictive": {"questions": 1, "video_groups": 1}
    }
    assert manifest["data_policy"]["model_fitting_or_scoring_performed"] is False


def test_rejects_candidate_video_without_train_disjoint_question(tmp_path):
    train_path, valid_path, report = _make_inputs(tmp_path)
    report["validation_candidates_after_train_content_deduplication"]["video_ids"] = [
        "video_valid_a",
        "video_valid_b",
    ]

    with pytest.raises(ValueError, match="no train-content-disjoint"):
        build_corpus(
            train_annotations=train_path,
            valid_annotations=valid_path,
            exposure_report=report,
        )


def test_fails_closed_on_archive_hash_mismatch(tmp_path):
    train_path, valid_path, report = _make_inputs(tmp_path)
    report["official_valid"]["annotation_archive_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="archive hash differs"):
        build_corpus(
            train_annotations=train_path,
            valid_annotations=valid_path,
            exposure_report=report,
        )


def test_rejects_video_group_overlap(tmp_path):
    train_path, valid_path, report = _make_inputs(tmp_path)
    _write_archive(
        valid_path,
        "valid",
        {"video_train_a": [_question(0, "Validation-only prompt")]},
    )
    report["official_valid"]["annotation_archive_sha256"] = sha256_file(valid_path)
    report["validation_candidates_after_train_content_deduplication"]["video_ids"] = [
        "video_train_a"
    ]

    with pytest.raises(ValueError, match="video overlap"):
        build_corpus(
            train_annotations=train_path,
            valid_annotations=valid_path,
            exposure_report=report,
        )
