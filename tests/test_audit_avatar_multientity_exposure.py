import json
from pathlib import Path

import pytest

from scripts.audit_avatar_multientity_exposure import (
    _source_group_from_annotation,
    build_report,
    provisional_group_split,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _annotation(root: Path, source: str, task: str) -> None:
    clip = f"{source}_0"
    path = root / clip / "00000.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "video_id": clip,
                "annotations": [
                    {
                        "task": task,
                        "audio_visual_category": "test-category",
                        "bbox": [0, 0, 1, 1],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _artifacts(root: Path) -> None:
    for index in range(1, 5):
        directory = root / "avatar-av-decision-20261101-attempt-01" / f"option-choices-{index:02d}"
        _write_jsonl(
            directory / "train-analysis-sidecar.jsonl",
            [{"audio_source_group": "AAAAAAAAAAA", "option_source_groups": ["BBBBBBBBBBB"]}],
        )
        _write_jsonl(
            directory / "validation-analysis-sidecar.jsonl",
            [{"audio_source_group": "CCCCCCCCCCC", "option_source_groups": ["DDDDDDDDDDD"]}],
        )
    corrected = root / "avatar-av-decision-20261102-attempt-01" / "cross-source-option-choices-02"
    _write_jsonl(corrected / "train-analysis-sidecar.jsonl", [])
    _write_jsonl(corrected / "validation-analysis-sidecar.jsonl", [])

    category = root / "avatar-category-transfer-20261102-attempt-03"
    category.mkdir(parents=True)
    (category / "split-manifest.json").write_text(
        json.dumps(
            {
                "splits": {
                    "fit": {"query_groups": ["EEEEEEEEEEE"]},
                    "inner_validation": {"query_groups": ["FFFFFFFFFFF"]},
                    "held_category_test": {"query_groups": ["GGGGGGGGGGG"]},
                }
            }
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        category / "held-category-predictions.jsonl",
        [{"audio_source_group": "HHHHHHHHHHH", "option_source_groups": ["IIIIIIIIIII"]}],
    )

    dino = root / "avatar-dinov2-train-fold-20261102-attempt-01"
    _write_jsonl(dino / "train-sidecar.jsonl", [])
    _write_jsonl(dino / "validation-sidecar.jsonl", [])
    native = root / "avatar-native-localization-20261102-attempt-01" / "fold"
    _write_jsonl(native / "train.jsonl", [])
    _write_jsonl(native / "validation.jsonl", [])


def test_source_group_comes_from_validated_clip_directory() -> None:
    row = {"video_id": "AAAAAAAAAAA_7"}
    assert _source_group_from_annotation(Path("AAAAAAAAAAA_00007/00003.json"), row) == "AAAAAAAAAAA"


def test_source_group_rejects_mismatched_video_identity() -> None:
    with pytest.raises(ValueError, match="does not match"):
        _source_group_from_annotation(
            Path("AAAAAAAAAAA_00007/00003.json"), {"video_id": "BBBBBBBBBBB_7"}
        )


def test_provisional_group_split_is_deterministic_and_disjoint() -> None:
    groups = {f"{index:011d}" for index in range(100)}
    train_a, validation_a = provisional_group_split(groups)
    train_b, validation_b = provisional_group_split(groups)

    assert train_a == train_b
    assert validation_a == validation_b
    assert not train_a & validation_a
    assert train_a | validation_a == groups
    assert len(validation_a) == 20


def test_report_counts_only_unexposed_multientity_train_groups(tmp_path: Path) -> None:
    train = tmp_path / "train-validation-annotations-02" / "train"
    _annotation(train, "AAAAAAAAAAA", "Multi-Entity")
    _annotation(train, "JJJJJJJJJJJ", "Multi-Entity")
    _annotation(train, "KKKKKKKKKKK", "Single-Sound")
    artifacts = tmp_path / "artifacts"
    _artifacts(artifacts)

    report = build_report(train_annotations=train, artifact_root=artifacts)

    multi = report["train_only_multientity"]
    assert multi["source_group_count"] == 3
    assert multi["multi_entity_source_group_count"] == 2
    assert multi["fit_exposed_group_count"] == 1
    assert multi["unexposed_to_prior_fit_or_evaluation_group_count"] == 1
    assert report["sealed_audit_accessed"] is False
    assert report["official_validation_annotations_or_media_accessed"] is False
    assert report["source_group_ids_emitted"] is False
    assert "JJJJJJJJJJJ" not in json.dumps(report)


def test_report_refuses_non_train_annotation_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="isolated TRAIN"):
        build_report(
            train_annotations=tmp_path / "validation",
            artifact_root=tmp_path / "not-needed",
        )
