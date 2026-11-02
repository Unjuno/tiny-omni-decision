import json
from pathlib import Path

import pytest

from scripts.build_avatar_multientity_choice_corpus import build_corpus


def _write_jsonl(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def _make_artifact_root(root: Path) -> None:
    for index in range(1, 5):
        directory = root / "avatar-av-decision-20261101-attempt-01" / f"option-choices-{index:02d}"
        _write_jsonl(directory / "train-analysis-sidecar.jsonl")
        _write_jsonl(directory / "validation-analysis-sidecar.jsonl")
    corrected = root / "avatar-av-decision-20261102-attempt-01" / "cross-source-option-choices-02"
    _write_jsonl(corrected / "train-analysis-sidecar.jsonl")
    _write_jsonl(corrected / "validation-analysis-sidecar.jsonl")
    category = root / "avatar-category-transfer-20261102-attempt-03"
    category.mkdir(parents=True)
    (category / "split-manifest.json").write_text(
        json.dumps({"splits": {"fit": {}, "inner_validation": {}, "held_category_test": {}}}),
        encoding="utf-8",
    )
    _write_jsonl(category / "held-category-predictions.jsonl")
    dino = root / "avatar-dinov2-train-fold-20261102-attempt-01"
    _write_jsonl(dino / "train-sidecar.jsonl")
    _write_jsonl(dino / "validation-sidecar.jsonl")
    native = root / "avatar-native-localization-20261102-attempt-01" / "fold"
    _write_jsonl(native / "train.jsonl")
    _write_jsonl(native / "validation.jsonl")


def _write_source_train(root: Path) -> None:
    for category_index in range(5):
        for group_index in range(8):
            source_group = f"g{category_index}{group_index:09d}"
            video_id = f"{source_group}_0"
            clip = root / video_id
            clip.mkdir(parents=True, exist_ok=True)
            (clip / "00030.json").write_text(
                json.dumps(
                    {
                        "video_id": video_id,
                        "frame_number": 30,
                        "fps": 30.0,
                        "duration": 10.0,
                        "annotations": [
                            {
                                "task": "Multi-Entity",
                                "audio_visual_category": f"category-{category_index}",
                                "bbox": [0, 0, 1, 1],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )


def test_builds_four_choice_corpus_with_group_safe_splits(tmp_path: Path) -> None:
    train_annotations = tmp_path / "train-validation-annotations" / "train"
    train_annotations.mkdir(parents=True)
    _write_source_train(train_annotations)
    artifact_root = tmp_path / "artifacts"
    _make_artifact_root(artifact_root)
    output_dir = tmp_path / "frozen-corpus"

    manifest = build_corpus(
        train_annotations=train_annotations,
        artifact_root=artifact_root,
        output_dir=output_dir,
    )

    assert manifest["status"] == "METADATA_FROZEN_PENDING_MEDIA_PREFLIGHT"
    assert manifest["split"]["source_group_overlap"] == 0
    assert manifest["split"]["query_and_candidate_source_group_cross_split_overlap"] == 0
    assert manifest["task"]["timestamps_are_encoder_alignment_metadata_only"] is True
    assert manifest["task"]["question_is_constant_and_not_a_scorer_feature"] is True
    assert "IDs, timestamps, target labels, and sidecars are not head features" in manifest[
        "task"
    ]["scorer_feature_contract"]
    assert (
        manifest["metadata_only_controls"]["source_group_or_category_sidecars_exposed_to_scorer"]
        is False
    )
    position_prior = manifest["metadata_only_controls"]["option_position_prior"]
    train_counts = manifest["counts"]["train"]["target_option_position_counts"]
    validation_counts = manifest["counts"]["validation"]["target_option_position_counts"]
    selected_position = min(range(4), key=lambda index: (-train_counts[str(index)], index))
    assert position_prior["selected_position_from_train_only"] == selected_position
    assert position_prior["train_accuracy"] == train_counts[str(selected_position)] / manifest[
        "counts"
    ]["train"]["query_count"]
    assert position_prior["validation_accuracy"] == validation_counts[
        str(selected_position)
    ] / manifest["counts"]["validation"]["query_count"]
    assert (
        manifest["metadata_only_controls"]["serialized_row_order_modulo_four"][
            "train_accuracy"
        ]
        < 1.0
    )
    assert (
        manifest["metadata_only_controls"]["serialized_row_order_modulo_four"][
            "validation_accuracy"
        ]
        < 1.0
    )
    assert manifest["counts"]["train"]["query_count"] > 0
    assert manifest["counts"]["validation"]["query_count"] > 0
    for split in ("train", "validation"):
        positions = manifest["counts"][split]["target_option_position_counts"].values()
        assert max(positions) - min(positions) <= 1
    for split in ("train", "validation"):
        model_rows = [
            json.loads(line)
            for line in (
                output_dir / f"{split}_model_input.jsonl"
            ).read_text(encoding="utf-8").splitlines()
        ]
        sidecars = [
            json.loads(line)
            for line in (
                output_dir / f"{split}_analysis_sidecar.jsonl"
            ).read_text(encoding="utf-8").splitlines()
        ]
        assert len(model_rows) == len(sidecars)
        for model, sidecar in zip(model_rows, sidecars, strict=True):
            assert len(model["options"]) == 4
            assert sidecar["query_source_group"] not in sidecar["option_source_groups"]
            assert len(set(sidecar["option_source_groups"])) == 4
            assert sidecar["option_categories"][sidecar["target_option_index"]] == sidecar[
                "query_category"
            ]
            assert "query_category" not in model
            assert "option_categories" not in model

    replay_dir = tmp_path / "frozen-corpus-replay"
    replay = build_corpus(
        train_annotations=train_annotations,
        artifact_root=artifact_root,
        output_dir=replay_dir,
    )
    assert replay["files"] == manifest["files"]
    assert replay["dependency_diagnostics"] == manifest["dependency_diagnostics"]


def test_builder_refuses_non_train_annotation_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="isolated TRAIN"):
        build_corpus(
            train_annotations=tmp_path / "validation",
            artifact_root=tmp_path / "not-needed",
            output_dir=tmp_path / "output",
        )


def test_builder_refuses_to_overwrite_existing_output(tmp_path: Path) -> None:
    train_annotations = tmp_path / "train"
    train_annotations.mkdir()
    output_dir = tmp_path / "existing"
    output_dir.mkdir()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_corpus(
            train_annotations=train_annotations,
            artifact_root=tmp_path / "not-needed",
            output_dir=output_dir,
        )


def test_builder_fails_closed_when_prior_exposure_sidecar_is_missing(tmp_path: Path) -> None:
    train_annotations = tmp_path / "train"
    train_annotations.mkdir()
    artifact_root = tmp_path / "artifacts"
    _make_artifact_root(artifact_root)
    missing = (
        artifact_root
        / "avatar-av-decision-20261101-attempt-01"
        / "option-choices-01"
        / "train-analysis-sidecar.jsonl"
    )
    missing.unlink()

    with pytest.raises(FileNotFoundError, match="missing inventoried train sidecar"):
        build_corpus(
            train_annotations=train_annotations,
            artifact_root=artifact_root,
            output_dir=tmp_path / "output",
        )
