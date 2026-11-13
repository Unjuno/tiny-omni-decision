import hashlib
import importlib
import json

import numpy as np
import pytest

prepare_module = importlib.import_module("scripts.prepare_perception_source_candidate_features")
build_validation_feature_plan = getattr(prepare_module, "build_validation_feature_plan", None)
eval_module = importlib.import_module(
    "scripts.evaluate_perception_source_candidate_validation_transfer"
)
option_prior_module = importlib.import_module(
    "scripts.evaluate_perception_source_candidate_option_prior"
)


def _record(record_id="v1:event:1", video_id="v1", split="validation"):
    candidates = [
        {
            "candidate_id": f"option_{index}",
            "frame_boxes": [{"frame_id": frame_id, "bbox_xyxy_normalized": [0.0, 0.0, 0.5, 0.5]}],
        }
        for index, frame_id in enumerate((2, 4))
    ]
    return {
        "record_id": record_id,
        "group_id": video_id,
        "status": "candidate",
        "split": split,
        "provenance": {"source_split": "official_validation", "upstream_cut_frame": -1},
        "request": {"audio_interval_us": [1_000_000, 2_500_000], "candidates": candidates},
        "supervision": {"target_candidate_ids": ["option_0"], "none_visible": False},
    }


def _clip(video_id="v1", split="validation"):
    return {
        "metadata": {"video_id": video_id, "split": split, "resolution": [100, 200]},
        "sha256": hashlib.sha256(video_id.encode()).hexdigest(),
        "bytes": 123,
    }


def _report(record, source_hash="a" * 64, acquisition_hash="b" * 64):
    return {
        "status": "PASS",
        "selection": "explicit_ordered_record_ids",
        "selection_source_jsonl_sha256": source_hash,
        "acquisition_manifest_sha256": acquisition_hash,
        "selected_record_count": 1,
        "records": [
            {
                "record_id": record["record_id"],
                "video_id": record["group_id"],
                "media_sha256": hashlib.sha256(record["group_id"].encode()).hexdigest(),
                "decoded_frame_ids": [2, 4],
                "terminal_frame_repair": None,
            }
        ],
    }


def _build(record):
    assert callable(build_validation_feature_plan), "validation plan builder is missing"
    return build_validation_feature_plan(
        [record],
        {"v1": _clip()},
        _report(record),
        record_jsonl_sha256="a" * 64,
        acquisition_manifest_sha256="b" * 64,
        media_report_sha256="c" * 64,
        selection_sha256="d" * 64,
    )


def test_validation_feature_plan_is_label_free_and_binds_official_provenance():
    plan = _build(_record())
    assert plan["format"] == "perception_test_source_candidate_validation_feature_plan_v1"
    assert plan["validation"]["record_count"] == 1
    planned = plan["validation"]["records"][0]
    assert "supervision" not in planned
    assert "target_candidate_ids" not in planned
    assert planned["video_id"] == "v1"
    assert len(planned["candidates"]) == 2
    assert plan["validation"]["videos"][0]["frame_ids"] == [2, 4]


def test_validation_plan_rejects_non_official_or_non_validation_rows():
    record = _record()
    record["provenance"]["source_split"] = "train"
    assert callable(build_validation_feature_plan), "validation plan builder is missing"
    with pytest.raises(ValueError, match="official validation"):
        _build(record)


def test_validation_plan_rejects_same_video_twice():
    first = _record()
    second = _record("v1:event:2")
    assert callable(build_validation_feature_plan), "validation plan builder is missing"
    with pytest.raises(ValueError, match="one event per video"):
        build_validation_feature_plan(
            [first, second],
            {"v1": _clip()},
            _report(first),
            record_jsonl_sha256="a" * 64,
            acquisition_manifest_sha256="b" * 64,
            media_report_sha256="c" * 64,
            selection_sha256="d" * 64,
        )


def test_validation_feature_join_uses_labels_only_after_feature_plan(tmp_path):
    plan = _build(_record())
    video_root, audio_root = tmp_path / "video", tmp_path / "audio"
    video_root.mkdir()
    audio_root.mkdir()
    np.save(video_root / "candidate_frame_roi_features.npy", np.ones((2, 768), np.float32))
    (video_root / "feature_index.jsonl").write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {
                    "feature_index": 0,
                    "record_id": "v1:event:1",
                    "video_id": "v1",
                    "candidate_id": "option_0",
                    "frame_id": 2,
                },
                {
                    "feature_index": 1,
                    "record_id": "v1:event:1",
                    "video_id": "v1",
                    "candidate_id": "option_1",
                    "frame_id": 4,
                },
            )
        ),
        encoding="utf-8",
    )
    np.save(audio_root / "audio_window_features.npy", np.ones((1, 384), np.float32))
    (audio_root / "record_window_references.jsonl").write_text(
        json.dumps({"record_id": "v1:event:1", "feature_index": 0, "variant": "source"}),
        encoding="utf-8",
    )
    example = eval_module.build_validation_examples(plan, [_record()], video_root, audio_root)[0]
    assert example.target_candidate_ids == frozenset({"option_0"})
    assert example.visual_mean_delta.shape == (2, 1536)
    assert example.audio.shape == (384,)


def test_validation_feature_join_rejects_label_ids_outside_fixed_plan(tmp_path):
    plan = _build(_record())
    with pytest.raises(ValueError, match="validation labels differ"):
        eval_module.build_validation_examples(plan, [], tmp_path, tmp_path)


def test_video_group_bootstrap_is_deterministic_and_rejects_repeated_groups():
    rows = [{"video_id": f"video_{index}", "top1_correct": index % 2 == 0} for index in range(10)]
    first = eval_module.video_group_bootstrap_accuracy_interval(rows, repetitions=100, seed=17)
    second = eval_module.video_group_bootstrap_accuracy_interval(rows, repetitions=100, seed=17)
    assert first == second
    assert first["video_groups"] == 10
    with pytest.raises(ValueError, match="one scored event per video"):
        eval_module.video_group_bootstrap_accuracy_interval([rows[0], rows[0]])


def test_paired_group_bootstrap_requires_matching_video_ids_and_is_deterministic():
    joint = [
        {
            "record_id": f"r{index}",
            "video_id": f"v{index}",
            "top1_correct": index % 2 == 0,
            "nll": 0.5 + index / 100,
            "candidate_ids": ["a", "b"],
            "target_candidate_ids": ["a"],
            "none_visible": False,
            "candidate_probabilities": {"a": 0.6, "b": 0.3},
            "none_visible_probability": 0.1,
        }
        for index in range(8)
    ]
    altered = [{**row, "top1_correct": False, "nll": row["nll"] + 0.1} for row in joint]
    first = eval_module.paired_video_group_bootstrap_differences(
        {"joint": joint, "control": altered}, repetitions=100, seed=17
    )
    second = eval_module.paired_video_group_bootstrap_differences(
        {"joint": joint, "control": altered}, repetitions=100, seed=17
    )
    assert first == second
    assert (
        first["comparisons"]["control"]["delta_comparison_minus_reference"]["top1_set_accuracy"]
        == -0.5
    )
    with pytest.raises(ValueError, match="do not match reference IDs"):
        eval_module.paired_video_group_bootstrap_differences(
            {"joint": joint, "control": altered[1:]}, repetitions=10
        )


def test_option_slot_prior_uses_training_labels_and_rejects_video_overlap():
    train = [
        {
            "record_id": "train_1",
            "group_id": "train_video_1",
            "request": {"candidates": [{"candidate_id": "option_0"}, {"candidate_id": "option_1"}]},
            "supervision": {"target_candidate_ids": ["option_1"], "none_visible": False},
        },
        {
            "record_id": "train_2",
            "group_id": "train_video_2",
            "request": {"candidates": [{"candidate_id": "option_0"}, {"candidate_id": "option_1"}]},
            "supervision": {"target_candidate_ids": [], "none_visible": True},
        },
    ]
    validation = [
        {
            "record_id": "valid_1",
            "group_id": "valid_video_1",
            "status": "candidate",
            "split": "validation",
            "request": {"candidates": [{"candidate_id": "option_0"}, {"candidate_id": "option_1"}]},
            "supervision": {"target_candidate_ids": ["option_1"], "none_visible": False},
        }
    ]
    predictions, prior = option_prior_module.score_option_slot_prior(train, validation)
    assert (
        predictions[0]["candidate_probabilities"]["option_1"]
        > predictions[0]["candidate_probabilities"]["option_0"]
    )
    assert prior["training_video_ids"] == {"train_video_1", "train_video_2"}
    overlap = [dict(validation[0], group_id="train_video_1")]
    with pytest.raises(ValueError, match="video groups overlap"):
        option_prior_module.score_option_slot_prior(train, overlap)
