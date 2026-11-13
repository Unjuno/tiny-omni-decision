import hashlib

import pytest

from scripts.freeze_perception_source_candidate_controls import build_split_controls
from scripts.prepare_perception_source_candidate_features import prepare_feature_plan


def _row(record_id, video_id, split="train"):
    candidates = [
        {
            "candidate_id": f"{record_id}-option-{index}",
            "frame_boxes": [
                {
                    "frame_id": frame_id,
                    "bbox_xyxy_normalized": [0.0, 0.0, 0.5, 0.5],
                }
                for frame_id in (2, 7, 9)
            ],
        }
        for index in range(2)
    ]
    return {
        "record_id": record_id,
        "group_id": video_id,
        "status": "candidate",
        "split": split,
        "request": {
            "audio_interval_us": [1_000_000, 2_500_000],
            "candidates": candidates,
        },
        "supervision": {
            "target_candidate_ids": [candidates[0]["candidate_id"]],
            "none_visible": False,
        },
    }


def _clip(video_id):
    return {
        "metadata": {
            "video_id": video_id,
            "split": "train",
            "resolution": [1080, 1920],
            "frame_rate": 10,
            "num_frames": 100,
            "audio_samples": 160_000,
            "audio_sample_rate": 16_000,
        },
        "sha256": hashlib.sha256(video_id.encode()).hexdigest(),
        "bytes": 100,
    }


def _controls(records, split, source_hash):
    return build_split_controls(
        records,
        {row["group_id"]: _clip(row["group_id"]) for row in records},
        seed=17,
        split_name=split,
        source_sha256=source_hash,
    )


def _media_report(records, source_hash, acquisition_sha, repairs=None):
    repair_map = repairs or {}
    return {
        "status": "PASS_WITH_EXPLICIT_TERMINAL_FRAME_REPAIR" if repair_map else "PASS",
        "selection": "explicit_ordered_record_ids",
        "selection_source_jsonl_sha256": source_hash,
        "acquisition_manifest_sha256": acquisition_sha,
        "selected_record_count": len(records),
        "records": [
            {
                "record_id": row["record_id"],
                "video_id": row["group_id"],
                "media_sha256": hashlib.sha256(row["group_id"].encode()).hexdigest(),
                "terminal_frame_repair": repair_map.get(row["record_id"]),
            }
            for row in records
        ],
        "terminal_frame_repairs": list(repair_map.values()),
    }


def test_feature_plan_is_label_free_and_preserves_every_fixed_input():
    train = [_row("tr1", "v1"), _row("tr2", "v2")]
    development = [_row("dv1", "v3"), _row("dv2", "v4")]
    clips = {f"v{i}": _clip(f"v{i}") for i in range(1, 5)}
    train_sha, dev_sha = "b" * 64, "c" * 64
    plan = prepare_feature_plan(
        train,
        development,
        clips,
        _controls(train, "train", train_sha),
        _controls(development, "development", dev_sha),
        train_jsonl_sha256=train_sha,
        development_jsonl_sha256=dev_sha,
        acquisition_manifest_sha256="d" * 64,
        train_media_report=_media_report(train, train_sha, "d" * 64),
        development_media_report=_media_report(development, dev_sha, "d" * 64),
        train_media_report_sha256="e" * 64,
        development_media_report_sha256="f" * 64,
    )
    assert plan["video_encoder"]["letterbox_frame_size"] == 288
    assert plan["video_encoder"]["token_grid_size"] == 16
    assert plan["audio_encoder"]["window_seconds"] == 2
    for split, source_rows in (("train", train), ("development", development)):
        result = plan[split]
        assert result["record_count"] == 2
        assert {row["record_id"] for row in result["records"]} == {
            row["record_id"] for row in source_rows
        }
        assert all("supervision" not in row for row in result["records"])
        first = result["records"][0]
        assert len(first["audio_windows"]) == 1
        assert len(first["shifted_audio_windows"]) == 1
        assert first["video_id"] != first["audio_shuffle_donor_video_id"]
        assert first["candidates"][0]["frames"][0]["window_slot"] == [0, 0]
        assert first["candidates"][0]["frames"][0]["roi_xyxy_tokens"] == pytest.approx(
            [0.0, 3.5, 8.0, 8.0]
        )


def test_feature_plan_rejects_video_overlap_between_splits():
    train = [_row("tr1", "same"), _row("tr2", "v2")]
    development = [_row("dv1", "same"), _row("dv2", "v3")]
    controls_train = _controls(train, "train", "b" * 64)
    controls_dev = _controls(development, "development", "c" * 64)
    with pytest.raises(ValueError, match="split video groups overlap"):
        prepare_feature_plan(
            train,
            development,
            {video_id: _clip(video_id) for video_id in ("same", "v2", "v3")},
            controls_train,
            controls_dev,
            train_jsonl_sha256="b" * 64,
            development_jsonl_sha256="c" * 64,
            acquisition_manifest_sha256="d" * 64,
            train_media_report=_media_report(train, "e" * 64, "d" * 64),
            development_media_report=_media_report(development, "c" * 64, "d" * 64),
            train_media_report_sha256="e" * 64,
            development_media_report_sha256="f" * 64,
        )


def test_feature_plan_rejects_stale_control_hashes():
    train = [_row("tr1", "v1"), _row("tr2", "v2")]
    development = [_row("dv1", "v3"), _row("dv2", "v4")]
    with pytest.raises(ValueError, match="control source hash mismatch"):
        prepare_feature_plan(
            train,
            development,
            {f"v{i}": _clip(f"v{i}") for i in range(1, 5)},
            _controls(train, "train", "b" * 64),
            _controls(development, "development", "c" * 64),
            train_jsonl_sha256="e" * 64,
            development_jsonl_sha256="c" * 64,
            acquisition_manifest_sha256="d" * 64,
            train_media_report=_media_report(train, "e" * 64, "d" * 64),
            development_media_report=_media_report(development, "c" * 64, "d" * 64),
            train_media_report_sha256="e" * 64,
            development_media_report_sha256="f" * 64,
        )


def test_feature_plan_applies_preflighted_frame_repairs_without_dropping_targets():
    train = [_row("tr1", "v1"), _row("tr2", "v2")]
    development = [_row("dv1", "v3"), _row("dv2", "v4")]
    clips = {f"v{i}": _clip(f"v{i}") for i in range(1, 5)}
    train_sha, dev_sha, acquisition_sha = "b" * 64, "c" * 64, "d" * 64
    repair = {
        "record_id": "tr1",
        "frame_id": 9,
        "candidate_ids": ["tr1-option-0", "tr1-option-1"],
        "retained_frame_count_by_candidate": {"tr1-option-0": 2, "tr1-option-1": 2},
        "declared_frame_count": 10,
        "decoded_frame_count": 9,
    }
    plan = prepare_feature_plan(
        train,
        development,
        clips,
        _controls(train, "train", train_sha),
        _controls(development, "development", dev_sha),
        train_jsonl_sha256=train_sha,
        development_jsonl_sha256=dev_sha,
        acquisition_manifest_sha256=acquisition_sha,
        train_media_report=_media_report(train, train_sha, acquisition_sha, {"tr1": repair}),
        development_media_report=_media_report(development, dev_sha, acquisition_sha),
        train_media_report_sha256="e" * 64,
        development_media_report_sha256="f" * 64,
    )
    tr1 = next(row for row in plan["train"]["records"] if row["record_id"] == "tr1")
    assert len(plan["train"]["records"]) == len(train)
    assert len(tr1["candidates"]) == 2
    assert [len(candidate["frames"]) for candidate in tr1["candidates"]] == [2, 2]
    assert plan["train"]["terminal_frame_repairs"] == [repair]
    assert "supervision" not in tr1


def test_feature_plan_rejects_media_report_for_different_fixed_split():
    train = [_row("tr1", "v1"), _row("tr2", "v2")]
    development = [_row("dv1", "v3"), _row("dv2", "v4")]
    with pytest.raises(ValueError, match="media report source hash mismatch"):
        prepare_feature_plan(
            train,
            development,
            {f"v{i}": _clip(f"v{i}") for i in range(1, 5)},
            _controls(train, "train", "b" * 64),
            _controls(development, "development", "c" * 64),
            train_jsonl_sha256="b" * 64,
            development_jsonl_sha256="c" * 64,
            acquisition_manifest_sha256="d" * 64,
            train_media_report=_media_report(train, "x" * 64, "d" * 64),
            development_media_report=_media_report(development, "c" * 64, "d" * 64),
            train_media_report_sha256="e" * 64,
            development_media_report_sha256="f" * 64,
        )
