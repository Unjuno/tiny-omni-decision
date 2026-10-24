import json

import pytest

from scripts.freeze_perception_source_candidate_controls import (
    build_split_controls,
    freeze_controls,
)


def _record(record_id, video_id):
    candidates = [
        {
            "candidate_id": f"{record_id}-{index}",
            "frame_boxes": [{"frame_id": index * 10 + frame} for frame in range(4)],
        }
        for index in range(3)
    ]
    return {
        "record_id": record_id,
        "group_id": video_id,
        "status": "candidate",
        "split": "train",
        "request": {"audio_interval_us": [1_000_000, 1_500_000], "candidates": candidates},
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
            "audio_samples": 160_000,
            "audio_sample_rate": 16_000,
            "num_frames": 100,
            "frame_rate": 10,
        },
        "bytes": 1,
        "sha256": "a" * 64,
        "cut_frame": -1,
    }


def test_split_controls_are_deterministic_and_label_free():
    records = [_record("r1", "v1"), _record("r2", "v2"), _record("r3", "v3")]
    clips = {video_id: _clip(video_id) for video_id in ("v1", "v2", "v3")}
    first = build_split_controls(
        records, clips, seed=17, split_name="train", source_sha256="b" * 64
    )
    second = build_split_controls(
        records, clips, seed=17, split_name="train", source_sha256="b" * 64
    )
    assert first == second
    for row in first["records"]:
        assert row["video_id"] != row["audio_shuffle_donor_video_id"]
        assert "supervision" not in row
        assert set(row["candidate_order"]) == set(row["static_frame_id_by_candidate"])
        assert set(row["candidate_order"]) == set(row["reversed_frame_ids_by_candidate"])
        assert row["offset_audio_interval_us"] in ([0, 500_000], [2_000_000, 2_500_000])


def test_audio_shuffle_donor_is_different_video():
    rows = [_record("r1", "v1"), _record("r2", "v2")]
    result = build_split_controls(
        rows,
        {"v1": _clip("v1"), "v2": _clip("v2")},
        seed=17,
        split_name="development",
        source_sha256="c" * 64,
    )
    assert all(row["video_id"] != row["audio_shuffle_donor_video_id"] for row in result["records"])


def test_out_of_bounds_time_shift_is_marked_ineligible():
    rows = [_record("r1", "v1"), _record("r2", "v2")]
    rows[0]["request"]["audio_interval_us"] = [600_000, 1_100_000]
    clips = {"v1": _clip("v1"), "v2": _clip("v2")}
    clips["v1"]["metadata"]["audio_samples"] = 19_200
    result = build_split_controls(rows, clips, seed=17, split_name="train", source_sha256="e" * 64)
    ineligible = next(row for row in result["records"] if row["record_id"] == "r1")
    assert ineligible["offset_audio_interval_us"] is None
    assert result["offset_eligible_count"] == 1


def test_missing_media_receipt_fails_closed():
    with pytest.raises(ValueError, match="media receipt missing"):
        build_split_controls(
            [_record("r1", "v1"), _record("r2", "v2")],
            {"v1": _clip("v1")},
            seed=17,
            split_name="train",
            source_sha256="d" * 64,
        )


def test_train_development_video_overlap_fails_closed(tmp_path):
    train_path = tmp_path / "train.jsonl"
    dev_path = tmp_path / "dev.jsonl"
    receipt_path = tmp_path / "receipts.json"
    train_path.write_text(json.dumps(_record("r1", "shared")) + "\n", encoding="utf-8")
    dev_path.write_text(json.dumps(_record("r2", "shared")) + "\n", encoding="utf-8")
    receipt_path.write_text(json.dumps({"clips": [_clip("shared")]}), encoding="utf-8")
    with pytest.raises(ValueError, match="video groups overlap"):
        freeze_controls(train_path, dev_path, receipt_path)
