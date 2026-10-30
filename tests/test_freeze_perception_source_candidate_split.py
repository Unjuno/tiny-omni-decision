import hashlib

import pytest

from scripts.freeze_perception_source_candidate_split import freeze_split


def sample(video_id, record_num):
    return {
        "status": "candidate",
        "record_id": f"{video_id}:sound:{record_num}",
        "group_id": video_id,
        "split": "train",
        "media_reference": {"video_id": video_id},
        "provenance": {"upstream_cut_frame": -1},
        "request": {"audio_interval_us": [100_000, 200_000], "candidates": []},
    }


def clips(video_ids):
    return {
        video_id: {
            "sha256": hashlib.sha256(video_id.encode()).hexdigest(),
            "cut_frame": -1,
            "metadata": {
                "split": "train",
                "num_frames": 10,
                "frame_rate": 10.0,
                "audio_samples": 1_000,
                "audio_sample_rate": 100.0,
            },
        }
        for video_id in video_ids
    }


def test_split_is_deterministic_grouped_and_asset_disjoint():
    records = [sample(f"v{i}", j) for i in range(8) for j in range(i % 3 + 1)]
    media = clips([f"v{i}" for i in range(8)])
    first = freeze_split(records, media, validation_group_count=2, seed=17)
    second = freeze_split(records, media, validation_group_count=2, seed=17)
    assert first == second
    assert len(first["train_video_ids"]) == 6
    assert len(first["validation_video_ids"]) == 2
    assert not set(first["train_video_ids"]) & set(first["validation_video_ids"])
    assert not set(first["train_asset_sha256"]) & set(first["validation_asset_sha256"])
    validation_records = first["validation_records"]
    assert {record["group_id"] for record in validation_records} == set(
        first["validation_video_ids"]
    )


def test_duplicate_media_identity_is_rejected_before_split():
    media = clips(["v1", "v2"])
    media["v2"]["sha256"] = media["v1"]["sha256"]
    with pytest.raises(ValueError, match="duplicate media asset"):
        freeze_split(
            [sample("v1", 0), sample("v2", 0)], media, validation_group_count=1, seed=17
        )


def test_unknown_or_nontrain_video_is_rejected():
    media = clips(["v1", "v2"])
    media["v2"]["metadata"]["split"] = "valid"
    with pytest.raises(ValueError, match="official train"):
        freeze_split(
            [sample("v1", 0), sample("v2", 0)], media, validation_group_count=1, seed=17
        )


def test_validation_group_count_must_leave_both_partitions_nonempty():
    with pytest.raises(ValueError, match="between 1 and group_count - 1"):
        freeze_split([sample("v1", 0)], clips(["v1"]), validation_group_count=1, seed=17)
