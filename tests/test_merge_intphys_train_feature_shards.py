import hashlib
import json

import numpy as np
import pytest

from scripts.extract_intphys_train_features import TRAIN_ARCHIVE_PINS
from scripts.merge_intphys_train_feature_shards import merge_feature_shards


def _write_shard(directory, shard_number, *, duplicate_first=False, frame_count=8):
    from scripts.audit_videoprism_checkpoint import CHECKPOINT_SHA256, MODEL_REVISION
    from scripts.extract_frozen_videoprism_features import SOURCE_COMMIT
    from scripts.merge_intphys_train_feature_shards import sha256_file

    directory.mkdir()
    sample_ids = [f"{index:05d}" for index in range(12)]
    status_hashes = [hashlib.sha256(f"status-{shard_number}-{i}".encode()).hexdigest()
                     for i in range(12)]
    rgb_hashes = [hashlib.sha256(f"rgb-{shard_number}-{i}".encode()).hexdigest()
                  for i in range(12)]
    sampled_hashes = [hashlib.sha256(f"sampled-{shard_number}-{i}".encode()).hexdigest()
                      for i in range(12)]
    if duplicate_first:
        status_hashes[0] = hashlib.sha256(b"shared-status").hexdigest()
        rgb_hashes[0] = hashlib.sha256(b"shared-rgb").hexdigest()
        sampled_hashes[0] = hashlib.sha256(b"shared-sampled").hexdigest()
    for split_name, indices in (("train", range(0, 9)), ("validation", range(9, 12))):
        indexes = np.asarray(list(indices))
        path = directory / f"{split_name}-features.npz"
        np.savez_compressed(
            path,
            features=np.full((len(indexes), frame_count, 768), shard_number, dtype=np.float32),
            sample_ids=np.asarray([sample_ids[i] for i in indexes]),
            group_ids=np.asarray([f"old-group-{shard_number}-{i}" for i in indexes]),
            status_identity_sha256=np.asarray([status_hashes[i] for i in indexes]),
            rgb_sequence_sha256=np.asarray([rgb_hashes[i] for i in indexes]),
            sampled_rgb_pixels_sha256=np.asarray([sampled_hashes[i] for i in indexes]),
        )
    pin = TRAIN_ARCHIVE_PINS[shard_number]
    report = {
        "status": "complete_intphys_train_frozen_feature_extraction",
        "archive_shard_number": shard_number,
        "archive_source": pin["url"],
        "archive_bytes": pin["bytes"],
        "archive_md5": pin["md5"],
        "archive_sha256": hashlib.sha256(f"archive-{shard_number}".encode()).hexdigest(),
        "archive_videos": 12,
        "all_train_labels_possible": True,
        "frame_count": frame_count,
        "model_revision": MODEL_REVISION,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "videoprism_source_commit": SOURCE_COMMIT,
        "feature_archive_sha256": {
            name: sha256_file(directory / f"{name}-features.npz")
            for name in ("train", "validation")
        },
    }
    (directory / "extract-report.json").write_text(
        json.dumps(report), encoding="utf-8"
    )


def test_merger_globally_groups_duplicate_media_across_shards(tmp_path):
    first = tmp_path / "shard-1"
    second = tmp_path / "shard-2"
    _write_shard(first, 1, duplicate_first=True)
    _write_shard(second, 2, duplicate_first=True)
    output = tmp_path / "merged"

    report = merge_feature_shards(
        [first, second], output, seed=17, required_shards=(1, 2)
    )

    assert report["sample_count"] == 24
    assert report["unique_identity_group_count"] == 23
    assert report["cross_shard_duplicate_records_grouped"] == 1
    assert report["cross_split_group_overlap"] == 0
    assert report["cross_split_status_identity_overlap"] == 0
    assert report["cross_split_rgb_sequence_identity_overlap"] == 0
    assert report["cross_split_sampled_rgb_input_overlap"] == 0
    with np.load(output / "train-features.npz", allow_pickle=False) as train:
        train_ids = set(train["sample_ids"].astype(str))
        train_status = set(train["status_identity_sha256"].astype(str))
        train_media = set(train["rgb_sequence_sha256"].astype(str))
        train_sampled = set(train["sampled_rgb_pixels_sha256"].astype(str))
    with np.load(output / "validation-features.npz", allow_pickle=False) as validation:
        validation_ids = set(validation["sample_ids"].astype(str))
        validation_status = set(validation["status_identity_sha256"].astype(str))
        validation_media = set(validation["rgb_sequence_sha256"].astype(str))
        validation_sampled = set(validation["sampled_rgb_pixels_sha256"].astype(str))
    assert train_ids and validation_ids
    assert not train_ids & validation_ids
    assert not train_status & validation_status
    assert not train_media & validation_media
    assert not train_sampled & validation_sampled


def test_merger_rejects_missing_shards_and_does_not_create_output(tmp_path):
    shard = tmp_path / "shard-1"
    _write_shard(shard, 1)
    output = tmp_path / "merged"

    with pytest.raises(ValueError, match="expected exactly official shards"):
        merge_feature_shards([shard], output)
    assert not output.exists()


def test_merger_rejects_inconsistent_frame_policy(tmp_path):
    first = tmp_path / "shard-1"
    second = tmp_path / "shard-2"
    _write_shard(first, 1)
    _write_shard(second, 2, frame_count=16)

    with pytest.raises(ValueError, match="8-frame policy"):
        merge_feature_shards(
            [first, second], tmp_path / "merged", required_shards=(1, 2)
        )


def test_merger_rejects_modified_cache_before_split(tmp_path):
    first = tmp_path / "shard-1"
    second = tmp_path / "shard-2"
    _write_shard(first, 1)
    _write_shard(second, 2)
    with (second / "train-features.npz").open("ab") as stream:
        stream.write(b"corrupt")

    with pytest.raises(ValueError, match="does not match the shard receipt"):
        merge_feature_shards(
            [first, second], tmp_path / "merged", required_shards=(1, 2)
        )
