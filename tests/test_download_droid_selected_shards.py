import hashlib
import json
from pathlib import Path

import pytest

from scripts.download_droid_selected_shards import validate_manifest, verify_metadata

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests/pretrained_reuse/droid-selected-shards-seed17-20261026.json"


def test_full_droid_selection_is_seed_pinned_and_metadata_consistent():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    assert len(manifest["selected_shards"]) == 12
    assert sum(row["episodes"] for row in manifest["selected_shards"]) == 539
    assert manifest["expected_total_bytes"] == 10_724_824_602


def test_selected_shards_reject_different_seed_or_episode_count():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest["selection_seed"] = 19
    with pytest.raises(ValueError, match="seed must remain 17"):
        validate_manifest(manifest)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest["expected_episode_count"] += 1
    with pytest.raises(ValueError, match="episode counts"):
        validate_manifest(manifest)


def test_selected_shards_reject_unexpected_object():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest["objects"][-1]["name"] = "robotics/droid/1.0.0/unpinned-file"
    with pytest.raises(ValueError, match="unexpected or duplicate"):
        validate_manifest(manifest)


def test_release_metadata_hashes_and_selected_shard_lengths_are_checked(tmp_path):
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    data_root = tmp_path / manifest["prefix"]
    data_root.mkdir(parents=True)
    lengths = ["1"] * 2048
    for row in manifest["selected_shards"]:
        lengths[row["index"]] = str(row["episodes"])
    dataset_info = {"version": "1.4.0", "splits": [{"name": "train", "shardLengths": lengths}]}
    info_bytes = json.dumps(dataset_info).encode()
    feature_bytes = b"pinned feature schema"
    (data_root / "dataset_info.json").write_bytes(info_bytes)
    (data_root / "features.json").write_bytes(feature_bytes)
    manifest["dataset_info_sha256"] = hashlib.sha256(info_bytes).hexdigest()
    manifest["features_sha256"] = hashlib.sha256(feature_bytes).hexdigest()
    verify_metadata(tmp_path, manifest)
    lengths[manifest["selected_shards"][0]["index"]] = "0"
    (data_root / "dataset_info.json").write_text(json.dumps(dataset_info), encoding="utf-8")
    manifest["dataset_info_sha256"] = hashlib.sha256(
        (data_root / "dataset_info.json").read_bytes()
    ).hexdigest()
    with pytest.raises(ValueError, match="episode count differs"):
        verify_metadata(tmp_path, manifest)
