from __future__ import annotations

import hashlib
import json

import numpy as np

from scripts.pretrained_reuse.extract_haptile_videoprism_features import (
    _load_development_inputs,
    finalize_feature_memmap,
)


def test_feature_memmap_closes_before_final_rename(tmp_path):
    partial = tmp_path / "features.partial.npy"
    final = tmp_path / "features.npy"
    features = np.lib.format.open_memmap(
        partial, mode="w+", dtype=np.float32, shape=(2, 3)
    )
    features[:] = [[1, 2, 3], [4, 5, 6]]

    finalize_feature_memmap(features, partial, final)

    assert not partial.exists()
    np.testing.assert_array_equal(np.load(final), [[1, 2, 3], [4, 5, 6]])


def test_development_feature_loader_requires_matching_validation_provenance(tmp_path):
    manifest_path = tmp_path / "split.json"
    receipt_path = tmp_path / "validation-receipt.json"
    action_path = tmp_path / "validation-action.json"
    manifest = {
        "development_splits": {
            "train": ["train_task"], "validation": ["valid_task"],
            "calibration": ["calib_task"],
        },
        "development_episode_counts": {"validation": 1, "calibration": 1},
        "sealed_audit": {"task_group_count": 1},
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    receipt = {
        "format": "haptile_development_split_corpus_v1",
        "source_revision": "54887384f1fe9e93b799e104194a582f9d8840b8",
        "conversion_revision": "c8593da98d3facb7a9293842514f0ee4ad3f2318",
        "split_manifest_file_sha256": manifest_hash,
        "accessed_splits": ["validation"], "sealed_audit_opened": False,
        "task_count": 1, "episode_count": 1,
        "files": [{
            "path": "data/valid_task/episode_001.fo.mcap",
            "sha256": "media-hash", "size": 12,
        }],
    }
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    receipt_hash = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    action = {
        "format": "haptile_validation_action_preflight_v1",
        "input_receipt_sha256": receipt_hash,
        "split_manifest_sha256": manifest_hash,
        "split_accessed": "validation", "sealed_audit_opened": False,
        "samples": [{
            "path": "data/valid_task/episode_001.fo.mcap",
            "sha256": "media-hash", "query_time_log_seconds": 1.5,
            "target": "+x",
        }],
    }
    action_path.write_text(json.dumps(action), encoding="utf-8")

    _, rows = _load_development_inputs(
        receipt_path, action_path, manifest_path, "validation"
    )
    assert rows == [{
        "path": "data/valid_task/episode_001.fo.mcap",
        "task": "valid_task", "sha256": "media-hash",
        "query_time_log_seconds": 1.5, "target": "+x",
    }]
