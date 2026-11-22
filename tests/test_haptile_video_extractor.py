from __future__ import annotations

import hashlib
import json

import numpy as np

from scripts.pretrained_reuse.extract_haptile_videoprism_features import (
    VIDEO_TOPICS,
    _apply_video_eligibility,
    _load_development_inputs,
    finalize_feature_memmap,
    sha256_file,
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
        "sealed_audit_accessed": False,
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
        "total_bytes": 12,
        "task_tree_receipts": {"valid_task": {"mcap_count": 1}},
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


def test_video_eligibility_filters_only_exact_ineligible_media(tmp_path):
    receipt_path = tmp_path / "receipt.json"
    action_path = tmp_path / "action.json"
    split_path = tmp_path / "split.json"
    receipt_path.write_text("receipt", encoding="utf-8")
    action_path.write_text("action", encoding="utf-8")
    split_path.write_text("split", encoding="utf-8")
    samples = [
        {"path": "data/task/ok.mcap", "sha256": "a" * 64,
         "task": "task", "query_time_log_seconds": 12.0, "target": "+x"},
        {"path": "data/task/short.mcap", "sha256": "b" * 64,
         "task": "task", "query_time_log_seconds": 12.0, "target": "-x"},
    ]
    times = np.linspace(11.0, 12.0, 8).tolist()
    eligibility = {
        "format": "haptile_train_video_window_eligibility_v2",
        "split": "train",
        "input_receipt_sha256": sha256_file(receipt_path),
        "action_preflight_sha256": sha256_file(action_path),
        "split_manifest_sha256": sha256_file(split_path),
        "selection_uses_labels": False,
        "sealed_audit_opened": False,
        "criteria": {
            "context_seconds": 1.0,
            "frames_per_stream": 8,
            "max_frame_time_error_seconds": 0.1,
            "causal_cutoff_inclusive": True,
            "streams": list(VIDEO_TOPICS),
            "excluded_only_for": "insufficient distinct causal frames in fixed window",
        },
        "eligible_examples": 1,
        "excluded_examples": 1,
        "corpus_examples": 2,
        "eligible_samples": [{
            "path": samples[0]["path"], "sha256": samples[0]["sha256"],
            "task": "task", "query_time_log_seconds": 12.0,
            "selected_frame_times_log_seconds": {
                topic: times for topic in VIDEO_TOPICS
            },
        }],
        "excluded_samples": [{
            "path": samples[1]["path"], "sha256": samples[1]["sha256"],
        }],
    }
    eligibility_path = tmp_path / "eligibility.json"
    eligibility_path.write_text(json.dumps(eligibility), encoding="utf-8")

    selected, excluded_count = _apply_video_eligibility(
        samples,
        eligibility_path=eligibility_path,
        split_name="train",
        receipt_path=receipt_path,
        action_path=action_path,
        split_manifest_path=split_path,
    )

    assert excluded_count == 1
    assert [row["path"] for row in selected] == ["data/task/ok.mcap"]
    assert "target" in selected[0]
    assert "target" not in eligibility["eligible_samples"][0]


def test_video_eligibility_rejects_wrong_input_hash(tmp_path):
    receipt_path = tmp_path / "receipt.json"
    action_path = tmp_path / "action.json"
    split_path = tmp_path / "split.json"
    for path in (receipt_path, action_path, split_path):
        path.write_text("fixed", encoding="utf-8")
    eligibility_path = tmp_path / "eligibility.json"
    eligibility_path.write_text(json.dumps({
        "format": "haptile_train_video_window_eligibility_v2",
        "split": "train",
        "input_receipt_sha256": sha256_file(receipt_path),
        "action_preflight_sha256": "0" * 64,
        "split_manifest_sha256": sha256_file(split_path),
        "selection_uses_labels": False,
        "sealed_audit_opened": False,
    }), encoding="utf-8")

    try:
        _apply_video_eligibility(
            [],
            eligibility_path=eligibility_path,
            split_name="train",
            receipt_path=receipt_path,
            action_path=action_path,
            split_manifest_path=split_path,
        )
    except ValueError as error:
        assert "action_preflight_sha256" in str(error)
    else:
        raise AssertionError("eligibility from a different format unexpectedly passed")
