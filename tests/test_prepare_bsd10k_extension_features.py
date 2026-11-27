from __future__ import annotations

import json

import numpy as np

from scripts.prepare_bsd10k_extension_features import prepare_feature_sets, sha256_file


def _features(path, ids, values, manifest_hash):
    path.mkdir()
    feature_path = path / "features.npy"
    np.save(feature_path, np.asarray(values, dtype=np.float32), allow_pickle=False)
    index_path = path / "sample-index.json"
    index_path.write_text(
        json.dumps(
            {
                "split": "train" if ids != ["2"] else "validation",
                "sound_ids": ids,
                "durations_seconds": [1.0] * len(ids),
                "window_count": 4,
            }
        ),
        encoding="utf-8",
    )
    report = {
        "status": "PASS_FROZEN_FEATURES",
        "partial_smoke": False,
        "split": "train" if ids != ["2"] else "validation",
        "split_manifest_sha256": manifest_hash,
        "feature_sha256": sha256_file(feature_path),
        "sample_index_sha256": sha256_file(index_path),
        "weights_sha256": "a" * 64,
        "source_commit": "b" * 40,
        "model_parameters": 100,
        "model_state_sha256_before": "c" * 64,
        "model_state_sha256_after": "c" * 64,
    }
    (path / "run.json").write_text(json.dumps(report), encoding="utf-8")
    return np.asarray(values, dtype=np.float32)


def test_features_combine_train_only_and_preserve_validation(tmp_path):
    base_manifest = {
        "schema": "tiny-omni-decision/bsd10k-group-split-v1",
        "source_record_id": "19868804",
        "metadata_zip_sha256": "d" * 64,
        "examples": [
            {"sound_id": "1", "split": "train", "license": "CC-BY-4.0"},
            {"sound_id": "2", "split": "validation", "license": "CC-BY-4.0"},
        ],
    }
    base_path = tmp_path / "base.json"
    base_path.write_text(json.dumps(base_manifest), encoding="utf-8")
    base_hash = sha256_file(base_path)
    combined_manifest = {
        **base_manifest,
        "examples": base_manifest["examples"]
        + [{"sound_id": "3", "split": "train", "license": "CC-BY-3.0"}],
        "training_extension": {"base_split_manifest_sha256": base_hash},
    }
    combined_path = tmp_path / "combined.json"
    combined_path.write_text(json.dumps(combined_manifest), encoding="utf-8")
    additions_manifest = {**base_manifest, "examples": [combined_manifest["examples"][-1]]}
    additions_path = tmp_path / "additions.json"
    additions_path.write_text(json.dumps(additions_manifest), encoding="utf-8")

    old_train = _features(tmp_path / "old-train", ["1"], np.ones((1, 384)), base_hash)
    added = _features(
        tmp_path / "additions", ["3"], np.full((1, 384), 2), sha256_file(additions_path)
    )
    old_validation = _features(tmp_path / "old-validation", ["2"], np.full((1, 384), 3), base_hash)
    report = prepare_feature_sets(
        combined_path,
        additions_path,
        tmp_path / "old-train",
        tmp_path / "additions",
        tmp_path / "old-validation",
        tmp_path / "new-train",
        tmp_path / "new-validation",
    )

    assert report["train_examples"] == 2
    assert report["added_train_examples"] == 1
    assert np.array_equal(
        np.load(tmp_path / "new-train" / "features.npy"), np.stack([old_train[0], added[0]])
    )
    assert np.array_equal(np.load(tmp_path / "new-validation" / "features.npy"), old_validation)
