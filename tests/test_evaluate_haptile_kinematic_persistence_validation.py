import json
from pathlib import Path

import numpy as np
import pytest

import scripts.pretrained_reuse.evaluate_haptile_kinematic_persistence_validation as validation
from scripts.pretrained_reuse.evaluate_haptile_kinematic_persistence_validation import (
    load_validation_cache,
)


def test_validation_loader_rejects_train_cache_before_reading_metadata(tmp_path: Path):
    candidate = tmp_path / "train-joined-v1" / "decision-features.npz"
    candidate.parent.mkdir()
    with pytest.raises(ValueError, match="only the frozen validation"):
        load_validation_cache(candidate)


def test_validation_loader_rejects_unpinned_validation_path(tmp_path: Path):
    candidate = tmp_path / "other-validation-cache" / "decision-features.npz"
    candidate.parent.mkdir()
    with pytest.raises(ValueError, match="only the frozen validation"):
        load_validation_cache(candidate)


def test_validation_loader_accepts_only_hash_pinned_validation_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cache = tmp_path / "validation-joined-v1" / "decision-features.npz"
    cache.parent.mkdir()
    arrays = {
        "state_values": np.zeros((2, 8, 25), dtype=np.float32),
        "state_timestamps_ms": np.tile(np.arange(8, dtype=np.float32), (2, 1)),
        "targets": np.array([0, 1], dtype=np.int64),
        "task_groups": np.array(["task-a", "task-b"]),
        "sample_paths": np.array(["a", "b"]),
    }
    np.savez(cache, **arrays)
    digest = validation.sha256_file(cache)
    metadata = {
        "split": "validation",
        "sealed_audit_opened": False,
        "examples": 2,
        "task_groups": 2,
        "feature_file_sha256": digest,
    }
    (cache.parent / "result.json").write_text(json.dumps(metadata), encoding="utf-8")
    monkeypatch.setattr(validation, "VALIDATION_CACHE_SHA256", digest)
    monkeypatch.setattr(validation, "VALIDATION_EXAMPLES", 2)
    monkeypatch.setattr(validation, "VALIDATION_TASK_GROUPS", 2)

    loaded, loaded_digest = load_validation_cache(cache)

    assert loaded_digest == digest
    assert loaded["state_values"].shape == (2, 8, 25)


def test_validation_loader_rejects_sealed_audit_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cache = tmp_path / "validation-joined-v1" / "decision-features.npz"
    cache.parent.mkdir()
    arrays = {
        "state_values": np.zeros((1, 8, 25), dtype=np.float32),
        "state_timestamps_ms": np.tile(np.arange(8, dtype=np.float32), (1, 1)),
        "targets": np.array([0], dtype=np.int64),
        "task_groups": np.array(["task-a"]),
        "sample_paths": np.array(["a"]),
    }
    np.savez(cache, **arrays)
    digest = validation.sha256_file(cache)
    metadata = {
        "split": "validation",
        "sealed_audit_opened": True,
        "examples": 1,
        "task_groups": 1,
        "feature_file_sha256": digest,
    }
    (cache.parent / "result.json").write_text(json.dumps(metadata), encoding="utf-8")
    monkeypatch.setattr(validation, "VALIDATION_CACHE_SHA256", digest)
    monkeypatch.setattr(validation, "VALIDATION_EXAMPLES", 1)
    monkeypatch.setattr(validation, "VALIDATION_TASK_GROUPS", 1)

    with pytest.raises(ValueError, match="frozen isolated validation"):
        load_validation_cache(cache)
