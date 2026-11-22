from __future__ import annotations

import pytest

from scripts.pretrained_reuse.analyze_haptile_development_preflight import (
    validate_development_receipt,
)


@pytest.fixture
def manifest_and_receipt() -> tuple[dict[str, object], dict[str, object]]:
    manifest = {
        "development_splits": {
            "train": ["train_task"],
            "validation": ["valid_a", "valid_b"],
            "calibration": ["calib_a"],
        },
        "development_episode_counts": {"validation": 2, "calibration": 1},
        "sealed_audit": {"task_group_count": 1},
        "sealed_audit_accessed": False,
    }
    files = [
        {
            "path": f"data/valid_{name}/episode_001.fo.mcap",
            "task": f"valid_{name}",
            "size": 1,
            "sha256": f"sha-{name}",
        }
        for name in ("a", "b")
    ]
    receipt = {
        "format": "haptile_development_split_corpus_v1",
        "source_revision": "54887384f1fe9e93b799e104194a582f9d8840b8",
        "conversion_revision": "c8593da98d3facb7a9293842514f0ee4ad3f2318",
        "split_manifest_file_sha256": "manifest-hash",
        "accessed_splits": ["validation"],
        "sealed_audit_opened": False,
        "task_count": 2,
        "episode_count": 2,
        "total_bytes": 2,
        "task_tree_receipts": {
            "valid_a": {"mcap_count": 1}, "valid_b": {"mcap_count": 1},
        },
        "files": files,
    }
    return manifest, receipt


def test_accepts_exact_validation_receipt(manifest_and_receipt) -> None:
    manifest, receipt = manifest_and_receipt
    actual = validate_development_receipt(receipt, manifest, "manifest-hash", "validation")
    assert actual == receipt["files"]


@pytest.mark.parametrize("split", ["train", "sealed_audit"])
def test_rejects_non_development_split(manifest_and_receipt, split: str) -> None:
    manifest, receipt = manifest_and_receipt
    with pytest.raises(ValueError, match="only validation or calibration"):
        validate_development_receipt(receipt, manifest, "manifest-hash", split)


def test_rejects_receipt_that_claims_multiple_splits(manifest_and_receipt) -> None:
    manifest, receipt = manifest_and_receipt
    receipt["accessed_splits"] = ["validation", "sealed_audit"]
    with pytest.raises(ValueError, match="different or multiple"):
        validate_development_receipt(receipt, manifest, "manifest-hash", "validation")


def test_rejects_sealed_audit_opened(manifest_and_receipt) -> None:
    manifest, receipt = manifest_and_receipt
    receipt["sealed_audit_opened"] = True
    with pytest.raises(ValueError, match="sealed audit isolation"):
        validate_development_receipt(receipt, manifest, "manifest-hash", "validation")


def test_rejects_manifest_that_does_not_prove_audit_isolation(manifest_and_receipt) -> None:
    manifest, receipt = manifest_and_receipt
    manifest["sealed_audit_accessed"] = True
    with pytest.raises(ValueError, match="manifest does not prove sealed audit"):
        validate_development_receipt(receipt, manifest, "manifest-hash", "validation")


def test_rejects_task_metadata_count_mismatch(manifest_and_receipt) -> None:
    manifest, receipt = manifest_and_receipt
    receipt["task_tree_receipts"]["valid_a"]["mcap_count"] = 0
    with pytest.raises(ValueError, match="media count differs"):
        validate_development_receipt(receipt, manifest, "manifest-hash", "validation")


def test_rejects_train_group_in_validation_receipt(manifest_and_receipt) -> None:
    manifest, receipt = manifest_and_receipt
    receipt["files"][0]["task"] = "train_task"
    receipt["files"][0]["path"] = "data/train_task/episode_001.fo.mcap"
    with pytest.raises(ValueError, match="nonrequested task"):
        validate_development_receipt(receipt, manifest, "manifest-hash", "validation")


def test_rejects_unsafe_and_duplicate_paths(manifest_and_receipt) -> None:
    manifest, receipt = manifest_and_receipt
    receipt["files"][0]["path"] = "data/valid_a/../episode.mcap"
    with pytest.raises(ValueError, match="unsafe media path"):
        validate_development_receipt(receipt, manifest, "manifest-hash", "validation")
    receipt["files"][0]["path"] = receipt["files"][1]["path"]
    receipt["files"][0]["task"] = receipt["files"][1]["task"]
    with pytest.raises(ValueError, match="duplicate media paths"):
        validate_development_receipt(receipt, manifest, "manifest-hash", "validation")
