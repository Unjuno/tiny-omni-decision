from __future__ import annotations

import hashlib

import pytest

from scripts.pretrained_reuse.extract_haptile_development_state_text import (
    validate_development_action,
)


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def action_inputs(tmp_path):
    receipt_path = tmp_path / "receipt.json"
    manifest_path = tmp_path / "manifest.json"
    receipt_path.write_text("{}", encoding="utf-8")
    manifest_path.write_text("{}", encoding="utf-8")
    files = [{"path": "data/valid_a/episode_001.fo.mcap", "sha256": "asset-hash"}]
    receipt = {"accessed_splits": ["validation"]}
    action = {
        "format": "haptile_validation_action_preflight_v1",
        "input_receipt_sha256": _sha(receipt_path),
        "split_manifest_sha256": _sha(manifest_path),
        "split_accessed": "validation",
        "sealed_audit_opened": False,
        "samples": [{
            "path": files[0]["path"], "sha256": "asset-hash",
            "query_time_log_seconds": 2.5, "target": "+x",
        }],
    }
    return receipt_path, manifest_path, receipt, files, action


def test_accepts_matched_development_targets(action_inputs) -> None:
    receipt_path, manifest_path, receipt, files, action = action_inputs
    result = validate_development_action(
        action=action, receipt=receipt, receipt_path=receipt_path,
        manifest_path=manifest_path, split_name="validation", file_rows=files,
    )
    assert result[files[0]["path"]]["target"] == "+x"


def test_rejects_target_artifact_from_other_split(action_inputs) -> None:
    receipt_path, manifest_path, receipt, files, action = action_inputs
    action["format"] = "haptile_calibration_action_preflight_v1"
    with pytest.raises(ValueError, match="provenance"):
        validate_development_action(
            action=action, receipt=receipt, receipt_path=receipt_path,
            manifest_path=manifest_path, split_name="validation", file_rows=files,
        )


def test_rejects_target_artifact_without_isolation_proof(action_inputs) -> None:
    receipt_path, manifest_path, receipt, files, action = action_inputs
    action["sealed_audit_opened"] = True
    with pytest.raises(ValueError, match="provenance"):
        validate_development_action(
            action=action, receipt=receipt, receipt_path=receipt_path,
            manifest_path=manifest_path, split_name="validation", file_rows=files,
        )


def test_rejects_duplicate_or_missing_target_rows(action_inputs) -> None:
    receipt_path, manifest_path, receipt, files, action = action_inputs
    action["samples"].append(dict(action["samples"][0]))
    with pytest.raises(ValueError, match="one-to-one"):
        validate_development_action(
            action=action, receipt=receipt, receipt_path=receipt_path,
            manifest_path=manifest_path, split_name="validation", file_rows=files,
        )
