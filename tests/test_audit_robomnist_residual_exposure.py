from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.audit_robomnist_residual_exposure import audit_residual_exposure


def _write_json(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, sort_keys=True).encode("utf-8")
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _group(index: int) -> str:
    return f"Ar1_Vhigh_Ac0_Rp{index:02d}_UNC062"


def test_metadata_audit_counts_residual_and_ignores_sealed_paths(tmp_path: Path) -> None:
    inventory = tmp_path / "inventory"
    all_ids = [_group(index) for index in range(1, 2136)]
    directory_hashes = {}
    for index in range(60):
        path = inventory / f"directory-{index:02d}.json"
        rows = [{"group_id": group_id} for group_id in all_ids[index::60]]
        directory_hashes[path.name] = _write_json(path, {"groups": rows})
    config_hash = _write_json(inventory / "config.json", {"source": "fixture"})
    source_manifest = tmp_path / "source-manifest.json"
    _write_json(
        source_manifest,
        {
            "source": {
                "config_sha256": config_hash,
                "directory_metadata_sha256": directory_hashes,
            }
        },
    )

    artifact_root = tmp_path / "artifacts"
    prior = artifact_root / "robomnist-prior-run"
    train_hash = _write_json(
        prior / "train.json",
        [{"source_group_id": _group(1)}, {"source_group_id": _group(2)}],
    )
    validation_hash = _write_json(
        prior / "validation.json", [{"source_group_id": _group(3)}, {"source_group_id": _group(4)}]
    )
    _write_json(
        prior / "summary.json",
        {
            "payload_opened": False,
            "training": False,
            "model_inference": False,
            "sealed_audit_evaluated": False,
            "train_sha256": train_hash,
            "validation_sha256": validation_hash,
        },
    )
    _write_json(
        artifact_root / "robomnist-old-split" / "manifest.json",
        {"groups": {group_id: {} for group_id in all_ids[4:43]}},
    )
    _write_json(
        artifact_root / "robomnist-sealed-audit" / "manifest.json",
        {"groups": {group_id: {} for group_id in all_ids[43:]}},
    )
    _write_json(
        artifact_root / "robomnist-residual-exposure-audit-previous" / "result.json",
        {"groups": {group_id: {} for group_id in all_ids[43:]}},
    )

    result = audit_residual_exposure(
        inventory_dir=inventory,
        source_manifest=source_manifest,
        prior_run=prior,
        artifact_root=artifact_root,
    )

    assert result["source_group_count"] == 2135
    assert result["prior_nonsealed_group_count"] == 4
    assert result["residual_group_count"] == 2131
    assert result["residual_groups_appearing_in_nonsealed_records"] == 39
    assert result["residual_groups_not_found_in_scanned_records"] == 2092
    assert result["maximum_pair_units_from_unmatched_groups"] == 1046
    assert result["maximum_rows_at_four_per_pair"] == 4184
    assert all("sealed" not in row["path"].lower() for row in result["metadata_exposure_records"])


def test_metadata_audit_fails_closed_on_prior_manifest_hash_mismatch(tmp_path: Path) -> None:
    inventory = tmp_path / "inventory"
    directory_hashes = {}
    all_ids = [_group(index) for index in range(1, 2136)]
    directory_hashes = {}
    for index in range(60):
        path = inventory / f"directory-{index:02d}.json"
        rows = [{"group_id": group_id} for group_id in all_ids[index::60]]
        directory_hashes[path.name] = _write_json(path, {"groups": rows})
    config_hash = _write_json(inventory / "config.json", {"source": "fixture"})
    source_manifest = tmp_path / "source-manifest.json"
    _write_json(
        source_manifest,
        {
            "source": {
                "config_sha256": config_hash,
                "directory_metadata_sha256": directory_hashes,
            }
        },
    )
    prior = tmp_path / "robomnist-prior-run"
    _write_json(prior / "train.json", [{"source_group_id": _group(1)}])
    validation_hash = _write_json(prior / "validation.json", [{"source_group_id": _group(2)}])
    _write_json(
        prior / "summary.json",
        {
            "payload_opened": False,
            "training": False,
            "model_inference": False,
            "sealed_audit_evaluated": False,
            "train_sha256": "incorrect",
            "validation_sha256": validation_hash,
        },
    )

    with pytest.raises(ValueError, match="train manifest hash mismatch"):
        audit_residual_exposure(
            inventory_dir=inventory,
            source_manifest=source_manifest,
            prior_run=prior,
            artifact_root=tmp_path / "artifacts",
        )
