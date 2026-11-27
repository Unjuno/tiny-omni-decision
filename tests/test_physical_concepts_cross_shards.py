from __future__ import annotations

import json

import pytest

from scripts.audit_physical_concepts_cross_shards import audit_cross_shards


def _report(object_name: str, suffix: str) -> dict[str, object]:
    return {
        "status": "complete",
        "object_name": object_name,
        "generation": f"generation-{suffix}",
        "records": 2,
        "unique_records": 2,
        "source_sha256": f"source-{suffix}",
        "unique_rgb_sequences": 2,
        "unique_mask_sequences": 2,
        "record_sha256": [f"row-{suffix}-a", f"row-{suffix}-b"],
        "rgb_sequence_sha256": [f"rgb-{suffix}-a", f"rgb-{suffix}-b"],
        "mask_sequence_sha256": [f"mask-{suffix}-a", f"mask-{suffix}-b"],
    }


def _write(path, report: dict[str, object]) -> None:
    path.write_text(json.dumps(report), encoding="utf-8")


def test_cross_shard_audit_reports_zero_exact_overlaps(tmp_path) -> None:
    train = tmp_path / "train.json"
    validation = tmp_path / "validation.json"
    _write(train, _report("train-shard", "train"))
    _write(validation, _report("validation-shard", "validation"))

    result = audit_cross_shards(train, validation)

    assert result["combined_records"] == 4
    assert result["cross_shard_exact_overlaps"] == {
        "serialized_records": 0,
        "rgb_sequences": 0,
        "mask_sequences": 0,
    }


def test_cross_shard_audit_rejects_rgb_overlap(tmp_path) -> None:
    train = _report("train-shard", "train")
    validation = _report("validation-shard", "validation")
    validation["rgb_sequence_sha256"] = ["rgb-train-a", "rgb-validation-b"]
    train_path, validation_path = tmp_path / "train.json", tmp_path / "validation.json"
    _write(train_path, train)
    _write(validation_path, validation)

    with pytest.raises(ValueError, match="cross-shard exact content overlap"):
        audit_cross_shards(train_path, validation_path)


def test_cross_shard_audit_rejects_same_source_shard(tmp_path) -> None:
    train_path, validation_path = tmp_path / "train.json", tmp_path / "validation.json"
    _write(train_path, _report("same-shard", "train"))
    _write(validation_path, _report("same-shard", "validation"))

    with pytest.raises(ValueError, match="different source shards"):
        audit_cross_shards(train_path, validation_path)


def test_cross_shard_audit_rejects_duplicate_within_split(tmp_path) -> None:
    train = _report("train-shard", "train")
    train["record_sha256"] = ["same", "same"]
    train_path, validation_path = tmp_path / "train.json", tmp_path / "validation.json"
    _write(train_path, train)
    _write(validation_path, _report("validation-shard", "validation"))

    with pytest.raises(ValueError, match="duplicate identities"):
        audit_cross_shards(train_path, validation_path)
