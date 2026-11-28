from __future__ import annotations

import hashlib
import json

import pytest

from scripts.extract_physionpp_validation_videoprism_features import (
    select_validation_rows,
    validate_split_report,
)


def _row(index: int, *, split: str = "validation", sha: str | None = None):
    return {
        "id": f"val-{index}",
        "split": split,
        "source": "physionpp/readout",
        "media": [{"kind": "video", "sha256": sha or f"{index + 100:064x}"}],
    }


def test_validation_selector_keeps_manifest_order_and_disjoint_rows() -> None:
    rows = [_row(0), _row(1)]
    assert select_validation_rows(rows, {"train-1"}, {f"{0:064x}"}) == rows


def test_validation_selector_rejects_train_rows_and_overlap() -> None:
    with pytest.raises(ValueError, match="development validation"):
        select_validation_rows([_row(0, split="train")], set(), set())
    row = _row(0)
    with pytest.raises(ValueError, match="overlap"):
        select_validation_rows([row], {row["id"]}, set())
    with pytest.raises(ValueError, match="overlap"):
        select_validation_rows([row], set(), {row["media"][0]["sha256"]})


def test_validation_selector_rejects_duplicate_media_and_missing_asset() -> None:
    first = _row(0)
    second = _row(1, sha=first["media"][0]["sha256"])
    with pytest.raises(ValueError, match="unique"):
        select_validation_rows([first, second], set(), set())
    with pytest.raises(ValueError, match="exactly one"):
        select_validation_rows([{**first, "media": []}], set(), set())


def test_split_report_must_match_both_frozen_manifests(tmp_path) -> None:
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    train_path.write_text("train", encoding="utf-8")
    validation_path.write_text("validation", encoding="utf-8")
    train_rows = [_row(1, split="train")]
    validation_rows = [_row(0)]
    report = {
        "status": "disjoint",
        "source_revision": "0c53b983ca138962ba636c0936e735490ad4c326",
        "train_manifest_sha256": hashlib.sha256(b"train").hexdigest(),
        "validation_manifest_sha256": hashlib.sha256(b"validation").hexdigest(),
        "train_records": 1,
        "validation_records": 1,
        "model_loaded": False,
        "media_decoded": False,
        "model_metrics_computed": False,
        "overlaps": {
            key: {"shared": 0}
            for key in (
                "record_id",
                "source_record_id",
                "source_asset_group_id",
                "media_sha256",
                "media_path",
            )
        },
    }
    report["overlaps"]["normalized_content_and_source_id"] = {
        "status": "disjoint",
        "shared_content_fingerprints": 0,
        "shared_source_ids": 0,
    }
    report_path = tmp_path / "split-audit.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    assert (
        validate_split_report(report_path, train_path, validation_path, train_rows, validation_rows)
        == report
    )
    report["overlaps"]["media_sha256"]["shared"] = 1
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="media_sha256 overlap"):
        validate_split_report(report_path, train_path, validation_path, train_rows, validation_rows)
