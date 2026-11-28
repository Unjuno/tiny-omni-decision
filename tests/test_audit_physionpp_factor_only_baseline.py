from __future__ import annotations

import hashlib
import json

import pytest

from scripts.audit_physionpp_factor_only_baseline import (
    factor_features,
    group_fold,
    run_factor_audit,
)


def _rows() -> list[dict]:
    groups = {}
    for index in range(100):
        group = f"group-{index}"
        groups.setdefault(group_fold(group), group)
        if len(groups) == 5:
            break
    rows = []
    for fold, group in sorted(groups.items()):
        for label, target in ((False, "No"), (True, "Yes")):
            trial = int(label)
            rows.append(
                {
                    "id": f"sample-{fold}-{trial}",
                    "split": "train",
                    "modality": "video",
                    "source_target": label,
                    "target": target,
                    "source_asset_group_id": group,
                    "source_record_id": (
                        f"readout_data_v1/scenario{fold}/scene-target=cube-variant={trial}"
                        f"#{trial:04d}"
                    ),
                }
            )
    return rows


def test_factor_features_extract_only_scenario_and_recorded_factors() -> None:
    row = _rows()[0]
    features = factor_features(row)
    assert f"scenario=scenario{group_fold(row['source_asset_group_id'])}" in features
    assert "target=cube" in features
    assert "variant=0" in features or "variant=1" in features
    assert all("sample-" not in feature for feature in features)


def test_group_fold_is_stable_and_group_preserving() -> None:
    assert group_fold("scene-identity") == group_fold("scene-identity")
    assert 0 <= group_fold("scene-identity") < 5


def test_train_only_factor_audit_writes_grouped_oof_metrics(tmp_path) -> None:
    rows = _rows()
    train_path = tmp_path / "train.jsonl"
    train_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    output = tmp_path / "audit"
    report = run_factor_audit(
        train_path,
        output,
        hashlib.sha256(train_path.read_bytes()).hexdigest(),
    )
    assert report["train_records"] == 10
    assert report["oof_prior_baseline"]["n"] == 10
    assert report["oof_factor_only"]["n"] == 10
    predictions = [
        json.loads(line)
        for line in (output / "oof-predictions.jsonl").read_text().splitlines()
    ]
    assert len({item["source_asset_group_id"] for item in predictions}) == 5
    assert {item["fold"] for item in predictions} == set(range(5))
    for fold in range(5):
        held_out = {item["source_asset_group_id"] for item in predictions if item["fold"] == fold}
        assert len(held_out) == 1


def test_factor_audit_rejects_non_train_rows_and_manifest_hash_mismatch(tmp_path) -> None:
    row = _rows()[0]
    row["split"] = "validation"
    train_path = tmp_path / "validation.jsonl"
    train_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="train split"):
        run_factor_audit(
            train_path,
            tmp_path / "audit",
            hashlib.sha256(train_path.read_bytes()).hexdigest(),
        )

    row["split"] = "train"
    train_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        run_factor_audit(train_path, tmp_path / "other-audit", "0" * 64)

