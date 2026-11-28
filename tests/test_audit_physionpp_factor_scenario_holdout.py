import hashlib
import json

import pytest

from scripts.audit_physionpp_factor_scenario_holdout import run_scenario_holdout


def _rows(scenarios=("alpha", "beta", "gamma"), *, split="train"):
    rows = []
    for scenario in scenarios:
        for label, zdloc in (("No", 0), ("Yes", 1)):
            record = (
                f"readout_data_v1/{scenario}/"
                f"{scenario}-zdloc={zdloc}-target=cube#000{len(rows)}"
            )
            rows.append(
                {
                    "id": f"physionpp/{record}",
                    "modality": "video",
                    "source_record_id": record,
                    "source_asset_group_id": f"{scenario}:seed={len(rows)}",
                    "target": label,
                    "source_target": label == "Yes",
                    "split": split,
                }
            )
    return rows


def _write_manifest(path, rows):
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_scenario_holdout_keeps_each_scenario_entirely_out_of_fit(tmp_path):
    manifest = tmp_path / "train.jsonl"
    output = tmp_path / "result"
    digest = _write_manifest(manifest, _rows())

    report = run_scenario_holdout(
        manifest, output, expected_sha256=digest
    )

    assert report["scenarios"] == 3
    assert report["train_records"] == 6
    assert len(report["folds"]) == 3
    assert all(fold["test_examples"] == 2 for fold in report["folds"])
    assert all(
        fold["train_source_asset_groups"] == 4
        and fold["test_source_asset_groups"] == 2
        for fold in report["folds"]
    )
    assert len((output / "oof-predictions.jsonl").read_text().splitlines()) == 6


def test_scenario_holdout_rejects_non_train_rows(tmp_path):
    manifest = tmp_path / "train.jsonl"
    digest = _write_manifest(manifest, _rows(split="validation"))

    with pytest.raises(ValueError, match="only train rows"):
        run_scenario_holdout(manifest, tmp_path / "result", expected_sha256=digest)


def test_scenario_holdout_rejects_inconsistent_scenario_group(tmp_path):
    rows = _rows()
    rows[0]["source_asset_group_id"] = rows[2]["source_asset_group_id"]
    manifest = tmp_path / "train.jsonl"
    digest = _write_manifest(manifest, rows)

    with pytest.raises(ValueError, match="group crosses scenario split"):
        run_scenario_holdout(manifest, tmp_path / "result", expected_sha256=digest)
