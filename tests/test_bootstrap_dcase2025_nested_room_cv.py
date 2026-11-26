from __future__ import annotations

import json

import pytest

pytest.importorskip("torch")

from scripts.bootstrap_dcase2025_nested_room_cv import summarize


def test_nested_oof_bootstrap_verifies_hashes_and_saves_paired_groups(tmp_path):
    cv_dir = tmp_path
    predictions = [
        {
            "sample_id": f"{room}-{index}",
            "parent_recording": f"{room}-parent-{index // 2}",
            "room_id": room,
            "target_onscreen": index % 2,
            "probabilities": {
                "audio": [0.8, 0.2] if index % 2 == 0 else [0.3, 0.7],
                "video": [0.7, 0.3] if index % 2 == 0 else [0.4, 0.6],
                "joint": [0.9, 0.1] if index % 2 == 0 else [0.2, 0.8],
            },
        }
        for room in ("r1", "r2", "r3")
        for index in range(4)
    ]
    prediction_path = cv_dir / "oof-predictions.jsonl"
    prediction_path.write_text(
        "".join(json.dumps(row) + "\n" for row in predictions), encoding="utf-8"
    )
    config = {"outer_protocol": "leave-one-room-out across 9 rooms; every sample tested once"}
    (cv_dir / "effective-config.json").write_text(json.dumps(config), encoding="utf-8")
    from scripts.evaluate_dcase2025_av_readout import sha256_file

    result = {
        "status": "PASS",
        "selection_used_outer_test": False,
        "development_test_accessed": False,
        "sealed_audit_accessed": False,
        "rows": len(predictions),
        "predictions_sha256": sha256_file(prediction_path),
    }
    (cv_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")

    report = summarize(cv_dir, replicates=64, seed=17)

    assert report["rows"] == 12
    assert report["room_groups"] == 3
    assert report["paired_group_differences"]["room"]["replicates"] == 64
    assert (cv_dir / "paired-bootstrap.json").is_file()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        summarize(cv_dir, replicates=64, seed=17)


def test_nested_oof_bootstrap_rejects_prediction_hash_mismatch(tmp_path):
    (tmp_path / "effective-config.json").write_text(
        json.dumps(
            {"outer_protocol": "leave-one-room-out across 9 rooms; every sample tested once"}
        ),
        encoding="utf-8",
    )
    (tmp_path / "result.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "selection_used_outer_test": False,
                "development_test_accessed": False,
                "sealed_audit_accessed": False,
                "rows": 1,
                "predictions_sha256": "incorrect",
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "oof-predictions.jsonl").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        summarize(tmp_path, replicates=8, seed=17)
