import json

import pytest

pytest.importorskip("torch")

from scripts.bootstrap_dcase2025_readout_architectures import paired_bootstrap
from scripts.run_dcase2025_nested_room_cv import room_folds


def _write_cv(path, rows, *, reference=False):
    path.mkdir()
    predictions = path / "oof-predictions.jsonl"
    predictions.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    config = {
        "outer_protocol": "leave-one-room-out across 9 rooms; every sample tested once",
        "feature_identity": {"query_ids_sha256": "same-id-order"},
        "seed": 17,
    }
    folds = room_folds(rows, seed=17)
    layout = [
        {
            "test_room": fold["test_room"],
            "validation_room": fold["validation_room"],
            "train_rooms": fold["train_rooms"],
            "train_rows": len(fold["train_indices"]),
            "validation_rows": len(fold["validation_indices"]),
            "test_rows": len(fold["test_indices"]),
        }
        for fold in folds
    ]
    if reference:
        config["split_seed"] = 17
        config["folds"] = layout
    (path / "effective-config.json").write_text(json.dumps(config), encoding="utf-8")
    import hashlib

    (path / "result.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "selection_used_outer_test": False,
                "development_test_accessed": False,
                "sealed_audit_accessed": False,
                "rows": len(rows),
                "predictions_sha256": hashlib.sha256(predictions.read_bytes()).hexdigest(),
                "fold_results": [
                    {
                        "test_room": fold["test_room"],
                        "validation_room": fold["validation_room"],
                    }
                    for fold in folds
                ],
            }
        ),
        encoding="utf-8",
    )


def test_paired_bootstrap_requires_exact_sample_group_target_alignment(tmp_path):
    rows = [
        {
            "sample_id": f"{room}-{index}",
            "room_id": room,
            "parent_recording": f"{room}-parent-{index // 2}",
            "target_onscreen": index % 2,
            "probabilities": {
                mode: ([0.8, 0.2] if index % 2 == 0 else [0.2, 0.8])
                for mode in ("audio", "video", "joint")
            },
        }
        for room in ("r1", "r2", "r3")
        for index in range(4)
    ]
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _write_cv(reference, rows, reference=True)
    _write_cv(candidate, rows)

    result = paired_bootstrap(reference, candidate, replicates=64, seed=17)

    assert result["sample_count"] == 12
    assert result["fold_layout_verified_against_reference"] is True
    assert result["paired_group_differences"]["room"]["groups"] == 3
    assert (
        result["paired_group_differences"]["room"]["modes"]["joint"]["candidate_minus_reference"][
            "accuracy"
        ]
        == 0
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        paired_bootstrap(reference, candidate, replicates=64, seed=17)


def test_paired_bootstrap_rejects_group_mismatch(tmp_path):
    rows = [
        {
            "sample_id": f"q{index}",
            "room_id": f"r{index}",
            "parent_recording": f"p{index}",
            "target_onscreen": index % 2,
            "probabilities": {mode: [0.7, 0.3] for mode in ("audio", "video", "joint")},
        }
        for index in range(3)
    ]
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _write_cv(reference, rows, reference=True)
    _write_cv(candidate, [{**rows[0], "room_id": "changed"}, *rows[1:]])

    with pytest.raises(ValueError, match="not exactly paired"):
        paired_bootstrap(reference, candidate, replicates=8, seed=17)
