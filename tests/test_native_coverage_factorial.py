import json
from pathlib import Path

import numpy as np
import pytest

from scripts.analyze_native_coverage_factorial import (
    FIT_DIRS,
    factorial_effects,
    inspect_conditions,
    paired_cluster_bootstrap,
    run,
    sha256,
)


def panel_rows():
    rows = []
    for modality in ("audio", "image", "text", "video"):
        for group in ("g0", "g1"):
            rows.append(
                {
                    "id": f"{modality}-{group}",
                    "modality": modality,
                    "source": f"{modality}-source",
                    "task": "decision",
                    "group": group,
                    "observation": f"{modality}-{group}",
                    "options": ["a", "b"],
                    "option_ids": ["a", "b"],
                    "target_index": 0,
                }
            )
    return rows


def scores(rows, correct_margin):
    values = np.zeros((len(rows), 2), dtype=np.float64)
    values[:, 0] = correct_margin
    return values


def test_factorial_effects_separate_main_effects_and_interaction():
    rows = panel_rows()
    # Only the full-extent + fresh-coverage cell gets a stronger correct logit.
    arrays = {
        "center_repeat": scores(rows, -3.0),
        "center_fresh": scores(rows, -3.0),
        "full_repeat": scores(rows, -3.0),
        "full_fresh": scores(rows, 3.0),
    }

    result = factorial_effects(rows, arrays)

    assert result["coverage_effect_center"]["macro"]["accuracy"] == 0.0
    assert result["coverage_effect_full"]["macro"]["accuracy"] == 1.0
    assert result["extent_effect_repeat"]["macro"]["accuracy"] == 0.0
    assert result["extent_effect_fresh"]["macro"]["accuracy"] == 1.0
    assert result["interaction"]["macro"]["accuracy"] == 1.0
    assert set(result["interaction"]["modalities"]) == {
        "audio",
        "image",
        "text",
        "video",
    }


def test_factorial_effects_reject_missing_cell_or_misaligned_rows():
    rows = panel_rows()
    arrays = {name: scores(rows, 1.0) for name in ("center_repeat", "center_fresh", "full_repeat")}
    with pytest.raises(ValueError, match="four factorial cells"):
        factorial_effects(rows, arrays)

    arrays["full_fresh"] = scores(rows[:-1], 1.0)
    with pytest.raises(ValueError, match="same validation rows"):
        factorial_effects(rows, arrays)


def test_cluster_bootstrap_is_paired_reproducible_and_reports_interaction_interval():
    rows = panel_rows()
    arrays = {
        "center_repeat": scores(rows, -3.0),
        "center_fresh": scores(rows, -3.0),
        "full_repeat": scores(rows, -3.0),
        "full_fresh": scores(rows, 3.0),
    }

    first = paired_cluster_bootstrap(rows, arrays, seed=17, resamples=200)
    second = paired_cluster_bootstrap(rows, arrays, seed=17, resamples=200)

    assert first == second
    assert first["interaction"]["macro"]["accuracy"] == {
        "estimate": 1.0,
        "low": 1.0,
        "high": 1.0,
    }
    assert first["protocol"]["unit"] == "paired rows clustered by group within modality"
    assert "ece" not in first["interaction"]["macro"]


def test_cluster_bootstrap_requires_independent_groups_per_modality():
    rows = panel_rows()
    for row in rows:
        row["group"] = row["modality"]
    arrays = {
        name: scores(rows, 1.0)
        for name in ("center_repeat", "center_fresh", "full_repeat", "full_fresh")
    }
    with pytest.raises(ValueError, match="two independent groups"):
        paired_cluster_bootstrap(rows, arrays, resamples=10)


def write_artifact_root(root: Path, missing=()):
    rows = panel_rows()
    data = root / "native-common-corpus-02"
    data.mkdir(parents=True)
    (data / "validation.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf8"
    )
    ids = [
        {"id": r["id"], "option_ids": r["option_ids"], "target_index": r["target_index"]}
        for r in rows
    ]
    expected = {}
    for cell, dirname in FIT_DIRS.items():
        if cell in missing:
            continue
        folder = root / dirname
        folder.mkdir()
        ids_path = folder / "validation-ids.json"
        ids_path.write_text(json.dumps(ids), encoding="utf8")
        scores_path = folder / "validation-00004.npy"
        np.save(scores_path, scores(rows, 1.0), allow_pickle=False)
        result = {
            "status": (
                "PASS_NATIVE_FULL_EXTENT_TRAIN"
                if cell.startswith("full_")
                else "PASS_NATIVE_LONG_BUDGET_TRAIN"
            ),
            "step": 4,
            "best_step": 2,
            "sealed_audit_loaded": False,
            "source_commit": "abc123",
            "config_sha256": "def456",
            "artifacts": {
                ids_path.name: sha256(ids_path),
                scores_path.name: sha256(scores_path),
            },
        }
        (folder / "result.json").write_text(json.dumps(result), encoding="utf8")
        expected[cell] = scores(rows, 1.0)
    return rows, expected


def test_inspect_conditions_checks_pins_and_reports_missing_cell(tmp_path):
    _, _ = write_artifact_root(tmp_path, missing=("full_fresh",))

    audit, rows, arrays = inspect_conditions(tmp_path, step=4)

    assert audit["status"] == "BLOCKED"
    assert len(audit["analyzer_sha256"]) == 64
    assert audit["missing_cells"] == ["full_fresh"]
    assert audit["validation_rows"] == len(rows) == 8
    assert set(arrays) == {"center_repeat", "center_fresh", "full_repeat"}


def test_inspect_conditions_accepts_four_hash_pinned_matched_cells(tmp_path):
    rows, expected = write_artifact_root(tmp_path)

    audit, loaded_rows, arrays = inspect_conditions(tmp_path, step=4)

    assert audit["status"] == "PASS"
    assert len(audit["analyzer_sha256"]) == 64
    assert loaded_rows == rows
    assert set(arrays) == set(expected)
    for cell in expected:
        np.testing.assert_array_equal(arrays[cell], expected[cell])


def test_inspect_conditions_accepts_explicit_full_fresh_fit_directory(tmp_path):
    rows, _ = write_artifact_root(tmp_path, missing=("full_fresh",))
    custom_dir = "native-full-extent-fresh-long-fit-04"
    folder = tmp_path / custom_dir
    folder.mkdir()
    ids = [
        {"id": row["id"], "option_ids": row["option_ids"], "target_index": row["target_index"]}
        for row in rows
    ]
    ids_path = folder / "validation-ids.json"
    ids_path.write_text(json.dumps(ids), encoding="utf8")
    scores_path = folder / "validation-00004.npy"
    np.save(scores_path, scores(rows, 1.0), allow_pickle=False)
    (folder / "result.json").write_text(
        json.dumps(
            {
                "status": "PASS_NATIVE_FULL_EXTENT_TRAIN",
                "step": 4,
                "best_step": 2,
                "sealed_audit_loaded": False,
                "source_commit": "abc123",
                "config_sha256": "def456",
                "artifacts": {
                    ids_path.name: sha256(ids_path),
                    scores_path.name: sha256(scores_path),
                },
            }
        ),
        encoding="utf8",
    )
    fit_dirs = dict(FIT_DIRS, full_fresh=custom_dir)

    audit, _, arrays = inspect_conditions(tmp_path, step=4, fit_dirs=fit_dirs)

    assert audit["status"] == "PASS"
    assert audit["conditions"]["full_fresh"]["fit_directory"].endswith(custom_dir)
    assert set(arrays) == set(FIT_DIRS)


def test_run_writes_factorial_metrics_when_all_pins_match(tmp_path):
    root = tmp_path / "input"
    output = tmp_path / "comparison"
    write_artifact_root(root)

    result = run(root, output=output, step=4, resamples=20)

    assert result["status"] == "PASS"
    assert result["primary_comparison"].startswith("fixed final step")
    saved = json.loads((output / "factorial-comparison.json").read_text(encoding="utf8"))
    assert saved["metrics"]["interaction"]["macro"]["accuracy"] == 0.0
    assert saved["paired_cluster_bootstrap"]["protocol"]["resamples"] == 20


def test_inspect_conditions_rejects_prediction_hash_mismatch(tmp_path):
    write_artifact_root(tmp_path)
    target = tmp_path / FIT_DIRS["full_fresh"] / "validation-00004.npy"
    np.save(target, np.zeros((8, 2)), allow_pickle=False)

    audit, _, _ = inspect_conditions(tmp_path, step=4)

    assert audit["status"] == "FAIL"
    assert audit["invalid_cells"] == ["full_fresh"]
    assert "endpoint prediction hash mismatch" in audit["conditions"]["full_fresh"]["errors"]


def test_incomplete_factorial_writes_preflight_only_and_never_estimates_effects(tmp_path):
    root = tmp_path / "input"
    output = tmp_path / "preflight"
    write_artifact_root(root, missing=("full_fresh",))

    result = run(root, output=output, step=4, resamples=10)

    assert result["status"] == "BLOCKED"
    assert "metrics" not in result
    assert (output / "preflight.json").is_file()
    assert not (output / "factorial-comparison.json").exists()
