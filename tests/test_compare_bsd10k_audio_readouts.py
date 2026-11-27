from __future__ import annotations

import json

import pytest

from scripts.compare_bsd10k_audio_readouts import compare
from scripts.train_bsd10k_audio_readout import LABELS


def _prediction(sound_id: str, target: str) -> dict:
    target_index = LABELS.index(target)
    probabilities = [0.2 / (len(LABELS) - 1)] * len(LABELS)
    probabilities[target_index] = 0.8
    return {
        "sound_id": sound_id,
        "target_class_id": target,
        "probabilities_uncalibrated": probabilities,
        "probabilities_temperature_scaled": probabilities,
    }


def test_comparison_requires_same_validation_order_and_reports_group_bootstrap(tmp_path):
    validation_rows = []
    predictions = []
    for index, label in enumerate(LABELS):
        sound_id = str(index + 1)
        uploader_hash = f"{index + 1:064x}"
        validation_rows.append(
            {
                "sound_id": sound_id,
                "split": "validation",
                "uploader_group_sha256": uploader_hash,
            }
        )
        predictions.append(_prediction(sound_id, label))

    manifest = tmp_path / "split.json"
    manifest.write_text(json.dumps({"examples": validation_rows}), encoding="utf-8")
    baseline_predictions = tmp_path / "baseline.jsonl"
    candidate_predictions = tmp_path / "candidate.jsonl"
    rendered = "".join(json.dumps(row) + "\n" for row in predictions)
    baseline_predictions.write_text(rendered, encoding="utf-8")
    candidate_predictions.write_text(rendered, encoding="utf-8")
    config = {"batch_size": 128}
    baseline_report = {
        "train_examples": 100,
        "train_unique_uploader_groups": 50,
        "fit_examples": 80,
        "epochs_run": 3,
        "selected_epoch": 3,
        "config": config,
    }
    candidate_report = {
        **baseline_report,
        "optimizer_steps": 2,
        "training_examples_seen": 200,
    }
    baseline_path, candidate_path = (
        tmp_path / "baseline-report.json",
        tmp_path / "candidate-report.json",
    )
    baseline_path.write_text(json.dumps(baseline_report), encoding="utf-8")
    candidate_path.write_text(json.dumps(candidate_report), encoding="utf-8")

    report = compare(
        baseline_path,
        baseline_predictions,
        candidate_path,
        candidate_predictions,
        manifest,
        tmp_path / "comparison.json",
        replicates=50,
    )

    assert report["status"] == "PASS_IDENTICAL_VALIDATION_IDS"
    assert report["validation"]["same_ids_same_order_and_same_targets"] is True
    assert report["validation"]["grouped_bootstrap"]["independent_groups"] == len(LABELS)
    assert report["baseline"]["optimizer_steps"] == 3
    assert report["candidate"]["optimizer_steps"] == 2

    changed = predictions[1:] + predictions[:1]
    candidate_predictions.write_text(
        "".join(json.dumps(row) + "\n" for row in changed), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="validation IDs"):
        compare(
            baseline_path,
            baseline_predictions,
            candidate_path,
            candidate_predictions,
            manifest,
            tmp_path / "bad-comparison.json",
            replicates=10,
        )
