from __future__ import annotations

import hashlib
import json

from scripts.replay_synthetic_av_readouts import replay
from tiny_omni_decision.synthetic_av_evaluation import replay_prediction_metrics


def _write_evaluation(tmp_path, *, expected_accuracy: float | None = None):
    evaluation_dir = tmp_path / "evaluation"
    evaluation_dir.mkdir()
    rows = [
        {
            "architecture": "conditional_av",
            "checkpoint": "best",
            "control": "full",
            "probabilities": [0.5, 0.5],
            "prediction": 1,
            "target": 1,
            "group_id": "group-0",
            "pair_id": "pair-0",
        },
        {
            "architecture": "conditional_av",
            "checkpoint": "best",
            "control": "full",
            "probabilities": [0.8, 0.2],
            "prediction": 0,
            "target": 0,
            "group_id": "group-0",
            "pair_id": "pair-0",
        },
    ]
    predictions_path = evaluation_dir / "predictions.jsonl"
    predictions_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    legacy_rows = [
        {key: value for key, value in row.items() if key != "prediction"}
        for row in rows
    ]
    probability_argmax_metrics = replay_prediction_metrics(legacy_rows)
    saved_prediction_metrics = replay_prediction_metrics(rows)
    expected = {
        **saved_prediction_metrics,
        "accuracy": probability_argmax_metrics["accuracy"],
        "ece": probability_argmax_metrics["ece"],
    }
    if expected_accuracy is not None:
        expected["accuracy"] = expected_accuracy
    evaluation = {
        "status": "PASS_SYNTHETIC_AV_READOUT_VALIDATION",
        "validation_count": len(rows),
        "prediction_sha256": hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
        "results": [
            {
                "architecture": "conditional_av",
                "checkpoint": "best",
                "controls": {"full": expected},
            }
        ],
    }
    (evaluation_dir / "evaluation.json").write_text(
        json.dumps(evaluation), encoding="utf-8"
    )
    return evaluation_dir


def test_replay_reports_only_probability_tie_metric_corrections(tmp_path):
    result = replay(_write_evaluation(tmp_path), tmp_path / "replay")

    assert result["status"] == "PASS_SYNTHETIC_AV_PREDICTION_REPLAY_WITH_METRIC_CORRECTIONS"
    assert result["mismatches"] == []
    assert result["metrics"][0]["tie_prediction_disagreements"] == 1
    assert any(row["metric"] == "accuracy" for row in result["metric_corrections"])


def test_replay_fails_if_saved_evaluation_metrics_do_not_match_probabilities(tmp_path):
    result = replay(_write_evaluation(tmp_path, expected_accuracy=0.0), tmp_path / "replay")

    assert result["status"] == "FAIL"
    assert result["mismatches"]
    assert result["metric_corrections"] == []
