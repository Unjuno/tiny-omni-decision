import hashlib
import json

import numpy as np
import pytest

from scripts.analyze_uci_har_validation_errors import (
    ACTIVITIES,
    analyze_predictions,
    analyze_run,
)


def _rows(model: str):
    rows = []
    for subject in (5, 11):
        for index, activity in enumerate(ACTIVITIES):
            prediction = activity
            if model == "lstm" and activity == "SITTING":
                prediction = "STANDING"
            probabilities = np.full(len(ACTIVITIES), 0.01 / 5, dtype=np.float64)
            probabilities[ACTIVITIES.index(prediction)] = 0.99
            rows.append({
                "sample_id": f"s{subject}-w{index}",
                "subject_id": subject,
                "target": activity,
                "prediction": prediction,
                "probabilities": probabilities,
            })
    return rows


def test_validation_error_analysis_reports_subject_and_class_confusion():
    result = analyze_predictions(_rows("static"), _rows("lstm"))

    assert result["examples"] == 12
    assert result["subjects"] == [5, 11]
    sitting = ACTIVITIES.index("SITTING")
    standing = ACTIVITIES.index("STANDING")
    assert result["static"]["confusion_matrix"][sitting][sitting] == 2
    assert result["lstm"]["confusion_matrix"][sitting][standing] == 2
    assert result["paired_by_subject_and_target"]["5"]["SITTING"][
        "static_correct"
    ] == 1
    assert result["paired_by_subject_and_target"]["5"]["SITTING"][
        "lstm_correct"
    ] == 0


def test_validation_error_analysis_rejects_prediction_id_mismatch():
    static = _rows("static")
    lstm = _rows("lstm")
    lstm[0]["sample_id"] = "different"

    with pytest.raises(ValueError, match="sample IDs/order differ"):
        analyze_predictions(static, lstm)


def test_validation_error_analysis_verifies_frozen_artifact_hashes(tmp_path):
    freeze = {
        "frozen_before_test_read": True,
        "candidates": {
            model: {"best_epoch": epoch, "checkpoint_sha256": ""}
            for model, epoch in (("static", 84), ("lstm", 15))
        },
    }
    artifacts = []
    for model, rows in (("static", _rows("static")), ("lstm", _rows("lstm"))):
        prediction_path = tmp_path / f"{model}-best-validation-predictions.jsonl"
        checkpoint_path = tmp_path / f"{model}-best.safetensors"
        checkpoint_path.write_bytes(f"{model}-checkpoint".encode())
        freeze["candidates"][model]["checkpoint_sha256"] = hashlib.sha256(
            checkpoint_path.read_bytes()
        ).hexdigest()
        serialized_rows = []
        for row in rows:
            serialized_rows.append({
                **{key: value for key, value in row.items() if key != "probabilities"},
                "probabilities": {
                    activity: float(row["probabilities"][index])
                    for index, activity in enumerate(ACTIVITIES)
                },
            })
        prediction_path.write_text("".join(
            json.dumps(row, sort_keys=True) + "\n" for row in serialized_rows
        ), encoding="utf-8")
        for path in (prediction_path, checkpoint_path):
            payload = path.read_bytes()
            artifacts.append({
                "path": path.name,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            })

    (tmp_path / "candidate-freeze.json").write_text(
        json.dumps(freeze), encoding="utf-8"
    )
    (tmp_path / "artifact-index.json").write_text(json.dumps({
        "run_id": "fixture",
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }), encoding="utf-8")

    report = analyze_run(tmp_path)

    assert report["examples"] == 12
    assert report["provenance"]["official_test_loaded"] is False
    assert report["provenance"]["source_signals_loaded"] is False
