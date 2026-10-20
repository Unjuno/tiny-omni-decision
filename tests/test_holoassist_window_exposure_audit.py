import json

import pytest

from scripts.analyze_holoassist_window_exposure import analyze_files


def _write_inputs(tmp_path, *, alter_prediction_id=False):
    windows = [
        {
            "id": "s1:w0",
            "split": "validation",
            "loss_mask": True,
            "session_id": "s1",
            "target": 0,
            "window_start_seconds": 0.0,
            "window_end_seconds": 30.0,
        },
        {
            "id": "s1:w1",
            "split": "validation",
            "loss_mask": True,
            "session_id": "s1",
            "target": 1,
            "window_start_seconds": 30.0,
            "window_end_seconds": 41.0,
        },
        {
            "id": "s2:w0",
            "split": "validation",
            "loss_mask": True,
            "session_id": "s2",
            "target": 1,
            "window_start_seconds": 0.0,
            "window_end_seconds": 30.0,
        },
        {
            "id": "s2:w1",
            "split": "validation",
            "loss_mask": False,
            "session_id": "s2",
            "target": None,
            "window_start_seconds": 30.0,
            "window_end_seconds": 35.0,
        },
    ]
    windows_path = tmp_path / "windows.jsonl"
    windows_path.write_text("".join(json.dumps(row) + "\n" for row in windows), encoding="utf-8")
    av = [
        {"id": "s1:w0", "session_id": "s1", "target": 0, "probabilities": [0.8, 0.2]},
        {"id": "s1:w1", "session_id": "s1", "target": 1, "probabilities": [0.7, 0.3]},
        {"id": "s2:w0", "session_id": "s2", "target": 1, "probabilities": [0.2, 0.8]},
    ]
    avs = [
        {"id": "s1:w0", "session_id": "s1", "target": 0, "probabilities": [0.7, 0.3]},
        {"id": "s1:w1", "session_id": "s1", "target": 1, "probabilities": [0.4, 0.6]},
        {"id": "s2:w0", "session_id": "s2", "target": 1, "probabilities": [0.1, 0.9]},
    ]
    if alter_prediction_id:
        avs[0]["id"] = "other"
    paths = {}
    for name, rows in {"av": av, "sensor_only": av, "avs": avs}.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(rows), encoding="utf-8")
        paths[name] = path
    return windows_path, paths


def test_audit_separates_full_and_tail_windows_and_excludes_unlabeled_context(tmp_path):
    windows, predictions = _write_inputs(tmp_path)

    result = analyze_files(
        windows,
        predictions,
        full_window_seconds=30.0,
        replicates=100,
        seed=17,
    )

    assert result["dataset"]["labeled_windows"] == 3
    assert result["dataset"]["unlabeled_context_windows_excluded"] == 1
    full = result["strata"]["full"]
    tail = result["strata"]["tail"]
    assert (full["n_windows"], full["n_sessions"], full["positive_rate"]) == (2, 2, 0.5)
    assert (tail["n_windows"], tail["n_sessions"], tail["positive_rate"]) == (1, 1, 1.0)
    assert full["metrics"]["av"]["accuracy"] == 1.0
    assert tail["metrics"]["av"]["accuracy"] == 0.0
    assert result["strata"]["full"]["comparisons"]["avs_minus_av"]["accuracy"] == 0.0


def test_audit_rejects_prediction_id_or_label_mismatch(tmp_path):
    windows, predictions = _write_inputs(tmp_path, alter_prediction_id=True)

    with pytest.raises(ValueError, match="prediction IDs must match labeled windows"):
        analyze_files(windows, predictions, full_window_seconds=30.0, replicates=100, seed=17)
