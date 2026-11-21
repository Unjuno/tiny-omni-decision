from __future__ import annotations

import pytest

from scripts.analyze_habit_train_only_grouped_oof import (
    connected_components,
    paired_component_bootstrap,
)


def test_components_join_shared_task_and_video_identity() -> None:
    records = [
        {"id": "a", "split": "train", "sid": "task-a", "video_sha256": "v1"},
        {"id": "b", "split": "train", "sid": "task-a", "video_sha256": "v2"},
        {"id": "c", "split": "train", "sid": "task-b", "video_sha256": "v2"},
        {"id": "d", "split": "train", "sid": "task-c", "video_sha256": "v3"},
        {"id": "audit", "split": "sealed_audit", "sid": "task-z", "video_sha256": "vz"},
    ]
    assert connected_components(records[:4]) == [["a", "b", "c"], ["d"]]


def test_bootstrap_requires_exact_paired_oof_coverage() -> None:
    records = [
        {"id": "a", "split": "train", "sid": "task-a", "video_sha256": "v1"},
        {"id": "b", "split": "train", "sid": "task-b", "video_sha256": "v2"},
    ]
    def prediction(sample_id: str) -> dict:
        return {
            "sample_id": sample_id,
            "target_index": 0,
            "prediction_index": 0,
            "probabilities": [0.8, 0.2],
        }
    metrics = {"predictions": [prediction("a"), prediction("b")]}
    report = {
        "curve_epochs": [1],
        "out_of_fold_metrics_by_mode_and_epoch": {
            mode: {"1": metrics} for mode in ("video_state", "video_only", "state_only")
        },
    }
    result = paired_component_bootstrap(report, records, draws=100, seed=17)
    assert result["component_count"] == 2
    assert result["sealed_audit_accessed"] is False
    assert result["contrasts"]["epoch_1_video_state_minus_video_only"]["accuracy"][
        "video_state_minus_comparator"
    ] == 0.0
    report["out_of_fold_metrics_by_mode_and_epoch"]["state_only"]["1"]["predictions"] = []
    with pytest.raises(ValueError, match="align exactly"):
        paired_component_bootstrap(report, records, draws=10)
