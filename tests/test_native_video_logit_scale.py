import hashlib
import json

import numpy as np

from scripts.diagnose_native_video_logit_scale import (
    analyze,
    paired_scene_bootstrap,
    per_example,
    summarize,
)


def test_per_example_and_summary_report_error_confidence_and_logit_scale():
    records = [
        {"id": "video:clevrer-validation-10005-q001", "option_ids": ["a", "b"], "target_index": 0},
        {"id": "video:clevrer-validation-10006-q001", "option_ids": ["a", "b"], "target_index": 0},
    ]
    rows = per_example(np.asarray([[4.0, 0.0], [0.0, 4.0]]), records)
    result = summarize(rows)

    assert result["n"] == 2
    assert result["accuracy"] == 0.5
    assert result["wrong_count"] == 1
    assert result["mean_confidence_when_wrong"] > 0.9
    assert result["median_logit_span"] == 4.0
    assert result["median_top_two_margin"] == 4.0


def test_scene_bootstrap_is_deterministic_and_keeps_queries_grouped():
    values = np.asarray([1.0, 1.0, 0.0, 0.0])
    groups = ["scene-1", "scene-1", "scene-2", "scene-2"]

    first = paired_scene_bootstrap(values, groups, iterations=100, seed=17)
    second = paired_scene_bootstrap(values, groups, iterations=100, seed=17)

    assert first == second
    assert first["scene_count"] == 2
    assert first["mean_scene_paired_difference"] == 0.5
    assert first["percentile_95_interval"][0] >= 0.0
    assert first["percentile_95_interval"][1] <= 1.0


def test_analyze_verifies_pinned_prediction_arrays_and_avoids_audit(tmp_path):
    records = [
        {"id": "video:clevrer-validation-10005-q001", "option_ids": ["a", "b"], "target_index": 0},
        {"id": "video:clevrer-validation-10005-q002", "option_ids": ["a", "b"], "target_index": 0},
        {"id": "video:clevrer-validation-10006-q001", "option_ids": ["a", "b"], "target_index": 0},
        {"id": "video:clevrer-validation-10006-q002", "option_ids": ["a", "b"], "target_index": 0},
    ]
    ids_path = tmp_path / "validation-ids.json"
    ids_path.write_text(json.dumps(records), encoding="utf-8")
    reference = np.asarray([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    endpoint = np.asarray([[8.0, 0.0], [0.0, 8.0], [8.0, 0.0], [0.0, 8.0]], dtype=np.float32)
    names = {512: "validation-0512.npy", 32768: "validation-32768.npy"}
    predictions = {512: reference, 32768: endpoint}
    for step, name in names.items():
        np.save(tmp_path / name, predictions[step], allow_pickle=False)
    artifacts = {
        "validation-ids.json": hashlib.sha256(ids_path.read_bytes()).hexdigest(),
        **{
            name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
            for name in names.values()
        },
    }
    source_result = {
        "status": "PASS_NATIVE_FULL_EXTENT_TRAIN",
        "sealed_audit_loaded": False,
        "step": 32768,
        "best_step": 4096,
        "source_commit": "abc123",
        "config_sha256": "config-hash",
        "artifacts": artifacts,
    }
    (tmp_path / "result.json").write_text(json.dumps(source_result), encoding="utf-8")

    result = analyze(tmp_path, [512, 32768], iterations=100)

    assert result["status"] == "PASS_READ_ONLY_VIDEO_LOGIT_DIAGNOSTIC"
    assert result["video_queries"] == 4
    assert result["independent_scene_ids"] == 2
    assert result["step_metrics"]["512"]["accuracy"] == 0.5
    assert (
        result["step_metrics"]["32768"]["mean_confidence"]
        > result["step_metrics"]["512"]["mean_confidence"]
    )
    assert result["paired_endpoint_transitions"]["correct_to_correct"] == 2
    assert result["sealed_audit_read"] is False
    assert result["model_loaded"] is False
