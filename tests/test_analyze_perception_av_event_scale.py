import hashlib
import json
from pathlib import Path

import pytest

from scripts.analyze_perception_av_event_scale import analyze_scale_runs


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _feature(event_id: str, video_id: str, split: str) -> dict:
    return {
        "id": event_id,
        "video_id": video_id,
        "split": split,
        "candidate_count": 2,
        "target_index": 0,
        "candidate_order_sha256": "c" * 64,
        "media_sha256": hashlib.sha256(video_id.encode()).hexdigest(),
    }


def _write_run(
    directory: Path,
    *,
    condition: str,
    features_path: Path,
    probabilities: list[float],
    updates: int = 2048,
) -> None:
    directory.mkdir()
    feature_rows = [
        json.loads(line) for line in features_path.read_text().splitlines() if line.strip()
    ]
    fit_rows = [row for row in feature_rows if row["split"] == "fit"]
    result = {
        "status": "PASS_AV_EVENT_READOUT_FIT",
        "condition": condition,
        "seed": 17,
        "optimizer_updates": updates,
        "max_updates": 2048,
        "best_optimizer_updates": 1536,
        "best_selection_metrics": {"accuracy": 0.5, "nll": 0.7, "brier": 0.5, "ece": 0.1},
        "checkpoint_reload_verified": True,
        "fit_events": len(fit_rows),
        "fit_videos": len({row["video_id"] for row in fit_rows}),
        "stopped_by_patience": False,
        "trainable_parameters": 246658 if condition == "joint" else 246115,
        "config": {
            "condition": condition,
            "seed": 17,
            "learning_rate": 0.001,
            "min_delta": 0.0001,
            "hidden_size": 128 if condition == "joint" else 339,
            "evaluation_interval": 512,
            "max_updates": 2048,
            "optimizer": "AdamW",
        },
        "feature_manifest_sha256": hashlib.sha256(features_path.read_bytes()).hexdigest(),
    }
    (directory / "result.json").write_text(json.dumps(result))
    _write_jsonl(
        directory / "development-validation-predictions.jsonl",
        [
            {
                "id": event_id,
                "video_id": video_id,
                "target_index": 0,
                "candidate_count": 2,
                "candidate_order_sha256": "c" * 64,
                "probabilities": probabilities,
            }
            for event_id, video_id in (("dev-1", "dev-video"), ("dev-2", "dev-video2"))
        ],
    )


def test_analyze_scale_runs_pairs_same_development_examples_and_records_bootstrap(tmp_path):
    shared = [
        _feature("selection-1", "selection-video", "selection"),
        _feature("cal-1", "cal-video", "calibration"),
        _feature("dev-1", "dev-video", "development_validation"),
        _feature("dev-2", "dev-video2", "development_validation"),
    ]
    old_features = tmp_path / "old.jsonl"
    expanded_features = tmp_path / "expanded.jsonl"
    _write_jsonl(old_features, [_feature("fit-old", "fit-old-video", "fit"), *shared])
    _write_jsonl(
        expanded_features,
        [
            _feature("fit-old", "fit-old-video", "fit"),
            _feature("fit-new", "fit-new-video", "fit"),
            *shared,
        ],
    )
    run_dirs = {}
    for arm, condition, features_path, probs in (
        ("old_joint", "joint", old_features, [0.4, 0.6]),
        ("expanded_joint", "joint", expanded_features, [0.8, 0.2]),
        ("old_video", "video_only", old_features, [0.6, 0.4]),
        ("expanded_video", "video_only", expanded_features, [0.7, 0.3]),
    ):
        run_dirs[arm] = tmp_path / arm
        _write_run(
            run_dirs[arm],
            condition=condition,
            features_path=features_path,
            probabilities=probs,
        )

    output = tmp_path / "analysis.json"
    report = analyze_scale_runs(
        old_features=old_features,
        expanded_features=expanded_features,
        old_joint=run_dirs["old_joint"],
        expanded_joint=run_dirs["expanded_joint"],
        old_video_only=run_dirs["old_video"],
        expanded_video_only=run_dirs["expanded_video"],
        output=output,
        bootstrap_draws=100,
    )

    assert report["status"] == "PASS_AV_EVENT_SCALE_ANALYSIS"
    joint = report["comparisons"]["joint"]
    assert joint["development_validation"]["old_only"]["accuracy"] == 0.0
    assert joint["development_validation"]["expanded"]["accuracy"] == 1.0
    assert joint["development_validation"]["expanded_minus_old"]["accuracy"] == 1.0
    assert joint["development_validation"]["paired_video_bootstrap"]["accuracy"]["draws"] == 100
    assert output.is_file()


def test_analyze_scale_runs_rejects_nonmatching_development_predictions(tmp_path):
    shared = [
        _feature("selection-1", "selection-video", "selection"),
        _feature("cal-1", "cal-video", "calibration"),
        _feature("dev-1", "dev-video", "development_validation"),
        _feature("dev-2", "dev-video2", "development_validation"),
    ]
    old_features = tmp_path / "old.jsonl"
    expanded_features = tmp_path / "expanded.jsonl"
    _write_jsonl(old_features, [_feature("fit-old", "fit-old-video", "fit"), *shared])
    _write_jsonl(
        expanded_features,
        [
            _feature("fit-old", "fit-old-video", "fit"),
            _feature("fit-new", "fit-new-video", "fit"),
            *shared,
        ],
    )
    run_dirs = {}
    for arm, condition, features_path in (
        ("old_joint", "joint", old_features),
        ("expanded_joint", "joint", expanded_features),
        ("old_video", "video_only", old_features),
        ("expanded_video", "video_only", expanded_features),
    ):
        run_dirs[arm] = tmp_path / arm
        _write_run(
            run_dirs[arm],
            condition=condition,
            features_path=features_path,
            probabilities=[0.5, 0.5],
        )
    prediction_path = run_dirs["expanded_joint"] / "development-validation-predictions.jsonl"
    predictions = [
        json.loads(line) for line in prediction_path.read_text().splitlines() if line.strip()
    ]
    predictions[0]["video_id"] = "different-video"
    _write_jsonl(prediction_path, predictions)

    with pytest.raises(ValueError, match="prediction rows do not match"):
        analyze_scale_runs(
            old_features=old_features,
            expanded_features=expanded_features,
            old_joint=run_dirs["old_joint"],
            expanded_joint=run_dirs["expanded_joint"],
            old_video_only=run_dirs["old_video"],
            expanded_video_only=run_dirs["expanded_video"],
            output=tmp_path / "analysis.json",
            bootstrap_draws=100,
        )


def test_analyze_scale_runs_rejects_schedule_mismatch(tmp_path):
    shared = [
        _feature("selection-1", "selection-video", "selection"),
        _feature("cal-1", "cal-video", "calibration"),
        _feature("dev-1", "dev-video", "development_validation"),
        _feature("dev-2", "dev-video2", "development_validation"),
    ]
    old_features = tmp_path / "old.jsonl"
    expanded_features = tmp_path / "expanded.jsonl"
    _write_jsonl(old_features, [_feature("fit-old", "fit-old-video", "fit"), *shared])
    _write_jsonl(
        expanded_features,
        [
            _feature("fit-old", "fit-old-video", "fit"),
            _feature("fit-new", "fit-new-video", "fit"),
            *shared,
        ],
    )
    run_dirs = {}
    for arm, condition, features_path in (
        ("old_joint", "joint", old_features),
        ("expanded_joint", "joint", expanded_features),
        ("old_video", "video_only", old_features),
        ("expanded_video", "video_only", expanded_features),
    ):
        run_dirs[arm] = tmp_path / arm
        _write_run(
            run_dirs[arm],
            condition=condition,
            features_path=features_path,
            probabilities=[0.5, 0.5],
        )
    result_path = run_dirs["old_joint"] / "result.json"
    result = json.loads(result_path.read_text())
    result["config"]["learning_rate"] = 0.002
    result_path.write_text(json.dumps(result))

    with pytest.raises(ValueError, match="frozen analysis contract"):
        analyze_scale_runs(
            old_features=old_features,
            expanded_features=expanded_features,
            old_joint=run_dirs["old_joint"],
            expanded_joint=run_dirs["expanded_joint"],
            old_video_only=run_dirs["old_video"],
            expanded_video_only=run_dirs["expanded_video"],
            output=tmp_path / "analysis.json",
            bootstrap_draws=100,
        )
