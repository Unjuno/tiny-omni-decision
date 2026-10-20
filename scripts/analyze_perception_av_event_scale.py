"""Analyze matched old-only versus expanded frozen AV event readout runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.analyze_av_event_readout_comparison import (
    _assert_matched,
    paired_video_bootstrap,
    score_rows,
)
from scripts.av_event_split_validation import validate_video_splits

EXPECTED_SEED = 17
EXPECTED_UPDATES = 2048
SPLITS = ("selection", "calibration", "development_validation")
METRICS = ("accuracy", "nll", "brier", "ece")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"feature manifest is empty or malformed: {path}")
    validate_video_splits(rows)
    return rows


def _rows_by_split(rows: list[dict[str, Any]], split: str) -> dict[str, dict[str, Any]]:
    selected = [row for row in rows if row["split"] == split]
    return {row["id"]: row for row in selected}


def _identity(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["video_id"],
        row.get("media_sha256"),
        row["target_index"],
        row["candidate_count"],
        row["candidate_order_sha256"],
    )


def _prediction_identity(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["video_id"],
        row["target_index"],
        row["candidate_count"],
        row["candidate_order_sha256"],
    )


def _validate_feature_pair(old_path: Path, expanded_path: Path):
    old_rows = _read_jsonl(old_path)
    expanded_rows = _read_jsonl(expanded_path)
    old_fit = _rows_by_split(old_rows, "fit")
    expanded_fit = _rows_by_split(expanded_rows, "fit")
    if not old_fit or not expanded_fit or not set(old_fit) < set(expanded_fit):
        raise ValueError("expanded fit must strictly include all old fit events plus new events")
    for event_id, old in old_fit.items():
        if _identity(old) != _identity(expanded_fit[event_id]):
            raise ValueError("expanded fit changed an existing old fit event")
    added_fit = [row for event_id, row in expanded_fit.items() if event_id not in old_fit]
    old_videos = {row["video_id"] for row in old_fit.values()}
    old_media = {row.get("media_sha256") for row in old_fit.values()}
    if old_videos & {row["video_id"] for row in added_fit}:
        raise ValueError("new fit videos overlap the old fit video identities")
    if old_media & {row.get("media_sha256") for row in added_fit}:
        raise ValueError("new fit media overlap the old fit media identities")

    for name, rows in (("old-only", old_rows), ("expanded", expanded_rows)):
        video_media: dict[str, str] = {}
        media_video: dict[str, str] = {}
        for row in rows:
            video_id, media_hash = row["video_id"], row.get("media_sha256")
            if not isinstance(media_hash, str):
                raise ValueError(f"{name} feature row is missing media identity")
            if video_media.setdefault(video_id, media_hash) != media_hash:
                raise ValueError(f"{name} video maps to multiple media identities")
            if media_video.setdefault(media_hash, video_id) != video_id:
                raise ValueError(f"{name} media identity maps to multiple videos")

    for split in SPLITS:
        old_split = _rows_by_split(old_rows, split)
        expanded_split = _rows_by_split(expanded_rows, split)
        if old_split.keys() != expanded_split.keys():
            raise ValueError(f"old-only and expanded {split} IDs differ")
        if any(_identity(old_split[key]) != _identity(expanded_split[key]) for key in old_split):
            raise ValueError(f"old-only and expanded {split} example identities differ")
    return old_rows, expanded_rows


def _load_run(
    run_dir: Path,
    *,
    expected_condition: str,
    features_path: Path,
    expected_dev: dict[str, dict[str, Any]],
    expected_fit_events: int,
    expected_fit_videos: int,
) -> dict[str, Any]:
    result_path = run_dir / "result.json"
    prediction_path = run_dir / "development-validation-predictions.jsonl"
    result = _read_json(result_path)
    config = result.get("config", {})
    expected_hidden_size = 128 if expected_condition == "joint" else 339
    expected_parameters = 246_658 if expected_condition == "joint" else 246_115
    if (
        result.get("status") != "PASS_AV_EVENT_READOUT_FIT"
        or result.get("condition") != expected_condition
        or result.get("seed") != EXPECTED_SEED
        or result.get("optimizer_updates") != EXPECTED_UPDATES
        or result.get("max_updates") != EXPECTED_UPDATES
        or result.get("checkpoint_reload_verified") is not True
        or result.get("feature_manifest_sha256") != _sha256(features_path)
        or result.get("fit_events") != expected_fit_events
        or result.get("fit_videos") != expected_fit_videos
        or result.get("trainable_parameters") != expected_parameters
        or result.get("best_optimizer_updates") not in {512, 1024, 1536, 2048}
        or result.get("stopped_by_patience") is not False
        or config.get("condition") != expected_condition
        or config.get("seed") != EXPECTED_SEED
        or config.get("learning_rate") != 0.001
        or config.get("min_delta") != 0.0001
        or config.get("hidden_size") != expected_hidden_size
        or config.get("evaluation_interval") != 512
        or config.get("max_updates") != EXPECTED_UPDATES
        or config.get("optimizer") != "AdamW"
    ):
        raise ValueError(f"run does not match the frozen analysis contract: {run_dir}")
    predictions = [
        json.loads(line)
        for line in prediction_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_id = {row.get("id"): row for row in predictions}
    if len(by_id) != len(predictions) or by_id.keys() != expected_dev.keys():
        raise ValueError("prediction rows do not match development-validation IDs")
    for event_id, feature in expected_dev.items():
        prediction = by_id[event_id]
        feature_identity = (
            feature["video_id"],
            feature["target_index"],
            feature["candidate_count"],
            feature["candidate_order_sha256"],
        )
        if _prediction_identity(prediction) != feature_identity:
            raise ValueError("prediction rows do not match development-validation labels/order")
    return {
        "result": result,
        "result_sha256": _sha256(result_path),
        "predictions": predictions,
        "predictions_sha256": _sha256(prediction_path),
        "metrics": score_rows(predictions),
    }


def _compare_pair(old: dict[str, Any], expanded: dict[str, Any], *, draws: int) -> dict[str, Any]:
    _assert_matched(expanded["predictions"], old["predictions"])
    expanded_metrics = expanded["metrics"]
    old_metrics = old["metrics"]
    return {
        "development_validation": {
            "old_only": old_metrics,
            "expanded": expanded_metrics,
            "expanded_minus_old": {
                metric: expanded_metrics[metric] - old_metrics[metric] for metric in METRICS
            },
            "paired_video_bootstrap": paired_video_bootstrap(
                expanded["predictions"], old["predictions"], draws=draws, seed=EXPECTED_SEED
            ),
        },
        "best_checkpoint": {
            "old_only_step": old["result"].get("best_optimizer_updates"),
            "expanded_step": expanded["result"].get("best_optimizer_updates"),
            "old_only_selection_metrics": old["result"].get("best_selection_metrics"),
            "expanded_selection_metrics": expanded["result"].get("best_selection_metrics"),
        },
        "trainable_parameters": {
            "old_only": old["result"].get("trainable_parameters"),
            "expanded": expanded["result"].get("trainable_parameters"),
        },
        "run_hashes": {
            "old_only_result": old["result_sha256"],
            "expanded_result": expanded["result_sha256"],
            "old_only_development_predictions": old["predictions_sha256"],
            "expanded_development_predictions": expanded["predictions_sha256"],
        },
    }


def analyze_scale_runs(
    *,
    old_features: Path,
    expanded_features: Path,
    old_joint: Path,
    expanded_joint: Path,
    old_video_only: Path,
    expanded_video_only: Path,
    output: Path,
    bootstrap_draws: int = 10_000,
) -> dict[str, Any]:
    if output.exists():
        raise ValueError("analysis output must be new")
    if bootstrap_draws < 100:
        raise ValueError("paired video bootstrap requires at least 100 draws")
    old_rows, expanded_rows = _validate_feature_pair(old_features, expanded_features)
    old_fit_rows = [row for row in old_rows if row["split"] == "fit"]
    expanded_fit_rows = [row for row in expanded_rows if row["split"] == "fit"]
    old_fit_videos = len({row["video_id"] for row in old_fit_rows})
    expanded_fit_videos = len({row["video_id"] for row in expanded_fit_rows})
    old_dev = _rows_by_split(old_rows, "development_validation")
    expanded_dev = _rows_by_split(expanded_rows, "development_validation")
    if old_dev.keys() != expanded_dev.keys():
        raise ValueError("development-validation rows are not paired")
    old_dev = {key: old_dev[key] for key in sorted(old_dev)}
    expanded_dev = {key: expanded_dev[key] for key in sorted(expanded_dev)}
    old_joint_run = _load_run(
        old_joint,
        expected_condition="joint",
        features_path=old_features,
        expected_dev=old_dev,
        expected_fit_events=len(old_fit_rows),
        expected_fit_videos=old_fit_videos,
    )
    expanded_joint_run = _load_run(
        expanded_joint,
        expected_condition="joint",
        features_path=expanded_features,
        expected_dev=expanded_dev,
        expected_fit_events=len(expanded_fit_rows),
        expected_fit_videos=expanded_fit_videos,
    )
    old_video_run = _load_run(
        old_video_only,
        expected_condition="video_only",
        features_path=old_features,
        expected_dev=old_dev,
        expected_fit_events=len(old_fit_rows),
        expected_fit_videos=old_fit_videos,
    )
    expanded_video_run = _load_run(
        expanded_video_only,
        expected_condition="video_only",
        features_path=expanded_features,
        expected_dev=expanded_dev,
        expected_fit_events=len(expanded_fit_rows),
        expected_fit_videos=expanded_fit_videos,
    )
    result = {
        "status": "PASS_AV_EVENT_SCALE_ANALYSIS",
        "seed": EXPECTED_SEED,
        "optimizer_updates_per_run": EXPECTED_UPDATES,
        "development_validation_video_count": len(
            {row["video_id"] for row in old_dev.values()}
        ),
        "development_validation_event_count": len(old_dev),
        "bootstrap": {
            "draws": bootstrap_draws,
            "seed": EXPECTED_SEED,
            "cluster_unit": "video_id",
            "comparison_direction": "expanded minus old-only",
        },
        "feature_manifests": {
            "old_only_sha256": _sha256(old_features),
            "expanded_sha256": _sha256(expanded_features),
        },
        "comparisons": {
            "joint": _compare_pair(old_joint_run, expanded_joint_run, draws=bootstrap_draws),
            "video_only": _compare_pair(old_video_run, expanded_video_run, draws=bootstrap_draws),
        },
        "scope": {
            "evaluation_split": "development_validation",
            "checkpoint_selection_split": "selection",
            "test_opened": False,
            "tuning_on_development_validation": False,
            "claim": (
                "task-specific development evidence for added fit-video coverage; not a blind test"
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    required_paths = (
        "old-features",
        "expanded-features",
        "old-joint",
        "expanded-joint",
        "old-video-only",
        "expanded-video-only",
        "output",
    )
    for name in required_paths:
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    args = parser.parse_args()
    result = analyze_scale_runs(
        old_features=args.old_features,
        expanded_features=args.expanded_features,
        old_joint=args.old_joint,
        expanded_joint=args.expanded_joint,
        old_video_only=args.old_video_only,
        expanded_video_only=args.expanded_video_only,
        output=args.output,
        bootstrap_draws=args.bootstrap_draws,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
