"""Evaluate unchanged DROID readouts on the frozen later-shard group holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tiny_omni_decision.droid_next_action_readout import (  # noqa: E402
    ACTION_OPTIONS,
    make_readout_features,
    multiclass_probability_metrics,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _predict_softmax(model: dict[str, Any], features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(model["mean"]):
        raise ValueError("feature width differs from the fitted readout")
    if not np.isfinite(x).all():
        raise ValueError("features must be finite")
    normalized = np.clip((x - model["mean"]) / model["scale"], -20.0, 20.0)
    logits = normalized @ model["weight"].T + model["bias"]
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / exponential.sum(axis=1, keepdims=True)


def evaluate_saved_readouts(
    *,
    task: dict[str, Any],
    video: np.ndarray,
    sensor: np.ndarray,
    readouts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    rows = task["splits"]["validation"]["rows"]
    if not rows or len(video) != len(rows) or len(sensor) != len(rows):
        raise ValueError("task rows and feature arrays must be non-empty and aligned")
    if task.get("model_training") is not False or task.get("sealed_audit_loaded") is not False:
        raise ValueError("transfer evaluation must be frozen and audit-free")
    ids = [str(row["id"]) for row in rows]
    groups = [str(row["group_id"]) for row in rows]
    if len(set(ids)) != len(ids) or len(set(groups)) != len(groups):
        raise ValueError("evaluation IDs and recording groups must be unique")
    if any(row.get("split") != "validation" for row in rows):
        raise ValueError("all transfer rows must be evaluation-only")
    targets = np.asarray(
        [ACTION_OPTIONS.index(row["target_option_id"]) for row in rows], dtype=np.int64
    )
    features = make_readout_features(video, sensor)
    per_site_indices: dict[str, list[int]] = defaultdict(list)
    per_shard_indices: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        per_site_indices[str(row["site"])].append(index)
        per_shard_indices[int(row["shard_index"])].append(index)

    arm_metrics: dict[str, Any] = {}
    predictions: list[dict[str, Any]] = []
    probability_by_arm: dict[str, np.ndarray] = {}
    for arm in ("video", "sensor", "video_sensor"):
        model = readouts[arm]
        probabilities = _predict_softmax(model, features[arm])
        probability_by_arm[arm] = probabilities
        arm_metrics[arm] = {
            "overall": multiclass_probability_metrics(targets, probabilities),
            "per_site": {
                site: multiclass_probability_metrics(targets[indices], probabilities[indices])
                for site, indices in sorted(per_site_indices.items())
            },
            "per_shard": {
                str(shard): multiclass_probability_metrics(targets[indices], probabilities[indices])
                for shard, indices in sorted(per_shard_indices.items())
            },
        }
    for index, row in enumerate(rows):
        predictions.append(
            {
                "id": row["id"],
                "group_id": row["group_id"],
                "site": row["site"],
                "shard_index": int(row["shard_index"]),
                "target_option_id": row["target_option_id"],
                "predictions": {
                    arm: ACTION_OPTIONS[int(probability_by_arm[arm][index].argmax())]
                    for arm in ("video", "sensor", "video_sensor")
                },
                "probabilities": {
                    arm: probability_by_arm[arm][index].tolist()
                    for arm in ("video", "sensor", "video_sensor")
                },
            }
        )
    return {
        "evaluation_count": len(rows),
        "unique_sites": len(per_site_indices),
        "unique_recording_groups": len(set(groups)),
        "label_counts": {
            name: int((targets == index).sum()) for index, name in enumerate(ACTION_OPTIONS)
        },
        "metrics": arm_metrics,
        "predictions": predictions,
    }


def validate_feature_progress(
    progress: list[dict[str, Any]], rows: list[dict[str, Any]]
) -> None:
    """Validate append-ordered extraction records through their stable row indices."""
    if len(progress) != len(rows):
        raise ValueError("feature progress does not cover every evaluation row")
    seen: set[int] = set()
    for item in progress:
        index = int(item.get("index", -1))
        if index < 0 or index >= len(rows) or index in seen:
            raise ValueError("feature progress contains an invalid or duplicate row index")
        row = rows[index]
        if item.get("id") != row["id"] or item.get("split") != "validation":
            raise ValueError("feature progress ID or split differs from the frozen task")
        expected_label = ACTION_OPTIONS.index(row["target_option_id"])
        if int(item.get("label", -1)) != expected_label:
            raise ValueError("feature progress target differs from the frozen task")
        seen.add(index)
    if seen != set(range(len(rows))):
        raise ValueError("feature progress is missing task rows")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-manifest", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--readout-dir", type=Path, required=True)
    parser.add_argument("--prior-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite transfer evaluation: {output}")
    if output.is_relative_to(ROOT.resolve()):
        raise ValueError("evaluation output must remain outside the repository")

    task_raw = args.task_manifest.read_bytes()
    task = json.loads(task_raw.decode("utf-8-sig"))
    previous = json.loads(args.prior_report.read_text(encoding="utf-8"))
    if task.get("task_id") != "droid-later-shard-frozen-readout-transfer-v1":
        raise ValueError("unexpected transfer task ID")
    rows = task["splits"]["validation"]["rows"]
    if task.get("model_training") is not False or task.get("sealed_audit_loaded") is not False:
        raise ValueError("task must be inference-only and sealed-audit-free")
    summary_path = args.feature_cache / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("status") != "PASS"
        or summary.get("count") != len(rows)
        or summary.get("task_manifest_sha256") != hashlib.sha256(task_raw).hexdigest()
        or summary.get("model_training") is not False
        or summary.get("sealed_audit_loaded") is not False
    ):
        raise ValueError("feature cache does not match the frozen task")
    for filename in ("video_features.npy", "sensor_sequences.npy", "labels.npy", "progress.jsonl"):
        if sha256_file(args.feature_cache / filename) != summary.get(f"{filename}_sha256"):
            raise ValueError(f"feature-cache hash mismatch: {filename}")
    progress = [
        json.loads(line)
        for line in (args.feature_cache / "progress.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    validate_feature_progress(progress, rows)
    video = np.load(args.feature_cache / "video_features.npy", mmap_mode="r")
    sensor = np.load(args.feature_cache / "sensor_sequences.npy", mmap_mode="r")
    label_array = np.load(args.feature_cache / "labels.npy", mmap_mode="r")
    if video.shape != (len(rows), 3, 8, 768) or sensor.shape != (len(rows), 8, 14):
        raise ValueError("unexpected frozen feature shape")
    if not np.array_equal(
        label_array,
        np.asarray([ACTION_OPTIONS.index(row["target_option_id"]) for row in rows]),
    ):
        raise ValueError("cached labels differ from the task manifest")

    expected_hashes = previous["artifacts"]["readout_sha256"]
    readouts: dict[str, dict[str, Any]] = {}
    for arm in ("video", "sensor", "video_sensor"):
        path = args.readout_dir / f"{arm}-readout.npz"
        if sha256_file(path) != expected_hashes[path.name]:
            raise ValueError(f"frozen reference readout hash mismatch: {arm}")
        with np.load(path, allow_pickle=False) as weights:
            readouts[arm] = {name: weights[name] for name in ("mean", "scale", "weight", "bias")}

    evaluated = evaluate_saved_readouts(
        task=task,
        video=np.asarray(video),
        sensor=np.asarray(sensor),
        readouts=readouts,
    )
    output.mkdir(parents=True, exist_ok=False)
    prediction_raw = "".join(
        json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
        for row in evaluated.pop("predictions")
    ).encode("utf-8")
    (output / "predictions.jsonl").write_bytes(prediction_raw)
    report = {
        "status": "PASS_FROZEN_READOUT_TRANSFER_EVALUATED",
        "experiment_id": "droid-r2-later-shard-frozen-readout-transfer-v1",
        "task_manifest_sha256": hashlib.sha256(task_raw).hexdigest(),
        "feature_cache_summary_sha256": sha256_file(summary_path),
        "prior_report_sha256": sha256_file(args.prior_report),
        "prior_task_manifest_sha256": previous["artifacts"]["task_manifest_sha256"],
        "frozen_readout_sha256": expected_hashes,
        "prediction_sha256": hashlib.sha256(prediction_raw).hexdigest(),
        "feature_cache_sha256": {
            name: summary[f"{name}_sha256"]
            for name in ("video_features.npy", "sensor_sequences.npy", "labels.npy")
        },
        "source_scope": task["source_partition"],
        "claim_scope": task["claim_scope"],
        "comparison": evaluated,
        "prior_development_validation": {
            "count": previous["task"]["validation_examples"],
            "metrics": previous["metrics"]["validation"],
            "not_paired_with_this_evaluation": True,
        },
        "limitations": [
            "The later-shard sample is official-train data and is not a sealed or blind audit.",
            (
                "Episode and recording-group identities are new; all sampled sites "
                "were already seen in the earlier cohort."
            ),
            (
                "This is narrow seven-way robot Cartesian next-action prediction, "
                "not general Omni decision quality."
            ),
            (
                "No model was fit, selected, calibrated, or changed; the three "
                "original readouts were replayed unchanged."
            ),
            "No training or sealed-audit split was loaded.",
        ],
        "model_training": False,
        "sealed_audit_loaded": False,
    }
    (output / "evaluation-report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": report["status"],
        "evaluation_count": evaluated["evaluation_count"],
        "label_counts": evaluated["label_counts"],
        "metrics": {
            arm: block["overall"] for arm, block in evaluated["metrics"].items()
        },
        "sealed_audit_loaded": False,
    }, indent=2))


if __name__ == "__main__":
    main()
