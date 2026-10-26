"""Train fixed frozen-feature DROID readouts on a group-safe development split."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from tiny_omni_decision.droid_early_outcome_readout import (
    binary_probability_metrics,
    fit_l2_logistic,
    predict_success_probability,
)

EXPECTED_TASK_SHA256 = "b882b66f8be2dd38aa7d69badd3d00b93f58769a9daa2be5f38088fbe084ec80"
EXPECTED_SHARD_MANIFEST_SHA256 = "2bfb5626d203344bf2bd976f711c0ed5f7b4c1979932b7b48565329099207e64"
EXPECTED_CHECKPOINT_SHA256 = "61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(task_dir: Path) -> list[dict[str, Any]]:
    result = []
    for name in ("train.jsonl", "validation.jsonl"):
        result.extend(
            json.loads(line)
            for line in (task_dir / name).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return result


def _validate_inputs(
    cache_dir: Path, task_dir: Path, checkpoint: Path
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    rows = _rows(task_dir)
    task_hash = sha256_file(task_dir / "task-manifest.json")
    if task_hash != EXPECTED_TASK_SHA256:
        raise ValueError("frozen task manifest SHA-256 mismatch")
    summary = json.loads((cache_dir / "summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "PASS" or summary.get("count") != len(rows):
        raise ValueError("complete frozen feature extraction is required")
    if summary.get("task_manifest_sha256") != task_hash:
        raise ValueError("feature cache references a different task manifest")
    if summary.get("shard_manifest_sha256") != EXPECTED_SHARD_MANIFEST_SHA256:
        raise ValueError("feature cache references a different shard manifest")
    checkpoint_hash = sha256_file(checkpoint)
    if (
        checkpoint_hash != EXPECTED_CHECKPOINT_SHA256
        or summary.get("checkpoint_sha256") != checkpoint_hash
    ):
        raise ValueError("frozen VideoPrism checkpoint identity mismatch")
    pinned_outputs = {
        "video_features.npy": "video_features.npy_sha256",
        "sensor_sequences.npy": "sensor_sequences.npy_sha256",
        "labels.npy": "labels.npy_sha256",
        "progress.jsonl": "progress.jsonl_sha256",
    }
    for filename, key in pinned_outputs.items():
        if sha256_file(cache_dir / filename) != summary.get(key):
            raise ValueError(f"feature cache artifact hash mismatch: {filename}")
    if (
        summary.get("sealed_audit_loaded") is not False
        or summary.get("model_training") is not False
    ):
        raise ValueError("feature cache must be inference-only and sealed-audit-free")

    if len(rows) != 537 or sum(row["split"] == "train" for row in rows) != 337:
        raise ValueError("unexpected frozen task split sizes")
    if sum(row["split"] == "validation" for row in rows) != 200:
        raise ValueError("unexpected frozen validation size")
    train_groups = {row["group_id"] for row in rows if row["split"] == "train"}
    validation_groups = {row["group_id"] for row in rows if row["split"] == "validation"}
    if train_groups & validation_groups:
        raise ValueError("train/validation recording group overlap")

    progress = [
        json.loads(line)
        for line in (cache_dir / "progress.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    progress_by_index = {int(entry["index"]): entry for entry in progress}
    if len(progress_by_index) != len(rows) or set(progress_by_index) != set(range(len(rows))):
        raise ValueError("feature progress index coverage mismatch")
    video = np.load(cache_dir / "video_features.npy", mmap_mode="r", allow_pickle=False)
    sensor = np.load(cache_dir / "sensor_sequences.npy", mmap_mode="r", allow_pickle=False)
    labels = np.load(cache_dir / "labels.npy", mmap_mode="r", allow_pickle=False)
    if video.shape != (len(rows), 3, 8, 768) or sensor.shape != (len(rows), 8, 14):
        raise ValueError("frozen feature array shape mismatch")
    if (
        labels.shape != (len(rows),)
        or not np.isfinite(video).all()
        or not np.isfinite(sensor).all()
    ):
        raise ValueError("feature arrays must be complete and finite")
    for index, row in enumerate(rows):
        record = progress_by_index[index]
        target = 1 if row["target_option_id"] == "success" else 0
        if (
            record.get("id") != row["id"]
            or record.get("split") != row["split"]
            or int(labels[index]) != target
        ):
            raise ValueError("feature IDs/order/split/target differ from frozen task rows")
        if hashlib.sha256(np.asarray(video[index]).tobytes()).hexdigest() != record.get(
            "feature_sha256"
        ):
            raise ValueError("temporal VideoPrism feature row hash mismatch")
    return rows, video, sensor, labels


def _feature_conditions(video: np.ndarray, sensor: np.ndarray) -> dict[str, np.ndarray]:
    # Pool views but retain a predeclared early/late temporal distinction.
    pooled_views = np.asarray(video, dtype=np.float32).mean(axis=1)
    visual = np.stack(
        (pooled_views[:, :4].mean(axis=1), pooled_views[:, 4:].mean(axis=1)), axis=1
    ).reshape(len(video), -1)
    proprioception = np.asarray(sensor, dtype=np.float32).reshape(len(sensor), -1)
    return {
        "video_temporal": visual,
        "sensor_sequence": proprioception,
        "video_plus_sensor": np.concatenate((visual, proprioception), axis=1),
    }


def _paired_bootstrap(
    targets: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    seed: int = 1701,
    replicates: int = 2000,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    by_class = [np.flatnonzero(targets == label) for label in (0, 1)]
    deltas: dict[str, list[float]] = {
        key: [] for key in ("accuracy", "balanced_accuracy", "nll", "brier", "ece")
    }
    for _ in range(replicates):
        indexes = np.concatenate(
            [
                rng.choice(class_indexes, len(class_indexes), replace=True)
                for class_indexes in by_class
            ]
        )
        left = binary_probability_metrics(targets[indexes], candidate[indexes])
        right = binary_probability_metrics(targets[indexes], reference[indexes])
        for key in deltas:
            deltas[key].append(left[key] - right[key])
    return {
        key: {
            "mean_delta": float(np.mean(values)),
            "ci95_percentile": [
                float(np.quantile(values, 0.025)),
                float(np.quantile(values, 0.975)),
            ],
        }
        for key, values in deltas.items()
    }


def run(cache_dir: Path, task_dir: Path, checkpoint: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite readout artifacts: {output_dir}")
    rows, video, sensor, labels = _validate_inputs(cache_dir, task_dir, checkpoint)
    output_dir.mkdir(parents=True)
    split_train = np.asarray([row["split"] == "train" for row in rows])
    split_validation = ~split_train
    y = np.asarray(labels, dtype=np.int64)
    conditions = _feature_conditions(video, sensor)
    reports: dict[str, Any] = {}
    validation_probabilities: dict[str, np.ndarray] = {}
    readout_seconds: dict[str, float] = {}
    started = time.perf_counter()

    for name, features in conditions.items():
        fit_started = time.perf_counter()
        model = fit_l2_logistic(features[split_train], y[split_train])
        readout_seconds[name] = time.perf_counter() - fit_started
        train_probabilities = predict_success_probability(model, features[split_train])
        validation = predict_success_probability(model, features[split_validation])
        validation_probabilities[name] = validation
        reports[name] = {
            "input_dimensions": int(features.shape[1]),
            "train": binary_probability_metrics(y[split_train], train_probabilities),
            "validation": binary_probability_metrics(y[split_validation], validation),
            "optimizer": {
                "objective": model["objective"],
                "iterations": model["iterations"],
                "converged": model["converged"],
                "regularization": "mean binary cross entropy + 0.5 * ||w||^2 / n",
            },
        }
        model_path = output_dir / f"{name}-readout.npz"
        np.savez(model_path, **model)
        with np.load(model_path, allow_pickle=False) as saved:
            reloaded = {key: saved[key] for key in ("mean", "scale", "weight", "bias")}
        reloaded_probabilities = predict_success_probability(reloaded, features[split_validation])
        if not np.array_equal(validation, reloaded_probabilities):
            raise ValueError("save/reload changed validation probabilities")
        prediction_path = output_dir / f"{name}-validation-predictions.jsonl"
        with prediction_path.open("x", encoding="utf-8") as stream:
            for row, target, probability in zip(
                (rows[index] for index in np.flatnonzero(split_validation)),
                y[split_validation],
                validation,
                strict=True,
            ):
                stream.write(
                    json.dumps(
                        {
                            "id": row["id"],
                            "group_id": row["group_id"],
                            "target": int(target),
                            "prediction": int(probability >= 0.5),
                            "p_success": float(probability),
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )

    validation_y = y[split_validation]
    train_prevalence = float(y[split_train].mean())
    majority = np.full(validation_y.shape, train_prevalence)
    comparisons = {name: report["validation"] for name, report in reports.items()}
    comparisons["train_prevalence_baseline"] = binary_probability_metrics(validation_y, majority)
    deltas = {
        "video_plus_sensor_minus_video": _paired_bootstrap(
            validation_y,
            validation_probabilities["video_plus_sensor"],
            validation_probabilities["video_temporal"],
        ),
        "video_plus_sensor_minus_sensor": _paired_bootstrap(
            validation_y,
            validation_probabilities["video_plus_sensor"],
            validation_probabilities["sensor_sequence"],
        ),
        "video_plus_sensor_minus_train_prevalence_baseline": _paired_bootstrap(
            validation_y, validation_probabilities["video_plus_sensor"], majority
        ),
    }
    summary = json.loads((cache_dir / "summary.json").read_text(encoding="utf-8"))
    result = {
        "status": "PASS_FIXED_VALIDATION_FROZEN_FEATURE_READOUTS",
        "task": (
            "early DROID source-folder outcome prediction; weak metadata labels, "
            "not human task-quality annotation"
        ),
        "claim_scope": (
            "one development task, 337 train and 200 validation episodes; "
            "no sealed audit or validation model selection"
        ),
        "split_counts": {
            "train": int(split_train.sum()),
            "validation": int(split_validation.sum()),
            "validation_failure": int((validation_y == 0).sum()),
            "validation_success": int((validation_y == 1).sum()),
        },
        "input_features": {
            "video": (
                "three views pooled; separate early and late four-frame means; 1536 dimensions"
            ),
            "sensor": "eight aligned steps x 14 fields; flattened to 112 dimensions",
            "fusion": "concatenation of fixed video and sensor features; no trainable fusion gate",
        },
        "conditions": reports,
        "training_seconds_by_readout": readout_seconds,
        "total_evaluation_seconds": time.perf_counter() - started,
        "comparisons": comparisons,
        "paired_validation_deltas": {
            "bootstrap_replicates": 2000,
            "bootstrap_seed": 1701,
            "bootstrap_method": (
                "class-stratified paired validation-example resampling; one episode per group"
            ),
            "results": deltas,
        },
        "provenance": {
            "checkpoint_sha256": sha256_file(checkpoint),
            "task_manifest_sha256": sha256_file(task_dir / "task-manifest.json"),
            "shard_manifest_sha256": summary["shard_manifest_sha256"],
            "video_features_sha256": sha256_file(cache_dir / "video_features.npy"),
            "sensor_sequences_sha256": sha256_file(cache_dir / "sensor_sequences.npy"),
            "labels_sha256": sha256_file(cache_dir / "labels.npy"),
            "feature_progress_sha256": sha256_file(cache_dir / "progress.jsonl"),
        },
        "sealed_audit_loaded": False,
        "encoder_training": False,
        "decision_readout_training": True,
    }
    (output_dir / "evaluation-report.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.cache_dir, args.task_dir, args.checkpoint, args.output_dir),
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
