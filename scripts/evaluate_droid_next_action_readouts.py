"""Fit and evaluate fixed DROID video, sensor, and joint next-action readouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from scripts.audit_droid100_groups import (
    episode_example_context,
    episode_example_float_values,
    iter_tfrecord,
)
from scripts.build_droid_group_split import _canonical_path
from scripts.download_droid_selected_shards import validate_manifest, verify_metadata
from tiny_omni_decision.droid_next_action_readout import (
    ACTION_OPTIONS,
    fit_l2_softmax,
    make_readout_features,
    multiclass_probability_metrics,
    next_cartesian_direction_label,
    predict_softmax,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "selection_manifest": "2bfb5626d203344bf2bd976f711c0ed5f7b4c1979932b7b48565329099207e64",
    "download_receipt": "2bfb6322e9c0c4649f875d8ad37fb3e990f786324fe2fdbd9cd6b2872af50667",
    "task_manifest": "b882b66f8be2dd38aa7d69badd3d00b93f58769a9daa2be5f38088fbe084ec80",
    "checkpoint": "61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29",
}
STOP_THRESHOLD = 1e-4
VIDEO_WIDTH = 2 * 768


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, expected_sha256: str | None = None) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"SHA-256 mismatch for {path}")
    return json.loads(raw), digest


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-receipt", type=Path, required=True)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPOSITORY_ROOT
        / "manifests/pretrained_reuse/droid-selected-shards-seed17-20261026.json",
    )
    return parser.parse_args()


def _task_rows(task: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    groups: dict[str, set[str]] = {"train": set(), "validation": set()}
    for split in ("train", "validation"):
        for row in task["splits"][split]["rows"]:
            if row.get("split") != split or row["id"] in rows:
                raise ValueError("task manifest contains a split mismatch or duplicate ID")
            rows[row["id"]] = row
            groups[split].add(row["group_id"])
    if groups["train"] & groups["validation"]:
        raise ValueError("recording-group overlap exists across train and validation")
    return rows


def _build_targets(
    dataset_root: Path,
    receipt_path: Path,
    task_dir: Path,
    manifest_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    manifest_raw = manifest_path.read_bytes()
    manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
    if manifest_sha != EXPECTED["selection_manifest"]:
        raise ValueError("selected-shard manifest differs from the fixed DROID protocol")
    manifest = json.loads(manifest_raw)
    validate_manifest(manifest)
    receipt, receipt_sha = _load_json(receipt_path, EXPECTED["download_receipt"])
    if receipt.get("manifest_sha256") != manifest_sha:
        raise ValueError("download receipt references another selected-shard manifest")
    if receipt.get("episode_count") != manifest.get("expected_episode_count"):
        raise ValueError("download receipt episode count differs from the source manifest")
    verify_metadata(dataset_root, manifest)
    task, task_sha = _load_json(task_dir / "task-manifest.json", EXPECTED["task_manifest"])
    rows_by_id = _task_rows(task)
    targets: list[dict[str, Any]] = []
    seen: set[str] = set()
    for shard in manifest["selected_shards"]:
        shard_path = dataset_root / shard["object_name"]
        count = 0
        for record in iter_tfrecord(shard_path):
            count += 1
            context = episode_example_context(record)
            episode_id = hashlib.sha256(
                _canonical_path(context["episode_metadata/file_path"]).encode("utf-8")
            ).hexdigest()
            row = rows_by_id.get(episode_id)
            if row is None:
                continue
            if episode_id in seen:
                raise ValueError("one selected episode appears in multiple source records")
            seen.add(episode_id)
            observation_indices = [int(index) for index in row["observation_indices"]]
            if len(observation_indices) != 8 or observation_indices != sorted(observation_indices):
                raise ValueError("DROID input indices are not the fixed eight-step ordered window")
            last_observation = observation_indices[-1]
            target_index = last_observation + 1
            source_steps = int(row["source_step_count"])
            if last_observation >= int(row["observed_step_count"]) or target_index >= source_steps:
                raise ValueError("next action target is outside the source episode")
            values = episode_example_float_values(record, "steps/action_dict/cartesian_velocity")
            if len(values) != source_steps * 6:
                raise ValueError("commanded Cartesian velocity has unexpected width or length")
            velocity = np.asarray(values[target_index * 6 : (target_index + 1) * 6])
            label = next_cartesian_direction_label(velocity, STOP_THRESHOLD)
            targets.append(
                {
                    "id": episode_id,
                    "group_id": row["group_id"],
                    "split": row["split"],
                    "input_indices": observation_indices,
                    "target_index": target_index,
                    "target_option_id": ACTION_OPTIONS[label],
                    "source_outcome_option_id": row["target_option_id"],
                    "source_step_count": source_steps,
                }
            )
        if count != int(shard["episodes"]):
            raise ValueError("selected TFRecord episode count differs from its pinned manifest")
    if seen != set(rows_by_id):
        raise ValueError("one or more fixed task IDs are missing from selected source records")
    targets.sort(key=lambda item: (item["split"], item["id"]))
    provenance = {
        "selection_manifest_sha256": manifest_sha,
        "download_receipt_sha256": receipt_sha,
        "task_manifest_sha256": task_sha,
    }
    return targets, provenance


def _load_features(
    feature_cache: Path, checkpoint: Path, rows: list[dict[str, Any]], provenance: dict[str, str]
) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    summary, _ = _load_json(feature_cache / "summary.json")
    if summary.get("status") != "PASS" or summary.get("count") != len(rows):
        raise ValueError("complete DROID frozen-feature cache is required")
    if summary.get("task_manifest_sha256") != provenance["task_manifest_sha256"]:
        raise ValueError("frozen feature cache references another task manifest")
    if summary.get("shard_manifest_sha256") != provenance["selection_manifest_sha256"]:
        raise ValueError("frozen feature cache references another shard manifest")
    checkpoint_sha = sha256_file(checkpoint)
    if (
        checkpoint_sha != EXPECTED["checkpoint"]
        or summary.get("checkpoint_sha256") != checkpoint_sha
    ):
        raise ValueError("frozen VideoPrism checkpoint identity mismatch")
    expected_artifacts = {
        "video_features.npy": "video_features.npy_sha256",
        "sensor_sequences.npy": "sensor_sequences.npy_sha256",
        "labels.npy": "labels.npy_sha256",
        "progress.jsonl": "progress.jsonl_sha256",
    }
    artifact_hashes = {}
    for filename, summary_key in expected_artifacts.items():
        digest = sha256_file(feature_cache / filename)
        if digest != summary.get(summary_key):
            raise ValueError(f"frozen feature artifact hash mismatch: {filename}")
        artifact_hashes[filename] = digest
    if (
        summary.get("sealed_audit_loaded") is not False
        or summary.get("model_training") is not False
    ):
        raise ValueError("feature cache was not an inference-only, sealed-audit-free extraction")
    video = np.load(feature_cache / "video_features.npy", mmap_mode="r")
    sensor = np.load(feature_cache / "sensor_sequences.npy", mmap_mode="r")
    if video.shape != (len(rows), 3, 8, 768) or sensor.shape != (len(rows), 8, 14):
        raise ValueError("frozen feature cache has unexpected video or sensor shape")
    progress = [
        json.loads(line)
        for line in (feature_cache / "progress.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(progress) != len(rows):
        raise ValueError("feature progress row count differs from the pinned task")
    index_by_id: dict[str, int] = {}
    seen_indices: set[int] = set()
    target_by_id = {row["id"]: row for row in rows}
    original_labels = np.load(feature_cache / "labels.npy", mmap_mode="r")
    if original_labels.shape != (len(rows),):
        raise ValueError("legacy task label array has an unexpected shape")
    for item in progress:
        sample_id = str(item["id"])
        index = int(item["index"])
        if sample_id in index_by_id or index in seen_indices or not 0 <= index < len(rows):
            raise ValueError("feature progress contains duplicate or invalid IDs/indices")
        target_row = target_by_id.get(sample_id)
        if target_row is None:
            raise ValueError("feature progress references an unknown task ID")
        expected_legacy_label = int(target_row["source_outcome_option_id"] == "success")
        if (
            target_row["split"] != item.get("split")
            or int(item.get("label")) != expected_legacy_label
            or int(original_labels[index]) != expected_legacy_label
        ):
            raise ValueError("feature cache ID, split, and source-task label do not align")
        index_by_id[sample_id] = index
        seen_indices.add(index)
    if set(index_by_id) != {row["id"] for row in rows}:
        raise ValueError("feature cache IDs differ from the fixed task IDs")
    ordered_video = np.asarray([video[index_by_id[row["id"]]] for row in rows], dtype=np.float32)
    ordered_sensor = np.asarray(
        [sensor[index_by_id[row["id"]]] for row in rows], dtype=np.float32
    )
    artifact_hashes["checkpoint_sha256"] = checkpoint_sha
    return ordered_video, ordered_sensor, artifact_hashes


def _macro_bootstrap(
    targets: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    repetitions: int = 5000,
    seed: int = 17,
) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    n = len(targets)
    deltas_accuracy = np.empty(repetitions)
    deltas_nll = np.empty(repetitions)
    for index in range(repetitions):
        sample = rng.integers(0, n, size=n)
        deltas_accuracy[index] = (first[sample].argmax(1) == targets[sample]).mean() - (
            second[sample].argmax(1) == targets[sample]
        ).mean()
        truth = targets[sample]
        deltas_nll[index] = -np.log(
            np.maximum(first[sample, truth], 1e-15)
        ).mean() + np.log(np.maximum(second[sample, truth], 1e-15)).mean()
    return {
        "accuracy_delta_95_percentile": [
            float(np.quantile(deltas_accuracy, 0.025)),
            float(np.quantile(deltas_accuracy, 0.975)),
        ],
        "nll_delta_95_percentile": [
            float(np.quantile(deltas_nll, 0.025)),
            float(np.quantile(deltas_nll, 0.975)),
        ],
    }


def _standardize(values: np.ndarray, model: dict[str, Any]) -> np.ndarray:
    return np.clip((values - model["mean"]) / model["scale"], -20, 20)


def main() -> None:
    args = _arguments()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output}")
    if output.is_relative_to(REPOSITORY_ROOT.resolve()):
        raise ValueError("experiment artifacts must stay outside the repository")
    started = time.perf_counter()
    targets_rows, provenance = _build_targets(
        args.dataset_root.resolve(strict=True),
        args.dataset_receipt.resolve(strict=True),
        args.task_dir.resolve(strict=True),
        args.manifest.resolve(strict=True),
    )
    video, sensor, artifact_hashes = _load_features(
        args.feature_cache.resolve(strict=True),
        args.checkpoint.resolve(strict=True),
        targets_rows,
        provenance,
    )
    classes = {name: index for index, name in enumerate(ACTION_OPTIONS)}
    labels = np.asarray([classes[row["target_option_id"]] for row in targets_rows], dtype=np.int64)
    splits = np.asarray([row["split"] for row in targets_rows])
    train = np.flatnonzero(splits == "train")
    validation = np.flatnonzero(splits == "validation")
    if len(train) != 337 or len(validation) != 200:
        raise ValueError("DROID task rows differ from the frozen 337/200 development split")
    groups = {
        split: {row["group_id"] for row in targets_rows if row["split"] == split}
        for split in ("train", "validation")
    }
    if groups["train"] & groups["validation"]:
        raise ValueError("recording-group overlap detected")
    counts = np.bincount(labels[train], minlength=len(ACTION_OPTIONS))
    if np.any(counts == 0):
        raise ValueError("a fixed action class is absent from training")

    features = make_readout_features(video, sensor)
    models: dict[str, dict[str, Any]] = {}
    train_metrics: dict[str, dict[str, Any]] = {}
    validation_metrics: dict[str, dict[str, Any]] = {}
    probabilities: dict[str, np.ndarray] = {}
    for arm in ("video", "sensor", "video_sensor"):
        model = fit_l2_softmax(features[arm][train], labels[train])
        models[arm] = model
        train_metrics[arm] = multiclass_probability_metrics(
            labels[train], predict_softmax(model, features[arm][train])
        )
        probabilities[arm] = predict_softmax(model, features[arm][validation])
        validation_metrics[arm] = multiclass_probability_metrics(
            labels[validation], probabilities[arm]
        )

    prior = counts / counts.sum()
    prior_probabilities = np.repeat(prior[None, :], len(validation), axis=0)
    validation_metrics["train_prior"] = multiclass_probability_metrics(
        labels[validation], prior_probabilities
    )

    fused_model = models["video_sensor"]
    video_width = features["video"].shape[1]
    controls: dict[str, np.ndarray] = {}
    controls["video_only"] = probabilities["video"]
    controls["sensor_only"] = probabilities["sensor"]

    missing_video = features["video_sensor"][validation].copy()
    missing_video[:, :video_width] = fused_model["mean"][:video_width]
    controls["fused_video_missing"] = predict_softmax(fused_model, missing_video)
    missing_sensor = features["video_sensor"][validation].copy()
    missing_sensor[:, video_width:] = fused_model["mean"][video_width:]
    controls["fused_sensor_missing"] = predict_softmax(fused_model, missing_sensor)

    def fused_with(video_values: np.ndarray, sensor_values: np.ndarray) -> np.ndarray:
        raw_video = make_readout_features(video_values, sensor[validation])["video"]
        raw_sensor = make_readout_features(video[validation], sensor_values)["sensor"]
        raw = np.concatenate((raw_video, raw_sensor), axis=1)
        return predict_softmax(fused_model, _standardize(raw, fused_model))

    # The same fixed cyclic derangement is used for both streams, with IDs sorted above.
    controls["fused_video_deranged"] = fused_with(
        np.roll(video[validation], shift=1, axis=0), sensor[validation]
    )
    controls["fused_sensor_deranged"] = fused_with(
        video[validation], np.roll(sensor[validation], shift=1, axis=0)
    )
    stale_sensor = np.concatenate((sensor[validation, :1], sensor[validation, :-1]), axis=1)
    advanced_sensor = np.concatenate((sensor[validation, 1:], sensor[validation, -1:]), axis=1)
    controls["fused_sensor_one_sample_stale"] = fused_with(video[validation], stale_sensor)
    controls["fused_sensor_one_sample_advanced"] = fused_with(video[validation], advanced_sensor)

    # A deterministic donor from a different target class creates a label-conflict proxy.
    val_labels = labels[validation]
    donor_indices = np.empty(len(validation), dtype=np.int64)
    by_class = {k: np.flatnonzero(val_labels == k) for k in range(len(ACTION_OPTIONS))}
    offsets = {k: 0 for k in range(len(ACTION_OPTIONS))}
    for index, label in enumerate(val_labels):
        donor_class = next(
            candidate
            for step in range(1, len(ACTION_OPTIONS))
            if len(by_class[(candidate := (int(label) + step) % len(ACTION_OPTIONS))])
        )
        pool = by_class[donor_class]
        donor_indices[index] = pool[offsets[donor_class] % len(pool)]
        offsets[donor_class] += 1
    controls["fused_sensor_different_target_class"] = fused_with(
        video[validation], sensor[validation][donor_indices]
    )
    control_metrics = {
        name: multiclass_probability_metrics(labels[validation], value)
        for name, value in controls.items()
    }
    paired = {
        name: _macro_bootstrap(labels[validation], probabilities["video_sensor"], value)
        for name, value in (
            ("video", probabilities["video"]),
            ("sensor", probabilities["sensor"]),
        )
    }

    output.mkdir(parents=True, exist_ok=False)
    labels_path = output / "target-manifest.jsonl"
    labels_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in targets_rows),
        encoding="utf-8",
    )
    prediction_path = output / "validation-predictions.jsonl"
    prediction_rows = []
    for offset, task_index in enumerate(validation):
        row = targets_rows[int(task_index)]
        prediction_rows.append(
            {
                "id": row["id"],
                "group_id": row["group_id"],
                "target_option_id": row["target_option_id"],
                "probabilities": {
                    arm: probabilities[arm][offset].tolist()
                    for arm in ("video", "sensor", "video_sensor")
                },
                "predictions": {
                    arm: ACTION_OPTIONS[int(probabilities[arm][offset].argmax())]
                    for arm in ("video", "sensor", "video_sensor")
                },
            }
        )
    prediction_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in prediction_rows),
        encoding="utf-8",
    )
    for arm, model in models.items():
        np.savez_compressed(
            output / f"{arm}-readout.npz",
            mean=model["mean"],
            scale=model["scale"],
            weight=model["weight"],
            bias=model["bias"],
        )
    report = {
        "schema_version": 1,
        "experiment_id": "droid-next-cartesian-direction-video-sensor-v0",
        "claim_scope": (
            "Development-only one-step robot action-direction readout; "
            "not product quality or R2 M-2 closure."
        ),
        "status": "PASS_EVALUATED",
        "input": {
            "modalities": ["video", "sensor"],
            "video_checkpoint": "google/videoprism-base-f16r288",
            "video_revision": "be719a406d563b66f0ac969e7c94bab8e997c81a",
            "video_parameters": 114365184,
            "video_frames": 8,
            "sensor_shape_per_episode": [8, 14],
            "frozen_feature_cache": str(args.feature_cache.resolve()),
            "feature_cache_hashes": artifact_hashes,
        },
        "task": {
            "target": (
                "Dominant signed xyz axis of commanded Cartesian velocity at the step "
                "immediately after the last observed state; stop if max |xyz| <= 1e-4."
            ),
            "options": list(ACTION_OPTIONS),
            "input_indices": (
                "Eight fixed early episode indices; target action is max(input_indices)+1."
            ),
            "group_split": "fixed recording-folder grouped task split",
            "train_count": len(train),
            "validation_count": len(validation),
            "train_group_count": len(groups["train"]),
            "validation_group_count": len(groups["validation"]),
            "group_overlap": 0,
            "train_label_counts": {
                name: int((labels[train] == index).sum())
                for index, name in enumerate(ACTION_OPTIONS)
            },
            "validation_label_counts": {
                name: int((labels[validation] == index).sum())
                for index, name in enumerate(ACTION_OPTIONS)
            },
            "sealed_audit_loaded": False,
        },
        "method": {
            "readout": "deterministic multinomial L2 logistic regression",
            "regularization": "0.5 * ||W||^2 / n_train",
            "standardization": "training rows only",
            "feature_layout": (
                "Video mean across 3 views then separate first/last 4 sampled frame means; "
                "sensor 8x14 flattened; fused concatenation."
            ),
            "validation_used_for_selection": False,
            "optimizer_iterations": {arm: model["iterations"] for arm, model in models.items()},
            "trainable_parameters": {
                arm: int(model["weight"].size + model["bias"].size)
                for arm, model in models.items()
            },
        },
        "metrics": {"train": train_metrics, "validation": validation_metrics},
        "paired_validation_bootstrap": {
            "method": (
                "5,000 paired episode resamples, seed 17, percentile 95% interval; "
                "positive accuracy delta and negative NLL delta favor fusion."
            ),
            "video_sensor_minus_video": paired["video"],
            "video_sensor_minus_sensor": paired["sensor"],
        },
        "validation_controls": {
            "metrics": control_metrics,
            "note": (
                "Controls are sampled-index perturbations, not timestamp-calibrated shifts. "
                "Different-target-class sensor donors are a conflict proxy, not an "
                "externally labeled contradiction set."
            ),
        },
        "resources": {
            "compute": "CPU only; no GPU or cloud compute used",
            "elapsed_seconds": time.perf_counter() - started,
            "model_training": (
                "Frozen encoders; three small CPU readouts fitted once with fixed L2 objective."
            ),
        },
        "provenance": {
            **provenance,
            "checkpoint_sha256": artifact_hashes["checkpoint_sha256"],
            "target_manifest_sha256": sha256_file(labels_path),
            "validation_predictions_sha256": sha256_file(prediction_path),
            "code_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True
            ).strip(),
            "code_worktree_clean": not subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=REPOSITORY_ROOT, text=True
            ).strip(),
            "code_sha256": {
                "evaluator": sha256_file(Path(__file__).resolve()),
                "readout": sha256_file(
                    REPOSITORY_ROOT / "src/tiny_omni_decision/droid_next_action_readout.py"
                ),
            },
        },
        "decision": (
            "This is a video+sensor next-action diagnostic only. Require fusion to "
            "outperform both unimodal controls with input sensitivity and adequate "
            "independent data before describing the task as both-input-required; this "
            "run does not close R2 M-2."
        ),
    }
    report_path = output / "evaluation-report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
