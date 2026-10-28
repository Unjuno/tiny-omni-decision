"""Fit fixed frozen-feature readouts on train and evaluate once on fresh DROID validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from tiny_omni_decision.droid_next_action_readout import (  # noqa: E402
    ACTION_OPTIONS,
    fit_l2_softmax,
    make_readout_features,
    multiclass_probability_metrics,
    predict_softmax,
    select_l2_strength_group_cv,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def paired_site_bootstrap(
    targets: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    sites: list[str],
    *,
    repetitions: int = 5000,
    seed: int = 17,
) -> dict[str, Any]:
    if len(sites) != len(targets) or repetitions < 1:
        raise ValueError("site bootstrap inputs are misaligned")
    sites_array = np.asarray(sites, dtype=str)
    unique = np.unique(sites_array)
    if len(unique) < 2:
        raise ValueError("site bootstrap requires at least two independent sites")
    rows_by_site = {site: np.flatnonzero(sites_array == site) for site in unique}
    rng = np.random.default_rng(seed)
    accuracy_delta = np.empty(repetitions)
    nll_delta = np.empty(repetitions)
    for index in range(repetitions):
        sampled_sites = rng.choice(unique, size=len(unique), replace=True)
        sample = np.concatenate([rows_by_site[site] for site in sampled_sites])
        y = targets[sample]
        p_first, p_second = first[sample], second[sample]
        accuracy_delta[index] = (p_first.argmax(axis=1) == y).mean() - (
            p_second.argmax(axis=1) == y
        ).mean()
        first_nll = -np.log(np.maximum(p_first[np.arange(len(y)), y], 1e-15)).mean()
        second_nll = -np.log(np.maximum(p_second[np.arange(len(y)), y], 1e-15)).mean()
        nll_delta[index] = first_nll - second_nll
    return {
        "unit": "collection_site",
        "site_count": int(len(unique)),
        "repetitions": repetitions,
        "seed": seed,
        "accuracy_delta_first_minus_second_95_percentile": [
            float(np.quantile(accuracy_delta, 0.025)),
            float(np.quantile(accuracy_delta, 0.975)),
        ],
        "nll_delta_first_minus_second_95_percentile": [
            float(np.quantile(nll_delta, 0.025)),
            float(np.quantile(nll_delta, 0.975)),
        ],
    }


def equal_weighted_site_metrics(
    per_site_metrics: dict[str, dict[str, Any]],
    arms: tuple[str, ...] = ("video", "sensor", "video_sensor"),
) -> dict[str, dict[str, float]]:
    """Average validation metrics with each held-out site receiving equal weight."""
    if not per_site_metrics:
        raise ValueError("at least one site is required")
    metric_keys = (
        "accuracy",
        "macro_accuracy",
        "nll",
        "macro_nll",
        "brier",
        "ece_10",
    )
    result: dict[str, dict[str, float]] = {}
    for arm in arms:
        site_rows = [
            site["metrics"][arm] for site in per_site_metrics.values()
        ]
        result[arm] = {
            key: float(np.mean([metric[key] for metric in site_rows]))
            for key in metric_keys
        }
    return result


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-manifest", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--readout-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite evaluation output: {output}")
    if output.is_relative_to(REPOSITORY_ROOT.resolve()):
        raise ValueError("evaluation output must stay outside the repository")
    started = time.perf_counter()
    task_raw = args.task_manifest.read_bytes()
    task = json.loads(task_raw)
    config_raw = args.config.read_bytes()
    config = json.loads(config_raw)
    readout_config_raw = args.readout_config.read_bytes()
    readout_config = json.loads(readout_config_raw)
    rows = [
        row
        for split in ("train", "validation")
        for row in task["splits"][split]["rows"]
    ]
    if not rows or task.get("sealed_audit_loaded") is not False:
        raise ValueError("task is empty or includes a sealed audit")
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("task contains duplicate episode IDs")
    groups_by_split = {
        split: {row["group_id"] for row in rows if row["split"] == split}
        for split in ("train", "validation")
    }
    if groups_by_split["train"] & groups_by_split["validation"]:
        raise ValueError("train/validation recording groups overlap")
    feature_cache = args.feature_cache.resolve(strict=True)
    summary = json.loads((feature_cache / "summary.json").read_text(encoding="utf-8"))
    checkpoint_hash = sha256_file(args.checkpoint.resolve(strict=True))
    if (
        summary.get("status") != "PASS"
        or summary.get("count") != len(rows)
        or summary.get("task_manifest_sha256") != hashlib.sha256(task_raw).hexdigest()
        or summary.get("checkpoint_sha256") != checkpoint_hash
        or summary.get("experiment_config_sha256") != hashlib.sha256(config_raw).hexdigest()
        or config["representation"]["video_checkpoint_sha256"] != checkpoint_hash
        or summary.get("sealed_audit_loaded") is not False
        or summary.get("model_training") is not False
    ):
        raise ValueError("feature cache metadata does not match the frozen task/checkpoint")
    if (
        readout_config.get("feature_extraction_config_sha256")
        != summary.get("experiment_config_sha256")
        or readout_config.get("task_manifest_sha256") != hashlib.sha256(task_raw).hexdigest()
        or readout_config.get("sealed_audit_loaded") is not False
    ):
        raise ValueError("readout fit config does not match frozen features or task")
    max_iterations = int(readout_config["readout"]["max_iterations"])
    l2_candidates = tuple(
        float(value) for value in readout_config["readout"]["l2_candidates"]
    )
    if max_iterations < 500:
        raise ValueError("fit retry must not reduce the original optimizer iteration cap")
    if not l2_candidates or any(value <= 0 or not np.isfinite(value) for value in l2_candidates):
        raise ValueError("readout L2 candidates must be positive finite values")
    for filename, key in (
        ("video_features.npy", "video_features.npy_sha256"),
        ("sensor_sequences.npy", "sensor_sequences.npy_sha256"),
        ("labels.npy", "labels.npy_sha256"),
        ("progress.jsonl", "progress.jsonl_sha256"),
    ):
        if sha256_file(feature_cache / filename) != summary.get(key):
            raise ValueError(f"feature cache hash mismatch: {filename}")
    video = np.load(feature_cache / "video_features.npy", mmap_mode="r")
    sensor = np.load(feature_cache / "sensor_sequences.npy", mmap_mode="r")
    labels_array = np.load(feature_cache / "labels.npy", mmap_mode="r")
    if video.shape != (len(rows), 3, 8, 768) or sensor.shape != (len(rows), 8, 14):
        raise ValueError("frozen feature dimensions are invalid")
    classes = {name: index for index, name in enumerate(ACTION_OPTIONS)}
    labels = np.asarray([classes[row["target_option_id"]] for row in rows], dtype=np.int64)
    progress = [
        json.loads(line)
        for line in (feature_cache / "progress.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    if len(progress) != len(rows):
        raise ValueError("feature progress count differs from task rows")
    seen_ids: set[str] = set()
    seen_indices: set[int] = set()
    for item in progress:
        index = int(item["index"])
        if not 0 <= index < len(rows) or index in seen_indices or item["id"] in seen_ids:
            raise ValueError("feature progress contains duplicate/invalid identities")
        row = rows[index]
        expected_label = classes[row["target_option_id"]]
        if item["id"] != row["id"] or item["split"] != row["split"]:
            raise ValueError("feature cache IDs or split order differs from task rows")
        if int(item["label"]) != expected_label or int(labels_array[index]) != expected_label:
            raise ValueError("feature cache label differs from fixed task target")
        seen_ids.add(item["id"])
        seen_indices.add(index)
    if seen_ids != set(ids) or seen_indices != set(range(len(rows))):
        raise ValueError("feature cache does not cover each frozen task row exactly once")

    train = np.asarray([i for i, row in enumerate(rows) if row["split"] == "train"])
    validation = np.asarray([i for i, row in enumerate(rows) if row["split"] == "validation"])
    if not len(train) or not len(validation):
        raise ValueError("both train and validation must contain examples")
    train_counts = np.bincount(labels[train], minlength=len(ACTION_OPTIONS))
    if np.any(train_counts == 0):
        raise ValueError("a fixed next-action class is absent from train")
    features = make_readout_features(np.asarray(video), np.asarray(sensor))
    models: dict[str, dict[str, Any]] = {}
    train_metrics: dict[str, Any] = {}
    validation_metrics: dict[str, Any] = {}
    probabilities: dict[str, np.ndarray] = {}
    selection: dict[str, Any] = {}
    train_groups = [rows[int(i)]["group_id"] for i in train]
    for arm in ("video", "sensor", "video_sensor"):
        selection[arm] = select_l2_strength_group_cv(
            features[arm][train],
            labels[train],
            train_groups,
            candidates=l2_candidates,
            folds=int(readout_config["readout"]["folds"]),
            seed=int(readout_config["readout"]["seed"]),
            max_iterations=max_iterations,
        )
        model = fit_l2_softmax(
            features[arm][train],
            labels[train],
            l2_strength=selection[arm]["selected_l2_strength"],
            max_iterations=max_iterations,
        )
        models[arm] = model
        train_metrics[arm] = multiclass_probability_metrics(
            labels[train], predict_softmax(model, features[arm][train])
        )
        probabilities[arm] = predict_softmax(model, features[arm][validation])
        validation_metrics[arm] = multiclass_probability_metrics(
            labels[validation], probabilities[arm]
        )
    prior = train_counts / train_counts.sum()
    validation_metrics["train_prior"] = multiclass_probability_metrics(
        labels[validation], np.repeat(prior[None, :], len(validation), axis=0)
    )
    validation_groups = [rows[int(i)]["group_id"] for i in validation]
    validation_sites = [rows[int(i)]["site"] for i in validation]
    paired = {
        arm: paired_site_bootstrap(
            labels[validation], probabilities["video_sensor"], probabilities[arm], validation_sites
        )
        for arm in ("video", "sensor")
    }
    per_site_metrics: dict[str, dict[str, Any]] = {}
    site_macro_metrics: dict[str, dict[str, float]] = {}
    for site in sorted(set(validation_sites)):
        site_mask = np.asarray([value == site for value in validation_sites])
        per_site_metrics[site] = {
            "count": int(site_mask.sum()),
            "target_counts": {
                name: int((labels[validation][site_mask] == class_id).sum())
                for class_id, name in enumerate(ACTION_OPTIONS)
            },
            "metrics": {
                arm: multiclass_probability_metrics(
                    labels[validation][site_mask], probabilities[arm][site_mask]
                )
                for arm in ("video", "sensor", "video_sensor")
            },
        }
    site_macro_metrics = equal_weighted_site_metrics(per_site_metrics)

    # Keep each site's video marginal intact while breaking example-level alignment.
    rng = np.random.default_rng(17)
    shuffled_video = features["video"][validation].copy()
    for site in sorted(set(validation_sites)):
        site_indices = np.flatnonzero(np.asarray(validation_sites) == site)
        if len(site_indices) > 1:
            shuffled_video[site_indices] = shuffled_video[
                rng.permutation(site_indices)
            ]
    shuffled_joint = np.concatenate(
        (shuffled_video, features["sensor"][validation]), axis=1
    )
    validation_metrics["video_sensor_video_permuted_within_site"] = (
        multiclass_probability_metrics(
            labels[validation], predict_softmax(models["video_sensor"], shuffled_joint)
        )
    )

    output.mkdir(parents=True, exist_ok=False)
    predictions = []
    for offset, index in enumerate(validation):
        row = rows[int(index)]
        predictions.append(
            {
                "id": row["id"],
                "group_id": row["group_id"],
                "site": row["site"],
                "target_option_id": row["target_option_id"],
                "probabilities": {
                    arm: probabilities[arm][offset].tolist()
                    for arm in ("video", "sensor", "video_sensor")
                },
            }
        )
    prediction_raw = "".join(json.dumps(row, sort_keys=True) + "\n" for row in predictions).encode()
    (output / "validation-predictions.jsonl").write_bytes(prediction_raw)
    readout_hashes = {}
    for arm, model in models.items():
        readout_path = output / f"{arm}-readout.npz"
        np.savez_compressed(
            readout_path,
            mean=model["mean"],
            scale=model["scale"],
            weight=model["weight"],
            bias=model["bias"],
        )
        readout_hashes[readout_path.name] = sha256_file(readout_path)
    report = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "source_scope": (
            "fresh, pinned DROID official-train shards; "
            "nonblind development validation"
        ),
        "status": "PASS_EVALUATED",
        "task": {
            "options": list(ACTION_OPTIONS),
            "train_examples": len(train),
            "validation_examples": len(validation),
            "train_target_counts": {
                ACTION_OPTIONS[index]: int(train_counts[index])
                for index in range(len(ACTION_OPTIONS))
            },
            "validation_target_counts": {
                name: int(sum(rows[int(i)]["target_option_id"] == name for i in validation))
                for name in ACTION_OPTIONS
            },
            "unique_train_recording_groups": len(set(train_groups)),
            "unique_validation_recording_groups": len(set(validation_groups)),
            "recording_group_overlap": 0,
            "train_sites": task["splits"]["train"]["site_counts"],
            "validation_sites": task["splits"]["validation"]["site_counts"],
            "site_overlap": 0,
            "exact_input_frame_hash_overlap": 0,
        },
        "selection": {
            "readout": "L2 multinomial logistic; feature standardization on train only",
            "regularization": selection,
            "readout_fit_config_sha256": hashlib.sha256(readout_config_raw).hexdigest(),
            "checkpoint_selection": "not applicable; frozen encoders, train-only grouped CV for L2",
        },
        "metrics": {"train": train_metrics, "validation": validation_metrics},
        "paired_site_bootstrap": paired,
        "per_site_validation_metrics": per_site_metrics,
        "equal_weighted_site_validation_metrics": site_macro_metrics,
        "video_permutation_control": {
            "procedure": (
                "within-site fixed-seed row permutation of frozen video features; "
                "sensor input unchanged; model not refit"
            ),
            "seed": 17,
            "metrics": validation_metrics[
                "video_sensor_video_permuted_within_site"
            ],
        },
        "artifacts": {
            "experiment_config_sha256": hashlib.sha256(config_raw).hexdigest(),
            "readout_fit_config_sha256": hashlib.sha256(readout_config_raw).hexdigest(),
            "task_manifest_sha256": hashlib.sha256(task_raw).hexdigest(),
            "validation_predictions_sha256": hashlib.sha256(prediction_raw).hexdigest(),
            "readout_sha256": readout_hashes,
            "feature_cache_summary_sha256": sha256_file(feature_cache / "summary.json"),
        },
        "limitations": [
            "Validation was constructed from official train shards and is not a blind benchmark.",
            (
                "This task measures one-step robot Cartesian direction, "
                "not general multimodal Decision quality."
            ),
            (
                "The frozen VideoPrism encoder is 114.37M parameters and is not "
                "within the provisional product package target."
            ),
        ],
        "sealed_audit_loaded": False,
        "duration_seconds": time.perf_counter() - started,
    }
    report_raw = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode()
    (output / "evaluation-report.json").write_bytes(report_raw)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
