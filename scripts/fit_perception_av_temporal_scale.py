"""Fit frozen-feature AV readouts across nested 96/224-video training budgets."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.fit_perception_temporal_match_sequence_readouts import (
    EVALUATION_INTERVAL,
    HIDDEN_DIMENSION,
    LEARNING_RATE,
    SEED,
    TOTAL_UPDATES,
    WEIGHT_DECAY,
    AudioCandidateScorer,
    TemporalCandidateScorer,
    _arrays,
    _fit_arm,
    _predict,
    categorical_metrics,
    load_dataset,
    sha256_file,
)

VIDEO_CHECKPOINT_SHA256 = "61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29"
AUDIO_CHECKPOINT_SHA256 = "49b0e63577021994257b3de2ac5f1686d71554b93c8149f9d2568ad5760b904e"
VIDEO_SOURCE_COMMIT = "c6eba0ece3dd49a24ffc44582966997a4a59f1d8"
AUDIO_SOURCE_COMMIT = "a425fdce92572e602a1d5634799bd9f1f2efa806"


def make_training_cohorts(
    manifest: dict[str, Any],
    train_samples: list[dict[str, Any]],
    validation_samples: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sample_by_id: dict[str, dict[str, Any]] = {}
    train_video_ids = set()
    for row in train_samples:
        sample_id = str(row["sample_id"])
        video_id = str(row["video_id"])
        if sample_id in sample_by_id:
            raise ValueError("duplicate sample ID in the large training cohort")
        sample_by_id[sample_id] = row
        train_video_ids.add(video_id)
    if len(train_video_ids) != len(train_samples):
        raise ValueError("duplicate video group in the large training cohort")

    validation_video_ids = [str(row["video_id"]) for row in validation_samples]
    if len(validation_video_ids) != len(set(validation_video_ids)):
        raise ValueError("duplicate video group in validation")
    if train_video_ids & set(validation_video_ids):
        raise ValueError("train/validation video group overlap")

    small_manifest_rows = manifest.get("train_small")
    if not isinstance(small_manifest_rows, list) or not small_manifest_rows:
        raise ValueError("manifest must contain a nonempty nested train_small cohort")
    small_ids = [str(row["sample_id"]) for row in small_manifest_rows]
    if len(small_ids) != len(set(small_ids)):
        raise ValueError("duplicate sample ID in train_small")
    unknown = set(small_ids) - set(sample_by_id)
    if unknown:
        raise ValueError("train_small sample IDs must be a subset of train")
    small = [sample_by_id[sample_id] for sample_id in small_ids]
    if len({str(row["video_id"]) for row in small}) != len(small):
        raise ValueError("duplicate video group in train_small")
    if {str(row["video_id"]) for row in small} & set(validation_video_ids):
        raise ValueError("train_small/validation video group overlap")
    return small, train_samples


def _row_metrics(scores: np.ndarray, targets: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(targets, dtype=np.int64)
    shifted = values - values.max(axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    probabilities = exp_values / exp_values.sum(axis=1, keepdims=True)
    predicted = probabilities.argmax(axis=1)
    correct = (predicted == labels).astype(np.float64)
    nll = -np.log(np.maximum(probabilities[np.arange(labels.size), labels], 1e-15))
    one_hot = np.eye(values.shape[1], dtype=np.float64)[labels]
    brier = ((probabilities - one_hot) ** 2).sum(axis=1)
    return {"accuracy": correct, "nll": nll, "brier": brier}


def paired_metric_interval(
    reference_scores: np.ndarray,
    candidate_scores: np.ndarray,
    targets: np.ndarray,
    *,
    repetitions: int = 5_000,
    seed: int = SEED,
) -> dict[str, Any]:
    reference = _row_metrics(reference_scores, targets)
    candidate = _row_metrics(candidate_scores, targets)
    count = len(targets)
    if count == 0 or len(reference_scores) != count or len(candidate_scores) != count:
        raise ValueError("paired predictions must contain one row per video group")
    rng = np.random.default_rng(seed)
    resamples = rng.integers(0, count, size=(repetitions, count))
    metrics = {}
    for name in ("accuracy", "nll", "brier"):
        delta = candidate[name] - reference[name]
        bootstrap = delta[resamples].mean(axis=1)
        metrics[name] = {
            "delta_candidate_minus_reference": float(delta.mean()),
            "ci_95": [
                float(np.quantile(bootstrap, 0.025)),
                float(np.quantile(bootstrap, 0.975)),
            ],
        }
    return {
        "count": count,
        "bootstrap_repetitions": repetitions,
        "bootstrap_seed": seed,
        "resampling_unit": "video group; one row per video",
        "delta_accuracy_candidate_minus_reference": metrics["accuracy"][
            "delta_candidate_minus_reference"
        ],
        "metrics": metrics,
    }


def _normalization(samples: list[dict[str, Any]]) -> tuple[np.ndarray, ...]:
    video = np.stack([row["video"] for row in samples])
    audio = np.stack([row["audio"] for row in samples])
    video_mean = video.reshape(-1, video.shape[-1]).mean(axis=0)
    video_std = video.reshape(-1, video.shape[-1]).std(axis=0).clip(1e-5)
    audio_mean = audio.reshape(-1, audio.shape[-1]).mean(axis=0)
    audio_std = audio.reshape(-1, audio.shape[-1]).std(axis=0).clip(1e-5)
    return video_mean, video_std, audio_mean, audio_std


def _data_for_arm(arm_name, train, validation):
    if arm_name == "audio_only":
        return (None, train[1], train[2]), (None, validation[1], validation[2])
    return train, validation


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _set_effective_config_status(path: Path, status: str) -> None:
    config = json.loads(path.read_text(encoding="utf-8"))
    config["status"] = status
    _atomic_json(path, config)


def _control_metrics(model, video, audio, targets, reference_scores) -> dict[str, Any]:
    controls = {}
    for name, video_values, audio_values in (
        ("video_missing", torch.zeros_like(video), audio),
        ("audio_missing", video, torch.zeros_like(audio)),
        ("time_reversed", video.flip(dims=(1,)), audio),
    ):
        scores = _predict(model, video_values, audio_values)
        controls[name] = {
            "metrics": {
                key: value
                for key, value in categorical_metrics(scores, targets.numpy()).items()
                if key != "probabilities"
            },
            "paired_delta_vs_original": paired_metric_interval(
                reference_scores, scores, targets.numpy()
            ),
        }
    return controls


def run(args: argparse.Namespace) -> dict[str, Any]:
    torch.set_num_threads(1)
    manifest_path = args.manifest.resolve(strict=True)
    video_path = args.video_features.resolve(strict=True)
    audio_path = args.audio_features.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite scale-fit output: {output}")

    video_feature_result = json.loads(video_path.read_text(encoding="utf-8"))
    audio_feature_result = json.loads(audio_path.read_text(encoding="utf-8"))
    if video_feature_result.get("status") != "PASS_TEMPORAL_MATCH_VIDEOPRISM_FEATURES":
        raise ValueError("VideoPrism feature result is not a passing terminal result")
    if audio_feature_result.get("status") != "PASS_TEMPORAL_MATCH_EFFICIENTAT_FEATURES":
        raise ValueError("EfficientAT feature result is not a passing terminal result")
    if video_feature_result.get("checkpoint_sha256") != VIDEO_CHECKPOINT_SHA256:
        raise ValueError("VideoPrism checkpoint pin mismatch")
    if audio_feature_result.get("weights_sha256") != AUDIO_CHECKPOINT_SHA256:
        raise ValueError("EfficientAT checkpoint pin mismatch")
    if video_feature_result.get("source_commit") != VIDEO_SOURCE_COMMIT:
        raise ValueError("VideoPrism source revision mismatch")
    if audio_feature_result.get("source_commit") != AUDIO_SOURCE_COMMIT:
        raise ValueError("EfficientAT source revision mismatch")
    if video_feature_result.get("feature_layout") != "per_frame_mean":
        raise ValueError("scale fit requires eight ordered per-frame VideoPrism features")

    manifest, dataset = load_dataset(manifest_path, video_path, audio_path)
    small_rows, large_rows = make_training_cohorts(
        manifest, dataset["train"], dataset["validation"]
    )
    validation_rows = dataset["validation"]
    if (len(small_rows), len(large_rows), len(validation_rows)) != (96, 224, 64):
        raise ValueError("frozen scale protocol requires 96/224/64 groups")

    output.mkdir(parents=True)
    run_metadata = {
        "status": "RUNNING",
        "manifest_sha256": sha256_file(manifest_path),
        "video_feature_result_sha256": sha256_file(video_path),
        "audio_feature_result_sha256": sha256_file(audio_path),
        "video_source_commit": video_feature_result.get("source_commit"),
        "audio_source_commit": audio_feature_result.get("source_commit"),
        "video_checkpoint_sha256": VIDEO_CHECKPOINT_SHA256,
        "audio_checkpoint_sha256": AUDIO_CHECKPOINT_SHA256,
        "seed": SEED,
        "updates_per_arm": TOTAL_UPDATES,
        "evaluation_interval": EVALUATION_INTERVAL,
        "device": "CPU",
        "arms": 6,
    }
    _atomic_json(output / "run.json", run_metadata)
    (output / "effective_config.json").write_text(
        json.dumps(
            {
                **run_metadata,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "optimizer": "AdamW",
                "batch_mode": "full batch; each arm repeatedly consumes all its train groups",
                "hidden_dimension": HIDDEN_DIMENSION,
                "normalization": "train-only per-cohort mean/std; std floor=1e-5",
                "selection": "earliest minimum validation categorical NLL",
                "validation_is_blind": False,
                "validation_group_count": len(validation_rows),
                "small_train_group_count": len(small_rows),
                "large_train_group_count": len(large_rows),
                "video_source_commit": video_feature_result.get("source_commit"),
                "audio_source_commit": audio_feature_result.get("source_commit"),
                "video_checkpoint_sha256": VIDEO_CHECKPOINT_SHA256,
                "audio_checkpoint_sha256": AUDIO_CHECKPOINT_SHA256,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    raw_by_size = {"train96": small_rows, "train224": large_rows}
    arrays_by_size = {}
    for size_name, rows in raw_by_size.items():
        stats = _normalization(rows)
        np.savez_compressed(
            output / f"{size_name}-normalization.npz",
            video_mean=stats[0],
            video_std=stats[1],
            audio_mean=stats[2],
            audio_std=stats[3],
        )
        arrays_by_size[size_name] = (
            _arrays(rows, *stats),
            _arrays(validation_rows, *stats),
        )

    factories = (
        ("audio_only", lambda: AudioCandidateScorer()),
        ("pooled_av", lambda: TemporalCandidateScorer(pool_video=True)),
        ("sequence_av", lambda: TemporalCandidateScorer(pool_video=False)),
    )
    results: dict[str, Any] = {}
    history_path = output / "training_history.jsonl"
    started = time.perf_counter()
    try:
        for size_name in ("train96", "train224"):
            train, validation = arrays_by_size[size_name]
            rows = raw_by_size[size_name]
            for arm_name, factory in factories:
                name = f"{arm_name}_{size_name}"
                torch.manual_seed(SEED)
                model = factory()
                if any(parameter.device.type != "cpu" for parameter in model.parameters()):
                    raise RuntimeError("scale readout must remain CPU-only")
                arm_train, arm_validation = _data_for_arm(arm_name, train, validation)
                results[name] = _fit_arm(
                    name,
                    model,
                    arm_train,
                    arm_validation,
                    rows,
                    validation_rows,
                    output,
                    history_path,
                )

        targets = arrays_by_size["train224"][1][2].numpy()
        comparisons = {}
        for arm in ("audio_only", "pooled_av", "sequence_av"):
            comparisons[f"train224_minus_train96_{arm}"] = paired_metric_interval(
                results[f"{arm}_train96"]["validation_scores"],
                results[f"{arm}_train224"]["validation_scores"],
                targets,
            )
        for size_name in ("train96", "train224"):
            comparisons[f"sequence_minus_pooled_{size_name}"] = paired_metric_interval(
                results[f"pooled_av_{size_name}"]["validation_scores"],
                results[f"sequence_av_{size_name}"]["validation_scores"],
                targets,
            )

        controls = {}
        valid_video, valid_audio, valid_targets = arrays_by_size["train224"][1]
        for size_name in ("train96", "train224"):
            key = f"sequence_av_{size_name}"
            controls[key] = _control_metrics(
                results[key]["model"],
                valid_video,
                valid_audio,
                valid_targets,
                results[key]["validation_scores"],
            )

        predictions = []
        for index, row in enumerate(validation_rows):
            predictions.append(
                {
                    "sample_id": row["sample_id"],
                    "video_id": row["video_id"],
                    "target_index": row["target"],
                    "arms": {
                        name: {
                            "scores": result["validation_scores"][index].tolist(),
                            "probabilities": categorical_metrics(
                                result["validation_scores"][index : index + 1],
                                np.asarray([row["target"]]),
                            )["probabilities"][0].tolist(),
                        }
                        for name, result in results.items()
                    },
                }
            )
        _atomic_json(output / "validation_predictions.json", {"rows": predictions})

        curves = [
            json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines()
        ]
        summary = {
            "status": "PASS_FRESH_AV_TEMPORAL_SCALE_FIT",
            "manifest_sha256": sha256_file(manifest_path),
            "video_feature_result_sha256": sha256_file(video_path),
            "audio_feature_result_sha256": sha256_file(audio_path),
            "video_source_commit": VIDEO_SOURCE_COMMIT,
            "audio_source_commit": AUDIO_SOURCE_COMMIT,
            "video_checkpoint_sha256": VIDEO_CHECKPOINT_SHA256,
            "audio_checkpoint_sha256": AUDIO_CHECKPOINT_SHA256,
            "train_small_groups": len(small_rows),
            "train_large_groups": len(large_rows),
            "validation_groups": len(validation_rows),
            "train_validation_video_overlap": 0,
            "updates_per_arm": TOTAL_UPDATES,
            "unique_train_examples_per_arm": {
                "train96": len(small_rows),
                "train224": len(large_rows),
            },
            "repeated_full_batch_group_visits_per_arm": {
                "train96": len(small_rows) * TOTAL_UPDATES,
                "train224": len(large_rows) * TOTAL_UPDATES,
            },
            "arms": {
                name: {
                    key: value
                    for key, value in result.items()
                    if key not in {"model", "validation_scores"}
                }
                for name, result in results.items()
            },
            "learning_curve_key_steps": [
                row
                for row in curves
                if row["step"] in {0, 32, 128, 256, 512, 1024, 1536, 1920, 2048}
            ],
            "paired_comparisons": comparisons,
            "sequence_input_controls": controls,
            "elapsed_seconds": time.perf_counter() - started,
            "validation_scope": (
                "fresh group-disjoint development holdout from official train; "
                "used for checkpoint selection; not blind or independent test"
            ),
        }
        _atomic_json(output / "result.json", summary)
        _set_effective_config_status(output / "effective_config.json", summary["status"])
        run_metadata.update(
            {
                "status": summary["status"],
                "result_sha256": sha256_file(output / "result.json"),
                "elapsed_seconds": summary["elapsed_seconds"],
            }
        )
        _atomic_json(output / "run.json", run_metadata)
        return summary
    except Exception as exc:
        _set_effective_config_status(output / "effective_config.json", "FAILED_SCALE_FIT")
        run_metadata.update(
            {"status": "FAILED_SCALE_FIT", "failure_type": type(exc).__name__, "failure": str(exc)}
        )
        _atomic_json(output / "run.json", run_metadata)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--video-features", type=Path, required=True)
    parser.add_argument("--audio-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2), flush=True)


if __name__ == "__main__":
    main()
