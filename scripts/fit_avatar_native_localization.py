"""Fit small frozen-feature AVATAR sound-source localization readouts."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.extract_avatar_dinov2_spatial_train import bbox_center_cell  # noqa: E402
from scripts.fit_avatar_decision_readout import sha256_file, write_json_atomic  # noqa: E402

SEED = 17
MAX_EPOCHS = 100
PATIENCE = 10
MIN_DELTA = 0.001
OPTION_COUNT = 40


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError(f"empty JSONL: {path}")
    return rows


def _ece(confidence: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(confidence)
    value = 0.0
    for index in range(bins):
        mask = (confidence >= edges[index]) & (
            confidence < edges[index + 1] if index < bins - 1 else confidence <= edges[index + 1]
        )
        if mask.any():
            value += float(mask.mean()) * abs(float(confidence[mask].mean() - correct[mask].mean()))
    return value if total else float("nan")


def localization_metrics(
    logits: np.ndarray, targets: np.ndarray
) -> tuple[dict[str, float], np.ndarray]:
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    correct = targets[np.arange(len(targets)), probabilities.argmax(axis=1)]
    target_sizes = targets.sum(axis=1, keepdims=True)
    target_distribution = targets / target_sizes
    target_probability = (probabilities * targets).sum(axis=1)
    nll = -np.log(np.clip(target_probability, 1e-12, 1.0))
    brier = np.square(probabilities - target_distribution).sum(axis=1)
    top5_indices = np.argsort(-probabilities, axis=1, kind="stable")[:, :5]
    top5_hit = np.take_along_axis(targets, top5_indices, axis=1).any(axis=1)
    return (
        {
            "accuracy_hit_at_1": float(correct.mean()),
            "hit_at_5": float(top5_hit.mean()),
            "nll_target_set": float(nll.mean()),
            "brier": float(brier.mean()),
            "ece_top1_correctness": _ece(probabilities.max(axis=1), correct.astype(np.float64)),
            "mean_confidence": float(probabilities.max(axis=1).mean()),
        },
        probabilities,
    )


def _load_data(args: argparse.Namespace) -> dict[str, Any]:
    train_rows = _read_jsonl(args.train_input.resolve(strict=True))
    validation_rows = _read_jsonl(args.validation_input.resolve(strict=True))
    all_rows = train_rows + validation_rows
    train_groups = {row["source_group"] for row in train_rows}
    validation_groups = {row["source_group"] for row in validation_rows}
    if train_groups & validation_groups:
        raise ValueError("source-group overlap between fit and development validation")
    if {row["asset_key"] for row in train_rows} & {row["asset_key"] for row in validation_rows}:
        raise ValueError("media identity overlap between fit and development validation")

    with np.load(args.video_features.resolve(strict=True), allow_pickle=False) as cache:
        video_keys = list(map(str, cache["asset_keys"].tolist()))
        video_values = np.asarray(cache["features"], dtype=np.float32).reshape(-1, 40, 768)
        video_times = np.asarray(cache["timestamps_seconds"], dtype=np.float64)
        video_errors = np.asarray(cache["frame_error_seconds"], dtype=np.float64)
    if len(video_keys) != len(set(video_keys)) or set(video_keys) != {
        row["asset_key"] for row in all_rows
    }:
        raise ValueError("DINO feature IDs do not exactly cover the localization fold")
    video_index = {key: index for index, key in enumerate(video_keys)}

    with np.load(args.audio_features.resolve(strict=True), allow_pickle=False) as cache:
        audio_keys = list(map(str, cache["asset_keys"].tolist()))
        audio_values = np.asarray(cache["features"], dtype=np.float32)
        audio_centers = np.asarray(cache["centers_seconds"], dtype=np.float64)
    if audio_values.shape != (len(audio_keys), 384) or len(audio_keys) != len(set(audio_keys)):
        raise ValueError("invalid TRAIN-only audio cache")
    if not set(audio_keys) <= {row["asset_key"] for row in all_rows}:
        raise ValueError("audio cache has asset keys outside this TRAIN-derived fold")
    audio_index = {key: index for index, key in enumerate(audio_keys)}
    audio_present = np.asarray([row["asset_key"] in audio_index for row in all_rows], dtype=bool)
    audio = np.zeros((len(all_rows), 384), dtype=np.float32)
    for index, row in enumerate(all_rows):
        key = row["asset_key"]
        video_position = video_index[key]
        if abs(video_times[video_position] - row["timestamp_seconds"]) > 1e-9:
            raise ValueError(f"video feature timestamp mismatch for {key}")
        if audio_present[index]:
            audio_position = audio_index[key]
            audio[index] = audio_values[audio_position]
            if abs(audio_centers[audio_position] - row["timestamp_seconds"]) > 1e-9:
                raise ValueError(f"audio feature center mismatch for {key}")

    labels = np.zeros((len(all_rows), OPTION_COUNT), dtype=bool)
    for index, row in enumerate(all_rows):
        width, height = row["original_width"], row["original_height"]
        scale = min(224 / width, 140 / height)
        resized_width, resized_height = max(1, round(width * scale)), max(1, round(height * scale))
        transform = {
            "scale_x": resized_width / width,
            "scale_y": resized_height / height,
            "pad_x": float((224 - resized_width) // 2),
            "pad_y": float((140 - resized_height) // 2),
        }
        cells = bbox_center_cell(row["bbox_xywhs"], transform)
        labels[index, cells] = True

    train_count = len(train_rows)
    # The split files are concatenated in train, validation order and preserve row order.
    return {
        "rows": all_rows,
        "train_count": train_count,
        "video": video_values[[video_index[row["asset_key"]] for row in all_rows]],
        "audio": audio,
        "audio_present": audio_present,
        "labels": labels,
        "video_frame_error": video_errors[[video_index[row["asset_key"]] for row in all_rows]],
        "audio_cache_count": len(audio_keys),
        "video_index": video_index,
    }


def _prepare_features(data: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    count = data["train_count"]
    video = data["video"].copy()
    video_mean = video[:count].mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    video_std = video[:count].std(axis=(0, 1), dtype=np.float64).astype(np.float32)
    video_std[video_std < 1e-6] = 1.0
    video = np.clip((video - video_mean) / video_std, -10.0, 10.0)

    audio = data["audio"].copy()
    available_train = data["audio_present"][:count]
    if not available_train.any():
        raise ValueError("no train examples have audio embeddings")
    audio_mean = audio[:count][available_train].mean(axis=0, dtype=np.float64).astype(np.float32)
    audio_std = audio[:count][available_train].std(axis=0, dtype=np.float64).astype(np.float32)
    audio_std[audio_std < 1e-6] = 1.0
    audio[data["audio_present"]] = np.clip(
        (audio[data["audio_present"]] - audio_mean) / audio_std, -10.0, 10.0
    )
    # Missing cache entries remain zero after normalization and are explicitly marked.
    audio_with_mask = np.concatenate(
        [audio, data["audio_present"][:, None].astype(np.float32)], axis=1
    )
    return video.astype(np.float32), audio_with_mask.astype(np.float32)


def _fit_models(
    data: dict[str, Any], video: np.ndarray, audio: np.ndarray, output: Path
) -> dict[str, Any]:
    import torch
    from torch import nn
    from torch.nn import functional as functional

    torch.set_num_threads(min(8, max(1, torch.get_num_threads())))
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cpu")
    x_video = torch.from_numpy(video)
    x_audio = torch.from_numpy(audio)
    targets = torch.from_numpy(data["labels"])
    train_count = data["train_count"]
    history_by_model: dict[str, Any] = {}
    probabilities_by_model: dict[str, np.ndarray] = {}
    joint_best_state = None

    class VideoOnly(nn.Module):
        def __init__(self):
            super().__init__()
            self.score = nn.Linear(768, 1, bias=False)
            self.cell_bias = nn.Parameter(torch.zeros(OPTION_COUNT))

        def forward(self, v, _a):
            return self.score(v).squeeze(-1) + self.cell_bias

    class AudioOnly(nn.Module):
        def __init__(self):
            super().__init__()
            self.score = nn.Linear(385, OPTION_COUNT)

        def forward(self, _v, a):
            return self.score(a)

    class Joint(nn.Module):
        def __init__(self):
            super().__init__()
            self.video_projection = nn.Linear(768, 32, bias=False)
            self.audio_projection = nn.Linear(385, 32, bias=False)
            self.video_score = nn.Linear(768, 1, bias=False)
            self.cell_bias = nn.Parameter(torch.zeros(OPTION_COUNT))

        def forward(self, v, a):
            video_latent = self.video_projection(v)
            audio_latent = self.audio_projection(a)
            interaction = (video_latent * audio_latent[:, None, :]).sum(-1) / np.sqrt(32)
            return interaction + self.video_score(v).squeeze(-1) + self.cell_bias

    constructors = {"video_only": VideoOnly, "audio_only": AudioOnly, "joint": Joint}
    results: dict[str, Any] = {}
    started_all = time.perf_counter()
    for model_index, (name, constructor) in enumerate(constructors.items()):
        torch.manual_seed(SEED + model_index)
        model = constructor().to(device)
        parameters = sum(parameter.numel() for parameter in model.parameters())
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=0.001)
        best_nll = float("inf")
        best_state = None
        best_epoch = 0
        best_train_metrics = None
        stale = 0
        history = []
        model_started = time.perf_counter()
        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            permutation = torch.randperm(train_count)
            train_loss_sum = 0.0
            for start in range(0, train_count, 128):
                indices = permutation[start : start + 128]
                logits = model(x_video[indices], x_audio[indices])
                log_probabilities = functional.log_softmax(logits, dim=1)
                selected = log_probabilities.masked_fill(~targets[indices], -torch.inf)
                loss = -torch.logsumexp(selected, dim=1).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite {name} training loss at epoch {epoch}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if not torch.isfinite(gradient_norm):
                    raise FloatingPointError(f"non-finite {name} gradient at epoch {epoch}")
                optimizer.step()
                train_loss_sum += float(loss.detach()) * len(indices)

            model.eval()
            with torch.no_grad():
                train_logits = model(x_video[:train_count], x_audio[:train_count]).numpy()
                validation_logits = model(x_video[train_count:], x_audio[train_count:]).numpy()
            train_metrics, _ = localization_metrics(train_logits, data["labels"][:train_count])
            validation_metrics, _ = localization_metrics(
                validation_logits, data["labels"][train_count:]
            )
            history.append(
                {
                    "epoch": epoch,
                    "train_cross_entropy": train_loss_sum / train_count,
                    "train_fixed_metrics": train_metrics,
                    "validation": validation_metrics,
                }
            )
            history_by_model[name] = history
            write_json_atomic(output / "training-history.json", history_by_model)
            if validation_metrics["nll_target_set"] < best_nll - MIN_DELTA:
                best_nll = validation_metrics["nll_target_set"]
                best_epoch = epoch
                best_train_metrics = train_metrics
                stale = 0
                best_state = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }
                torch.save(best_state, output / f"{name}-best.pt")
            else:
                stale += 1
            if epoch == 1 or epoch % 10 == 0:
                print(
                    json.dumps(
                        {
                            "model": name,
                            "epoch": epoch,
                            "max_epochs": MAX_EPOCHS,
                            "train_nll": train_metrics["nll_target_set"],
                            "validation_nll": validation_metrics["nll_target_set"],
                            "best_epoch": best_epoch,
                            "elapsed_seconds": round(time.perf_counter() - started_all, 1),
                        }
                    ),
                    flush=True,
                )
            if stale >= PATIENCE:
                break
        if best_state is None:
            raise RuntimeError(f"{name} did not produce a validation-selected checkpoint")
        if name == "joint":
            joint_best_state = best_state
        last_state = {
            key: value.detach().cpu().clone() for key, value in model.state_dict().items()
        }
        torch.save(last_state, output / f"{name}-last.pt")
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            validation_logits = model(x_video[train_count:], x_audio[train_count:]).numpy()
        metrics, probabilities = localization_metrics(
            validation_logits, data["labels"][train_count:]
        )
        results[name] = {
            "trainable_parameters": parameters,
            "best_epoch": best_epoch,
            "epochs_run": len(history),
            "best_train": best_train_metrics,
            "best_validation": metrics,
            "selected_state_sha256": sha256_file(output / f"{name}-best.pt"),
            "last_state_sha256": sha256_file(output / f"{name}-last.pt"),
            "wall_seconds": round(time.perf_counter() - model_started, 3),
        }
        probabilities_by_model[name] = probabilities
        history_by_model[name] = history
        prediction_path = output / f"{name}-validation-predictions.jsonl"
        with prediction_path.open("x", encoding="utf-8") as stream:
            for offset, (row, prob, target) in enumerate(
                zip(
                    data["rows"][train_count:],
                    probabilities,
                    data["labels"][train_count:],
                    strict=True,
                )
            ):
                target_cells = np.flatnonzero(target).tolist()
                prediction = int(np.argmax(prob))
                stream.write(
                    json.dumps(
                        {
                            "example_id": row["asset_key"],
                            "source_group": row["source_group"],
                            "target_cells": target_cells,
                            "prediction_cell": prediction,
                            "probabilities": prob.astype(float).tolist(),
                            "correct": prediction in target_cells,
                            "audio_present": bool(data["audio_present"][train_count + offset]),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )

        groups = [row["source_group"] for row in data["rows"][train_count:]]
        group_metrics = {}
        for group in sorted(set(groups)):
            selected = np.asarray([value == group for value in groups])
            values, _ = localization_metrics(
                validation_logits[selected], data["labels"][train_count:][selected]
            )
            group_metrics[group] = {"count": int(selected.sum()), **values}
        results[name]["validation_by_source_group"] = group_metrics
        results[name]["source_group_macro"] = {
            metric: float(np.mean([row[metric] for row in group_metrics.values()]))
            for metric in (
                "accuracy_hit_at_1",
                "nll_target_set",
                "brier",
                "ece_top1_correctness",
            )
        }
        results[name]["train_validation_gap"] = {
            "accuracy_hit_at_1": (
                best_train_metrics["accuracy_hit_at_1"] - metrics["accuracy_hit_at_1"]
            ),
            "nll_target_set": metrics["nll_target_set"] - best_train_metrics["nll_target_set"],
            "brier": metrics["brier"] - best_train_metrics["brier"],
            "ece_top1_correctness": (
                metrics["ece_top1_correctness"] - best_train_metrics["ece_top1_correctness"]
            ),
        }

    # Keep the missingness pattern fixed while testing whether joint predictions use audio content.
    model = Joint().to(device)
    if joint_best_state is None:
        raise RuntimeError("joint validation selector did not retain a checkpoint")
    model.load_state_dict(joint_best_state)
    model.eval()
    val_video = x_video[train_count:]
    val_audio = x_audio[train_count:].clone()
    with torch.no_grad():
        zero_content = val_audio.clone()
        zero_content[:, :384] = 0
        zero_logits = model(val_video, zero_content).numpy()
        shuffled_audio = val_audio.clone()
        available = torch.nonzero(shuffled_audio[:, 384] > 0.5, as_tuple=False).flatten()
        if len(available) > 1:
            permutation = torch.randperm(
                len(available), generator=torch.Generator().manual_seed(SEED)
            )
            shuffled_audio[available, :384] = shuffled_audio[available[permutation], :384]
        shuffled_logits = model(val_video, shuffled_audio).numpy()
    zero_metrics, _ = localization_metrics(zero_logits, data["labels"][train_count:])
    shuffled_metrics, _ = localization_metrics(shuffled_logits, data["labels"][train_count:])
    results["joint"]["validation_audio_controls"] = {
        "zeroed_audio_content_missingness_preserved": zero_metrics,
        "audio_content_permuted_missingness_preserved": shuffled_metrics,
        "permutation_seed": SEED,
    }

    write_json_atomic(output / "training-history.json", history_by_model)
    return {
        "models": results,
        "paired_joint_minus_video_accuracy": _paired_group_bootstrap(
            probabilities_by_model["joint"],
            probabilities_by_model["video_only"],
            data["labels"][train_count:],
            [row["source_group"] for row in data["rows"][train_count:]],
        ),
        "total_fit_wall_seconds": round(time.perf_counter() - started_all, 3),
    }


def _baseline_prior(targets: np.ndarray, evaluation_count: int) -> np.ndarray:
    counts = targets.sum(axis=0).astype(np.float64)
    probabilities = (counts + 1.0) / (counts.sum() + len(counts))
    return np.broadcast_to(np.log(probabilities), (evaluation_count, len(probabilities))).copy()


def _paired_group_bootstrap(
    joint_probabilities: np.ndarray,
    video_probabilities: np.ndarray,
    targets: np.ndarray,
    groups: list[str],
    *,
    repetitions: int = 2000,
) -> dict[str, Any]:
    joint_correct = targets[np.arange(len(targets)), joint_probabilities.argmax(axis=1)]
    video_correct = targets[np.arange(len(targets)), video_probabilities.argmax(axis=1)]
    difference = joint_correct.astype(np.float64) - video_correct.astype(np.float64)
    unique_groups = sorted(set(groups))
    group_indices = {group: np.flatnonzero(np.asarray(groups) == group) for group in unique_groups}
    rng = np.random.default_rng(SEED)
    samples = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        selected_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        selected_rows = np.concatenate([group_indices[group] for group in selected_groups])
        samples[index] = difference[selected_rows].mean()
    return {
        "point_estimate": float(difference.mean()),
        "ci95_percentile": [float(value) for value in np.quantile(samples, [0.025, 0.975])],
        "repetitions": repetitions,
        "seed": SEED,
        "unit": "source YouTube group; paired rows resampled with replacement",
        "validation_source_group_count": len(unique_groups),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output}")
    output.mkdir(parents=True)
    data = _load_data(args)
    video, audio = _prepare_features(data)
    count = data["train_count"]
    import torch

    effective_config = {
        "experiment_id": "avatar-native-audio-video-localization-readout-v1",
        "seed": SEED,
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": PATIENCE,
        "min_delta": MIN_DELTA,
        "optimizer": "AdamW",
        "learning_rate": 0.02,
        "weight_decay": 0.001,
        "batch_size": 128,
        "option_count": OPTION_COUNT,
        "modality_encoders_frozen": True,
        "device": "cpu",
        "train_input_sha256": sha256_file(args.train_input.resolve(strict=True)),
        "validation_input_sha256": sha256_file(args.validation_input.resolve(strict=True)),
        "video_features_sha256": sha256_file(args.video_features.resolve(strict=True)),
        "audio_features_sha256": sha256_file(args.audio_features.resolve(strict=True)),
        "fit_script_sha256": sha256_file(Path(__file__).resolve()),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "sealed_audit_accessed": False,
    }
    write_json_atomic(output / "effective-config.json", effective_config)
    baseline_metrics, _ = localization_metrics(
        _baseline_prior(data["labels"][:count], len(data["labels"]) - count),
        data["labels"][count:],
    )
    fitted = _fit_models(data, video, audio, output)
    result = {
        "experiment_id": "avatar-native-audio-video-localization-readout-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "selection": {
            "metric": "validation target-set NLL",
            "min_delta": MIN_DELTA,
            "patience": PATIENCE,
        },
        "counts": {
            "train_examples": count,
            "validation_examples": len(data["rows"]) - count,
            "audio_feature_count": data["audio_cache_count"],
            "train_audio_missing": int((~data["audio_present"][:count]).sum()),
            "validation_audio_missing": int((~data["audio_present"][count:]).sum()),
            "unique_train_source_groups": len(
                {row["source_group"] for row in data["rows"][:count]}
            ),
            "unique_validation_source_groups": len(
                {row["source_group"] for row in data["rows"][count:]}
            ),
            "frame_time_error_max_seconds": float(data["video_frame_error"].max(initial=0.0)),
            "option_count": OPTION_COUNT,
            "bbox_targets_with_multiple_boxes": sum(
                len(row["bbox_xywhs"]) > 1 for row in data["rows"]
            ),
        },
        "baseline_most_frequent_train_cell": baseline_metrics,
        **fitted,
        "train_input_sha256": sha256_file(args.train_input.resolve(strict=True)),
        "validation_input_sha256": sha256_file(args.validation_input.resolve(strict=True)),
        "video_features_sha256": sha256_file(args.video_features.resolve(strict=True)),
        "audio_features_sha256": sha256_file(args.audio_features.resolve(strict=True)),
        "device": "cpu",
        "sealed_audit_accessed": False,
    }
    write_json_atomic(output / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-input", type=Path, required=True)
    parser.add_argument("--validation-input", type=Path, required=True)
    parser.add_argument("--video-features", type=Path, required=True)
    parser.add_argument("--audio-features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    result = run(parser.parse_args())
    summary = {
        "experiment_id": result["experiment_id"],
        "counts": result["counts"],
        "baseline_most_frequent_train_cell": result["baseline_most_frequent_train_cell"],
        "models": {
            name: {
                key: model[key]
                for key in (
                    "trainable_parameters",
                    "best_epoch",
                    "epochs_run",
                    "best_train",
                    "best_validation",
                    "source_group_macro",
                    "train_validation_gap",
                    "validation_audio_controls",
                    "wall_seconds",
                )
                if key in model
            }
            for name, model in result["models"].items()
        },
        "paired_joint_minus_video_accuracy": result["paired_joint_minus_video_accuracy"],
        "total_fit_wall_seconds": result["total_fit_wall_seconds"],
        "sealed_audit_accessed": result["sealed_audit_accessed"],
    }
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
