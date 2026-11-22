"""Fit fixed-budget HapTile Decision ablations on CPU; select by validation only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
import yaml
from torch import Tensor
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tiny_omni_decision.haptile_action import ACTION_LABELS  # noqa: E402
from tiny_omni_decision.haptile_decision import (  # noqa: E402
    HAPTILE_MODES,
    HapTileActionDecisionScorer,
)

CONFIG_PATH = ROOT / "configs/pretrained_reuse/haptile_temporal_decision.yaml"
SPLIT_PATH = ROOT / "manifests/pretrained_reuse/haptile-task-split-v1-20261122.json"
SEED = 17
BOOTSTRAP_DRAWS = 2000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_config(config_path: Path) -> tuple[dict[str, Any], str]:
    payload = config_path.read_bytes()
    config = yaml.safe_load(payload)
    if not isinstance(config, dict):
        raise ValueError("experiment config must be a YAML mapping")
    expected = {
        "seed": SEED,
        "device": "cpu",
        "experiment_id": "haptile-temporal-action-decision-seed17-v1",
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("seed, device or experiment identity differs from the frozen protocol")
    training = config.get("training", {})
    if (training.get("epochs") != 80 or training.get("batch_size") != 32
            or training.get("learning_rate") != 1e-3
            or training.get("weight_decay") != 1e-4
            or training.get("min_delta") != 1e-4
            or training.get("early_stopping") is not False
            or training.get("seed_per_arm") != SEED
            or training.get("option_order_randomized_each_train_batch") is not True
            or training.get("checkpoint_selection")
            != "minimum_validation_macro_task_group_nll"):
        raise ValueError("training schedule differs from the frozen protocol")
    if config.get("readout", {}).get("modes") != list(HAPTILE_MODES):
        raise ValueError("configured ablation modes differ from the model implementation")
    if config.get("prohibitions", {}).get("sealed_audit_loading") is not False:
        raise ValueError("sealed-audit loading must remain disabled")
    if config.get("evaluation", {}).get("calibration_temperature") != {
        "fit_after_validation_selection": True,
        "objective": "minimum_calibration_macro_task_group_nll",
        "search": "log_spaced_scalar_grid",
        "minimum": 0.25,
        "maximum": 4.0,
        "points": 151,
    }:
        raise ValueError("temperature calibration policy differs from the frozen protocol")
    if config.get("evaluation", {}).get("paired_controls") != [
        "best_single_stream", "missing_video", "missing_sensor",
        "task_group_shuffled_video", "task_group_shuffled_sensor",
    ]:
        raise ValueError("paired validation controls differ from the frozen protocol")
    if sha256_file(SPLIT_PATH) != config["data"].get("split_manifest_sha256"):
        raise ValueError("frozen split manifest SHA-256 differs")
    return config, hashlib.sha256(payload).hexdigest()


def _load_split(
    artifact_root: Path, split: str, expected_count: int, config: dict[str, Any],
) -> dict[str, np.ndarray]:
    directory = artifact_root / f"{split}-joined-v1"
    result_path = directory / "result.json"
    feature_path = directory / "decision-features.npz"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if split == "train":
        source_root = artifact_root / "train-corpus-v1"
        receipt_path = source_root / "train-corpus-receipt.json"
        action_path = source_root / "train-action-preflight-v2.json"
        expected_receipt = config["data"]["train_receipt_sha256"]
        expected_action = config["data"]["train_action_preflight_sha256"]
        video_directory = artifact_root / "train-video-features-v3"
    else:
        source_root = artifact_root / f"{split}-corpus-v1"
        receipt_path = source_root / f"{split}-corpus-receipt.json"
        action_path = artifact_root / f"{split}-action-preflight-v1.json"
        expected_receipt = config["data"][f"{split}_receipt_sha256"]
        expected_action = config["data"][f"{split}_action_preflight_sha256"]
        video_directory = artifact_root / (
            "calibration-video-features-v2"
            if split == "calibration" else "validation-video-features-v1"
        )
    eligibility_path = artifact_root / f"{split}-video-window-eligibility-v2.json"
    expected_eligibility = config["data"][f"{split}_video_window_eligibility_sha256"]
    state_directory = artifact_root / (
        "train-state-text-v3" if split == "train" else f"{split}-state-text-v2"
    )
    text_directory = artifact_root / f"{split}-text-features-v2"
    input_hashes = result.get("input_hashes", {})
    state_result_path = state_directory / "result.json"
    video_result_path = video_directory / "result.json"
    text_result_path = text_directory / "result.json"
    if (sha256_file(receipt_path) != expected_receipt
            or sha256_file(action_path) != expected_action
            or input_hashes.get("receipt") != expected_receipt
            or input_hashes.get("action_preflight") != expected_action
            or sha256_file(eligibility_path) != expected_eligibility
            or input_hashes.get("video_window_eligibility") != expected_eligibility
            or input_hashes.get("state_result") != sha256_file(state_result_path)
            or input_hashes.get("video_result") != sha256_file(video_result_path)
            or input_hashes.get("text_result") != sha256_file(text_result_path)):
        raise ValueError(f"{split} feature package points to unexpected frozen inputs")
    video_result = json.loads(video_result_path.read_text(encoding="utf-8"))
    text_result = json.loads(text_result_path.read_text(encoding="utf-8"))
    video_audit_opened = video_result.get(
        "sealed_audit_opened", video_result.get("validation_or_sealed_audit_opened")
    )
    if (video_result.get("model_revision") != config["video_encoder"]["revision"]
            or video_result.get("checkpoint_sha256")
            != config["video_encoder"]["checkpoint_sha256"]
            or video_audit_opened is not False
            or text_result.get("model_revision") != config["text_encoder"]["revision"]
            or text_result.get("model_weights_sha256")
            != config["text_encoder"]["weights_sha256"]
            or text_result.get("sealed_audit_opened") is not False):
        raise ValueError(f"{split} feature encoders are not the frozen checkpoints")
    if (result.get("split") != split or result.get("sealed_audit_opened") is not False
            or sha256_file(feature_path) != result.get("feature_file_sha256")
            or result.get("examples") != expected_count):
        raise ValueError(f"{split} feature package provenance, size or split gate failed")
    with np.load(feature_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    required = {
        "sample_paths": (expected_count,),
        "media_sha256": (expected_count,),
        "task_groups": (expected_count,),
        "targets": (expected_count,),
        "video_values": (expected_count, 4, 8, 768),
        "video_timestamps_ms": (expected_count, 8),
        "state_values": (expected_count, 8, 25),
        "state_timestamps_ms": (expected_count, 8),
        "context_features": (expected_count, 384),
        "option_features": (expected_count, 7, 1536),
        "context_features_no_instruction": (expected_count, 384),
        "option_features_no_instruction": (expected_count, 7, 1536),
    }
    if any(name not in arrays or arrays[name].shape != shape
           for name, shape in required.items()):
        raise ValueError(f"{split} feature arrays differ from the typed readout contract")
    if (len(set(arrays["sample_paths"].tolist())) != expected_count
            or len(set(arrays["media_sha256"].tolist())) != expected_count
            or not np.isin(arrays["targets"], np.arange(len(ACTION_LABELS))).all()):
        raise ValueError(f"{split} media IDs, hashes or targets are invalid")
    for name in required:
        if (
            name not in {"sample_paths", "media_sha256", "task_groups", "targets"}
            and not np.isfinite(arrays[name]).all()
        ):
            raise ValueError(f"{split} {name} contains non-finite values")
    return arrays


def _metrics(
    logits: Tensor, targets: Tensor, task_groups: list[str], sample_paths: list[str],
) -> dict[str, Any]:
    probabilities = torch.softmax(logits.float(), dim=-1).cpu().numpy()
    target_ids = targets.cpu().numpy().astype(np.int64, copy=False)
    predicted = probabilities.argmax(axis=1)
    correct = (predicted == target_ids).astype(np.float64)
    nll = -np.log(np.maximum(probabilities[np.arange(len(target_ids)), target_ids], 1e-12))
    one_hot = np.eye(len(ACTION_LABELS), dtype=np.float64)[target_ids]
    brier = np.square(probabilities - one_hot).sum(axis=1)
    confidence = probabilities.max(axis=1)
    ece = 0.0
    for bucket in range(15):
        lower, upper = bucket / 15, (bucket + 1) / 15
        mask = (confidence >= lower) & (
            (confidence < upper) | ((bucket == 14) & (confidence <= upper))
        )
        if mask.any():
            ece += float(mask.mean()) * abs(
                float(correct[mask].mean()) - float(confidence[mask].mean())
            )
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(task_groups):
        grouped[group].append(index)
    per_group = {}
    for group, indices in sorted(grouped.items()):
        idx = np.asarray(indices, dtype=np.int64)
        per_group[group] = {
            "count": int(len(idx)),
            "accuracy": float(correct[idx].mean()),
            "nll": float(nll[idx].mean()),
            "brier": float(brier[idx].mean()),
            "ece": _ece(confidence[idx], correct[idx]),
        }
    macro = {
        metric: float(np.mean([row[metric] for row in per_group.values()]))
        for metric in ("accuracy", "nll", "brier", "ece")
    }
    return {
        "count": int(len(target_ids)),
        "task_group_count": len(per_group),
        "accuracy": float(correct.mean()),
        "nll": float(nll.mean()),
        "brier": float(brier.mean()),
        "ece": ece,
        "macro_task_group": macro,
        "macro_task_group_nll": macro["nll"],
        "per_task_group": per_group,
        "predictions": [
            {
                "sample_id": sample_paths[i],
                "task_group": task_groups[i],
                "target": ACTION_LABELS[int(target_ids[i])],
                "prediction": ACTION_LABELS[int(predicted[i])],
                "probabilities": {
                    label: float(probabilities[i, column])
                    for column, label in enumerate(ACTION_LABELS)
                },
            }
            for i in range(len(target_ids))
        ],
    }


def _train_prior_metrics(
    train_targets: np.ndarray, validation: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Evaluate a train-label-frequency prior without looking at validation labels."""
    counts = np.bincount(
        np.asarray(train_targets, dtype=np.int64), minlength=len(ACTION_LABELS)
    )
    if counts.sum() != len(train_targets) or counts.sum() == 0:
        raise ValueError("train targets do not form a valid class prior")
    probabilities = counts.astype(np.float64) / counts.sum()
    logits = torch.log(torch.from_numpy(probabilities).float().clamp_min(1e-12))
    repeated_logits = logits.unsqueeze(0).expand(len(validation["targets"]), -1)
    return _metrics(
        repeated_logits, torch.from_numpy(validation["targets"].copy()).long(),
        validation["task_groups"].tolist(), validation["sample_paths"].tolist(),
    )


def _ece(confidence: np.ndarray, correct: np.ndarray) -> float:
    value = 0.0
    for bucket in range(15):
        lower, upper = bucket / 15, (bucket + 1) / 15
        mask = (confidence >= lower) & (
            (confidence < upper) | ((bucket == 14) & (confidence <= upper))
        )
        if mask.any():
            value += float(mask.mean()) * abs(
                float(correct[mask].mean()) - float(confidence[mask].mean())
            )
    return value


def _batch_tensors(
    arrays: dict[str, np.ndarray], indices: list[int], state_mean: Tensor,
    state_std: Tensor, *, mode: str, shuffle_options: bool, rng: random.Random,
) -> tuple[Tensor, ...]:
    pair_key = "option_features_no_instruction" if mode == "state_only" else "option_features"
    context_key = "context_features_no_instruction" if mode == "state_only" else "context_features"
    options = torch.from_numpy(arrays[pair_key][indices].copy())
    context = torch.from_numpy(arrays[context_key][indices].copy())
    targets = torch.from_numpy(arrays["targets"][indices].copy()).long()
    if shuffle_options:
        order = [rng.sample(range(7), 7) for _ in indices]
        options = torch.stack([options[row, permutation] for row, permutation in enumerate(order)])
        targets = torch.tensor(
            [permutation.index(int(targets[row])) for row, permutation in enumerate(order)],
            dtype=torch.long,
        )
    video = torch.from_numpy(arrays["video_values"][indices].copy())
    video_times = torch.from_numpy(arrays["video_timestamps_ms"][indices].copy())
    state = torch.from_numpy(arrays["state_values"][indices].copy())
    state = (state - state_mean) / state_std
    state_times = torch.from_numpy(arrays["state_timestamps_ms"][indices].copy())
    option_valid = torch.ones((len(indices), 7), dtype=torch.bool)
    return video, video_times, state, state_times, context, options, option_valid, targets


def _run_batch(
    model: HapTileActionDecisionScorer, mode: str, batch: tuple[Tensor, ...]
) -> tuple[Tensor, Tensor]:
    video, video_times, state, state_times, context, options, valid, targets = batch
    logits = model(
        video, video_times, state, state_times, context, options, valid, mode=mode
    )
    return logits, targets


@torch.inference_mode()
def _predict_logits(
    model: HapTileActionDecisionScorer, mode: str, arrays: dict[str, np.ndarray],
    state_mean: Tensor, state_std: Tensor, *, batch_size: int = 64,
) -> tuple[Tensor, Tensor]:
    model.eval()
    outputs: list[Tensor] = []
    targets: list[Tensor] = []
    count = len(arrays["targets"])
    for start in range(0, count, batch_size):
        indices = list(range(start, min(count, start + batch_size)))
        batch = _batch_tensors(
            arrays, indices, state_mean, state_std, mode=mode,
            shuffle_options=False, rng=random.Random(SEED),
        )
        logits, batch_targets = _run_batch(model, mode, batch)
        outputs.append(logits.cpu())
        targets.append(batch_targets.cpu())
    return torch.cat(outputs), torch.cat(targets)


def _evaluate(
    model: HapTileActionDecisionScorer, mode: str, arrays: dict[str, np.ndarray],
    state_mean: Tensor, state_std: Tensor, *, batch_size: int = 64,
    temperature: float = 1.0,
) -> dict[str, Any]:
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    logits, targets = _predict_logits(
        model, mode, arrays, state_mean, state_std, batch_size=batch_size
    )
    return _metrics(
        logits / temperature, targets, arrays["task_groups"].tolist(),
        arrays["sample_paths"].tolist(),
    )


def _fit_temperature(
    logits: Tensor, targets: Tensor, task_groups: list[str], *,
    minimum: float = 0.25, maximum: float = 4.0, points: int = 151,
) -> dict[str, Any]:
    """Fit one scalar on calibration tasks, minimizing macro task-group NLL."""
    if (logits.ndim != 2 or logits.shape[1] != len(ACTION_LABELS)
            or targets.shape != (len(logits),) or len(task_groups) != len(logits)
            or len(logits) == 0 or not torch.isfinite(logits).all()):
        raise ValueError("calibration logits, labels or groups are invalid")
    groups = sorted(set(task_groups))
    if len(groups) < 2:
        raise ValueError("temperature fitting requires multiple calibration task groups")
    grouped_indices = [
        torch.tensor([i for i, group in enumerate(task_groups) if group == name], dtype=torch.long)
        for name in groups
    ]
    if (not math.isfinite(minimum) or not math.isfinite(maximum)
            or minimum <= 0 or maximum <= minimum or points < 2):
        raise ValueError("temperature search grid is invalid")
    temperatures = np.geomspace(minimum, maximum, points)
    scores = []
    for temperature in temperatures:
        losses = F.cross_entropy(
            logits.float() / float(temperature), targets.long(), reduction="none"
        )
        score = float(np.mean([
            losses[index].mean().item() for index in grouped_indices
        ]))
        scores.append(score)
    selected = min(
        range(len(temperatures)),
        key=lambda index: (scores[index], abs(math.log(float(temperatures[index])))),
    )
    return {
        "method": "deterministic scalar temperature grid search",
        "objective": "minimum calibration macro task-group NLL",
        "temperature_range": [minimum, maximum],
        "grid_points": len(temperatures),
        "temperature": float(temperatures[selected]),
        "calibration_macro_task_group_nll": scores[selected],
        "calibration_task_groups": groups,
        "calibration_examples": len(logits),
        "calibration_fitting_metrics_are_in_sample": True,
    }


def _shuffle_values_across_task_groups(
    values: np.ndarray, task_groups: list[str],
) -> np.ndarray:
    """Deterministically replace every task group with another group's stream."""
    if values.shape[0] != len(task_groups):
        raise ValueError("stream values and task groups are misaligned")
    groups = sorted(set(task_groups))
    if len(groups) < 2:
        raise ValueError("cross-task stream control requires multiple groups")
    indices = {group: np.flatnonzero(np.asarray(task_groups) == group) for group in groups}
    shuffled = values.copy()
    for group_index, group in enumerate(groups):
        source_group = groups[(group_index + 1) % len(groups)]
        destination = indices[group]
        source = indices[source_group]
        source_indices = source[np.arange(len(destination)) % len(source)]
        shuffled[destination] = values[source_indices]
    return shuffled


def _apply_validation_control(
    arrays: dict[str, np.ndarray], control: str, state_mean: Tensor,
) -> dict[str, np.ndarray]:
    controlled = dict(arrays)
    if control == "missing_video":
        controlled["video_values"] = np.zeros_like(arrays["video_values"])
    elif control == "missing_sensor":
        mean = state_mean.detach().cpu().numpy().astype(np.float32, copy=False)
        controlled["state_values"] = np.broadcast_to(
            mean[None, None, :], arrays["state_values"].shape
        ).copy()
    elif control == "task_group_shuffled_video":
        controlled["video_values"] = _shuffle_values_across_task_groups(
            arrays["video_values"], arrays["task_groups"].tolist()
        )
    elif control == "task_group_shuffled_sensor":
        controlled["state_values"] = _shuffle_values_across_task_groups(
            arrays["state_values"], arrays["task_groups"].tolist()
        )
    else:
        raise ValueError(f"unknown validation control: {control}")
    return controlled


def _fit_arm(
    mode: str, train: dict[str, np.ndarray], validation: dict[str, np.ndarray],
    state_mean: Tensor, state_std: Tensor, output: Path, *, epochs: int,
    batch_size: int, learning_rate: float, weight_decay: float, min_delta: float,
) -> dict[str, Any]:
    torch.manual_seed(SEED)
    model = HapTileActionDecisionScorer()
    active = (model.text_only_readout if mode == "text_only" else model.readout)
    parameters = [parameter for parameter in active.parameters() if parameter.requires_grad]
    trainable_count = sum(parameter.numel() for parameter in parameters)
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    rng = random.Random(SEED)
    history_path = output / f"{mode}-learning-curve.jsonl"
    prediction_path = output / f"{mode}-validation-predictions.jsonl"
    best_path = output / f"{mode}-best.pt"
    last_path = output / f"{mode}-last.pt"
    best_score = math.inf
    best_epoch = 0
    best_metrics: dict[str, Any] | None = None
    last_metrics: dict[str, Any] | None = None
    updates = 0
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        order = list(range(len(train["targets"])))
        rng.shuffle(order)
        losses: list[float] = []
        for start in range(0, len(order), batch_size):
            selected = order[start:start + batch_size]
            batch = _batch_tensors(
                train, selected, state_mean, state_std, mode=mode,
                shuffle_options=True, rng=rng,
            )
            optimizer.zero_grad(set_to_none=True)
            logits, targets = _run_batch(model, mode, batch)
            loss = F.cross_entropy(logits.float(), targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite loss in {mode} epoch {epoch}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"nonfinite gradient in {mode} epoch {epoch}")
            optimizer.step()
            losses.append(float(loss.detach()))
            updates += 1

        train_metrics = _evaluate(model, mode, train, state_mean, state_std)
        validation_metrics = _evaluate(model, mode, validation, state_mean, state_std)
        score = float(validation_metrics["macro_task_group_nll"])
        improved = score < best_score - min_delta
        if improved:
            best_score, best_epoch, best_metrics = score, epoch, validation_metrics
            torch.save(model.state_dict(), best_path)
        torch.save(model.state_dict(), last_path)
        history_row = {
            "epoch": epoch,
            "optimizer_updates_total": updates,
            "train_cross_entropy_mean_batch": float(np.mean(losses)),
            "train": {key: value for key, value in train_metrics.items()
                      if key != "predictions"},
            "validation": {key: value for key, value in validation_metrics.items()
                           if key != "predictions"},
            "selection_score": score,
            "best_epoch": best_epoch,
            "best_score": best_score,
            "best_updated": improved,
            "train_validation_nll_gap": (
                validation_metrics["nll"] - train_metrics["nll"]
            ),
            "elapsed_seconds": time.perf_counter() - started,
        }
        with history_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(history_row, separators=(",", ":")) + "\n")
        with prediction_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({
                "epoch": epoch,
                "predictions": validation_metrics["predictions"],
            }, separators=(",", ":")) + "\n")
        last_metrics = validation_metrics
        (output / "progress.json").write_text(json.dumps({
            "status": "in_progress",
            "active_mode": mode,
            "completed_epochs": epoch,
            "planned_epochs": epochs,
            "best_epoch": best_epoch,
            "optimizer_updates_total": updates,
        }, indent=2) + "\n", encoding="utf-8")
        print(
            f"{mode} epoch {epoch}/{epochs} train_nll={train_metrics['nll']:.4f} "
            f"val_macro_nll={score:.4f} best={best_score:.4f}@{best_epoch}",
            flush=True,
        )
    if best_metrics is None or last_metrics is None:
        raise RuntimeError(f"no checkpoint was selected for {mode}")
    reloaded = HapTileActionDecisionScorer()
    reloaded.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=True))
    reload_metrics = _evaluate(reloaded, mode, validation, state_mean, state_std)
    if abs(reload_metrics["macro_task_group_nll"] - best_score) > 1e-8:
        raise RuntimeError(f"best checkpoint reload failed metric parity for {mode}")
    return {
        "mode": mode,
        "trainable_parameters": trainable_count,
        "optimizer_updates": updates,
        "best_epoch": best_epoch,
        "best_validation": {key: value for key, value in best_metrics.items()
                            if key != "predictions"},
        "last_validation": {key: value for key, value in last_metrics.items()
                            if key != "predictions"},
        "best_reload_verified": True,
        "best_checkpoint": str(best_path),
        "best_checkpoint_sha256": sha256_file(best_path),
        "last_checkpoint": str(last_path),
        "last_checkpoint_sha256": sha256_file(last_path),
        "learning_curve_sha256": sha256_file(history_path),
        "validation_predictions_sha256": sha256_file(prediction_path),
        "wall_seconds": time.perf_counter() - started,
    }


def _paired_group_bootstrap(
    reference: dict[str, Any], target: dict[str, Any], *, draws: int = BOOTSTRAP_DRAWS,
) -> dict[str, Any]:
    ref, alt = reference["per_task_group"], target["per_task_group"]
    groups = sorted(set(ref) & set(alt))
    if not groups:
        raise ValueError("paired group bootstrap requires shared task groups")
    rng = random.Random(SEED)
    output: dict[str, Any] = {}
    for metric in ("accuracy", "nll", "brier", "ece"):
        deltas = [
            float(np.mean([
                ref[group][metric] - alt[group][metric]
                for group in (rng.choices(groups, k=len(groups)))
            ]))
            for _ in range(draws)
        ]
        deltas.sort()
        observed = float(np.mean([ref[group][metric] - alt[group][metric] for group in groups]))
        output[metric] = {
            "reference_minus_target_macro_group": observed,
            "percentile_95_interval": [
                deltas[int(0.025 * draws)],
                deltas[min(draws - 1, int(0.975 * draws))],
            ],
        }
    return {
        "method": "paired task-group bootstrap with replacement",
        "seed": SEED,
        "draws": draws,
        "task_groups": groups,
        "selection_affected_validation": True,
        "metrics": output,
    }


def run_experiment(
    artifact_root: Path, output: Path, config_path: Path = CONFIG_PATH
) -> dict[str, Any]:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite experiment output: {output}")
    if ROOT == output or ROOT in output.parents:
        raise ValueError("model artifacts must be saved outside the Git worktree")
    config, config_hash = _load_config(config_path)
    data = config["data"]
    train = _load_split(
        artifact_root, "train", int(data["train_examples"]), config
    )
    validation = _load_split(
        artifact_root, "validation", int(data["validation_examples"]), config
    )
    train_groups = set(train["task_groups"].tolist())
    validation_groups = set(validation["task_groups"].tolist())
    if train_groups & validation_groups:
        raise ValueError("train and validation task groups overlap")
    if (len(train_groups) != int(data["train_task_groups"])
            or len(validation_groups) != int(data["validation_task_groups"])):
        raise ValueError("frozen task group counts differ from config")
    seed = int(config["seed"])
    torch.manual_seed(seed)
    random.seed(seed)
    torch.set_num_threads(4)
    if torch.cuda.is_initialized():
        raise RuntimeError("this frozen pilot config is CPU-only")
    state_training = torch.from_numpy(train["state_values"].copy()).float()
    state_mean = state_training.mean(dim=(0, 1))
    state_std = state_training.std(dim=(0, 1), unbiased=False).clamp_min(1e-5)

    output.mkdir(parents=True)
    shutil.copy2(config_path, output / "effective_config.yaml")
    shutil.copy2(Path(__file__), output / "executed_script.py")
    np.savez(output / "train_only_state_normalization.npz",
             mean=state_mean.numpy(), std=state_std.numpy())
    started = datetime.now(UTC).isoformat()
    training = config["training"]
    train_prior = _train_prior_metrics(train["targets"], validation)
    (output / "train-prior-validation-predictions.jsonl").write_text(
        json.dumps(train_prior["predictions"], separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    train_prior_summary = {
        key: value for key, value in train_prior.items() if key != "predictions"
    }
    arm_results = {}
    for mode in HAPTILE_MODES:
        arm_results[mode] = _fit_arm(
            mode, train, validation, state_mean, state_std, output,
            epochs=int(training["epochs"]), batch_size=int(training["batch_size"]),
            learning_rate=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
            min_delta=float(training["min_delta"]),
        )
    selected_mode = min(
        HAPTILE_MODES,
        key=lambda mode: (
            float(arm_results[mode]["best_validation"]["macro_task_group_nll"]),
            HAPTILE_MODES.index(mode),
        ),
    )
    selected = arm_results[selected_mode]
    validation_selection = {
        "rule": "minimum best validation macro task-group NLL across frozen modes",
        "validation_only": True,
        "selected_mode": selected_mode,
        "selected_epoch": selected["best_epoch"],
        "selected_checkpoint": selected["best_checkpoint"],
        "selected_checkpoint_sha256": selected["best_checkpoint_sha256"],
        "calibration_loaded": False,
        "sealed_audit_loaded": False,
        "product_promotion": False,
    }
    (output / "validation-model-selection.json").write_text(
        json.dumps(validation_selection, indent=2) + "\n", encoding="utf-8"
    )
    best_single_stream_mode = min(
        ("text_only", "state_only"),
        key=lambda mode: (
            float(arm_results[mode]["best_validation"]["macro_task_group_nll"]), mode
        ),
    )
    fusion_modes = ("vision_text", "tactile_text", "vision_tactile_text", "full")
    best_fusion_mode = min(
        fusion_modes,
        key=lambda mode: (
            float(arm_results[mode]["best_validation"]["macro_task_group_nll"]), mode
        ),
    )
    fusion_model = HapTileActionDecisionScorer()
    fusion_model.load_state_dict(
        torch.load(
            arm_results[best_fusion_mode]["best_checkpoint"],
            map_location="cpu", weights_only=True,
        )
    )
    fusion_validation = _evaluate(
        fusion_model, best_fusion_mode, validation, state_mean, state_std
    )
    stream_controls: dict[str, Any] = {
        "best_single_stream": {
            "mode": best_single_stream_mode,
            "metrics": {
                key: value for key, value in
                arm_results[best_single_stream_mode]["best_validation"].items()
            },
        },
        "best_fusion": {
            "mode": best_fusion_mode,
            "best_epoch": arm_results[best_fusion_mode]["best_epoch"],
            "metrics": {
                key: value for key, value in fusion_validation.items()
                if key != "predictions"
            },
        },
        "controls": {},
        "paired_intervals_vs_unperturbed_fusion": {},
        "stream_assignment": (
            "sorted task groups rotate to the next group; source rows cycle in order"
        ),
        "selection_affected_validation": True,
    }
    with (output / "validation-stream-control-predictions.jsonl").open(
        "w", encoding="utf-8"
    ) as prediction_stream:
        prediction_stream.write(json.dumps({
            "control": "unperturbed_fusion",
            "mode": best_fusion_mode,
            "predictions": fusion_validation["predictions"],
        }, separators=(",", ":")) + "\n")
        for control in (
            "missing_video", "missing_sensor", "task_group_shuffled_video",
            "task_group_shuffled_sensor",
        ):
            controlled_arrays = _apply_validation_control(
                validation, control, state_mean
            )
            controlled_metrics = _evaluate(
                fusion_model, best_fusion_mode, controlled_arrays,
                state_mean, state_std,
            )
            stream_controls["controls"][control] = {
                key: value for key, value in controlled_metrics.items()
                if key != "predictions"
            }
            stream_controls["paired_intervals_vs_unperturbed_fusion"][control] = (
                _paired_group_bootstrap(fusion_validation, controlled_metrics)
            )
            prediction_stream.write(json.dumps({
                "control": control,
                "mode": best_fusion_mode,
                "predictions": controlled_metrics["predictions"],
            }, separators=(",", ":")) + "\n")
    (output / "validation-stream-controls.json").write_text(
        json.dumps(stream_controls, indent=2) + "\n", encoding="utf-8"
    )
    calibration = _load_split(
        artifact_root, "calibration", int(data["calibration_examples"]), config
    )
    calibration_groups = set(calibration["task_groups"].tolist())
    if (train_groups & calibration_groups or validation_groups & calibration_groups
            or len(calibration_groups) != int(data["calibration_task_groups"])):
        raise ValueError("frozen train, validation and calibration task groups overlap")
    selected_model = HapTileActionDecisionScorer()
    selected_model.load_state_dict(
        torch.load(selected["best_checkpoint"], map_location="cpu", weights_only=True)
    )
    calibration_logits, calibration_targets = _predict_logits(
        selected_model, selected_mode, calibration, state_mean, state_std
    )
    temperature_config = config["evaluation"]["calibration_temperature"]
    temperature_fit = _fit_temperature(
        calibration_logits, calibration_targets, calibration["task_groups"].tolist(),
        minimum=float(temperature_config["minimum"]),
        maximum=float(temperature_config["maximum"]),
        points=int(temperature_config["points"]),
    )
    temperature = float(temperature_fit["temperature"])
    validation_logits, validation_targets = _predict_logits(
        selected_model, selected_mode, validation, state_mean, state_std
    )
    calibrated_validation = _metrics(
        validation_logits / temperature, validation_targets,
        validation["task_groups"].tolist(), validation["sample_paths"].tolist(),
    )
    raw_calibration = _metrics(
        calibration_logits, calibration_targets, calibration["task_groups"].tolist(),
        calibration["sample_paths"].tolist(),
    )
    calibrated_calibration = _metrics(
        calibration_logits / temperature, calibration_targets,
        calibration["task_groups"].tolist(), calibration["sample_paths"].tolist(),
    )
    calibration_report = {
        **temperature_fit,
        "raw_calibration_metrics_in_sample": {
            key: value for key, value in raw_calibration.items() if key != "predictions"
        },
        "temperature_scaled_calibration_metrics_in_sample": {
            key: value for key, value in calibrated_calibration.items()
            if key != "predictions"
        },
        "temperature_scaled_validation_metrics": {
            key: value for key, value in calibrated_validation.items()
            if key != "predictions"
        },
        "calibration_feature_file_sha256": sha256_file(
            artifact_root / "calibration-joined-v1" / "decision-features.npz"
        ),
    }
    (output / "selected-calibration-fit.json").write_text(
        json.dumps(calibration_report, indent=2) + "\n", encoding="utf-8"
    )
    for name, metrics in (
        ("selected-calibration-fitted-predictions.jsonl", calibrated_calibration),
        ("selected-validation-calibrated-predictions.jsonl", calibrated_validation),
    ):
        with (output / name).open("w", encoding="utf-8") as stream:
            stream.write(json.dumps({
                "selected_mode": selected_mode,
                "temperature": temperature,
                "predictions": metrics["predictions"],
            }, separators=(",", ":")) + "\n")
    selection = {
        "rule": "minimum best validation macro task-group NLL across frozen modes",
        "validation_only": True,
        "selected_mode": selected_mode,
        "selected_epoch": selected["best_epoch"],
        "selected_checkpoint": selected["best_checkpoint"],
        "selected_checkpoint_sha256": selected["best_checkpoint_sha256"],
        "calibration_loaded_after_validation_selection": True,
        "calibration_selection_used": False,
        "temperature": temperature,
        "sealed_audit_loaded": False,
        "product_promotion": False,
    }
    (output / "model-selection.json").write_text(
        json.dumps(selection, indent=2) + "\n", encoding="utf-8"
    )
    comparisons = {}
    baseline = min(("text_only", "state_only"), key=lambda mode: (
        float(arm_results[mode]["best_validation"]["macro_task_group_nll"]), mode
    ))
    for mode in ("vision_text", "tactile_text", "vision_tactile_text", "full"):
        comparisons[f"{mode}_vs_{baseline}"] = _paired_group_bootstrap(
            arm_results[mode]["best_validation"], arm_results[baseline]["best_validation"]
        )
    feature_hashes = {
        split: sha256_file(artifact_root / f"{split}-joined-v1" / "decision-features.npz")
        for split in ("train", "validation", "calibration")
    }
    script = Path(__file__).resolve()
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    report = {
        "status": "completed_validation_selected_haptile_pilot",
        "experiment_id": config["experiment_id"],
        "source_commit": source_commit,
        "config_path": str(config_path),
        "config_sha256": config_hash,
        "executed_script_sha256": sha256_file(script),
        "run_started_at_utc": started,
        "run_finished_at_utc": datetime.now(UTC).isoformat(),
        "data_root": str(artifact_root),
        "feature_hashes": feature_hashes,
        "train_examples": len(train["targets"]),
        "train_unique_task_groups": len(train_groups),
        "train_unique_media_assets": len(set(train["media_sha256"].tolist())),
        "validation_examples": len(validation["targets"]),
        "validation_unique_task_groups": len(validation_groups),
        "validation_unique_media_assets": len(set(validation["media_sha256"].tolist())),
        "calibration_examples": len(calibration["targets"]),
        "calibration_unique_task_groups": len(calibration_groups),
        "calibration_unique_media_assets": len(set(calibration["media_sha256"].tolist())),
        "source": "Voxel51/HapTile",
        "available_modalities": ["text", "vision", "tactile_video", "robot_state"],
        "audio_available": False,
        "epochs_per_mode": int(training["epochs"]),
        "optimizer_updates_by_mode": {
            mode: arm_results[mode]["optimizer_updates"] for mode in HAPTILE_MODES
        },
        "selected_by_validation": selection,
        "train_prior_validation": train_prior_summary,
        "train_class_support": np.bincount(
            train["targets"], minlength=len(ACTION_LABELS)
        ).tolist(),
        "validation_class_support": np.bincount(
            validation["targets"], minlength=len(ACTION_LABELS)
        ).tolist(),
        "calibration_class_support": np.bincount(
            calibration["targets"], minlength=len(ACTION_LABELS)
        ).tolist(),
        "calibration_fit": calibration_report,
        "validation_stream_controls": stream_controls,
        "arms": arm_results,
        "paired_group_bootstrap_vs_best_unimodal": comparisons,
        "train_state_normalization_sha256": sha256_file(
            output / "train_only_state_normalization.npz"
        ),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "torch_threads": torch.get_num_threads(),
            "device": "cpu",
            "cuda_used": False,
            "working_set_bytes_at_exit": psutil.Process().memory_info().rss,
            "paid_compute_cost": 0,
        },
        "all_four_modality_goal_evaluated": False,
        "sealed_audit_loaded": False,
        "quantization_performed": False,
        "product_promotion": False,
    }
    (output / "experiment-report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (output / "progress.json").write_text(json.dumps({
        "status": "complete", "completed_modes": list(HAPTILE_MODES),
        "selected_mode": selected_mode, "selected_epoch": selected["best_epoch"],
    }, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()
    report = run_experiment(args.artifact_root, args.output, args.config)
    print(json.dumps({
        "status": report["status"],
        "selected_by_validation": report["selected_by_validation"],
        "arms": {
            mode: {
                "best_epoch": arm["best_epoch"],
                "validation": arm["best_validation"],
            }
            for mode, arm in report["arms"].items()
        },
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
