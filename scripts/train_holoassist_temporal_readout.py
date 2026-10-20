"""Fit matched small temporal readouts on a frozen HoloAssist feature cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

# Prefer this checkout's package over a globally installed editable worktree.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from scripts.audit_holoassist_metadata import sha256_file  # noqa: E402
from tiny_omni_decision.temporal_decision import (  # noqa: E402
    CausalGatedTemporalDecision,
    TaskConditionedAVSensorTemporalDecision,
    TaskConditionedTemporalDecision,
)

CONFIG = Path("configs/pretrained_reuse/holoassist_temporal_readout_v1.json")
FEATURE_FIELDS = ("video", "audio", "text")
METADATA_FIELDS = (
    "target",
    "loss_mask",
    "sample_id",
    "session_id",
    "window_index",
    "media_sha256",
    "task_type",
)


def load_feature_split(path: Path, *, require_text: bool = True) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    feature_fields = set(FEATURE_FIELDS) if require_text else {"video", "audio"}
    required = {*feature_fields, *METADATA_FIELDS}
    if set(data) != required:
        raise ValueError(f"feature archive keys differ: expected={sorted(required)}")
    count = len(data["sample_id"])
    expected_dims = {"video": 768, "audio": 384, "text": 384}
    for name in feature_fields:
        width = expected_dims[name]
        if data[name].shape != (count, width) or data[name].dtype != np.float32:
            raise ValueError(f"invalid {name} feature shape/dtype")
        if not np.isfinite(data[name]).all():
            raise ValueError(f"non-finite {name} features")
    for name in (
        "target",
        "loss_mask",
        "sample_id",
        "session_id",
        "window_index",
        "media_sha256",
        "task_type",
    ):
        if data[name].shape != (count,):
            raise ValueError(f"invalid {name} array shape")
    if (
        not count
        or len(set(map(str, data["sample_id"]))) != count
        or any(not str(value) for value in data["sample_id"])
    ):
        raise ValueError("feature split must have unique, nonempty sample IDs")
    if data["target"].dtype.kind not in "iu" or data["loss_mask"].dtype != np.bool_:
        raise ValueError("targets must be integers and loss_mask must be boolean")
    if data["window_index"].dtype.kind not in "iu":
        raise ValueError("window indices must be integers")
    if any(not str(value) for value in data["session_id"]):
        raise ValueError("session IDs must be nonempty")
    if any(not str(value) for name in ("media_sha256", "task_type") for value in data[name]):
        raise ValueError("media identities and task types must be nonempty")
    if not set(np.unique(data["target"])).issubset({-1, 0, 1}):
        raise ValueError("targets must be -1 (unlabelled), 0, or 1")
    if not np.array_equal(data["loss_mask"], data["target"] >= 0):
        raise ValueError("loss mask must exactly identify labelled windows")
    if not data["loss_mask"].any():
        raise ValueError("feature split has no labelled windows")
    sessions: dict[str, list[int]] = defaultdict(list)
    for index, session in enumerate(data["session_id"]):
        sessions[str(session)].append(index)
    for session, indices in sessions.items():
        ordered = sorted(indices, key=lambda i: int(data["window_index"][i]))
        window_indices = [int(data["window_index"][i]) for i in ordered]
        if window_indices != list(range(len(indices))):
            raise ValueError(f"non-contiguous window indices in session {session}")
        if len({str(data["media_sha256"][index]) for index in indices}) != 1:
            raise ValueError(f"media identity changes within session {session}")
        if len({str(data["task_type"][index]) for index in indices}) != 1:
            raise ValueError(f"task type changes within session {session}")
    data["session_names"] = np.asarray(sorted(sessions), dtype=str)
    data["session_indices"] = [
        np.asarray(
            sorted(sessions[name], key=lambda i: int(data["window_index"][i])),
            dtype=np.int64,
        )
        for name in data["session_names"]
    ]
    return data


def validate_split_identity(
    train: dict[str, Any], validation: dict[str, Any]
) -> None:
    for field in ("sample_id", "session_id", "media_sha256"):
        overlap = set(map(str, train[field])) & set(map(str, validation[field]))
        if overlap:
            raise ValueError(f"train/validation {field} overlap: {sorted(overlap)[:3]}")


def attach_imu_features(
    data: dict[str, Any], feature_path: Path, *, split: str, receipt: dict[str, Any]
) -> dict[str, Any]:
    """Join flattened 1-second IMU windows only after exact ID and metadata checks."""
    expected_receipt = receipt.get("splits", {}).get(split, {})
    if (
        expected_receipt.get("sha256") != sha256_file(feature_path)
        or expected_receipt.get("sample_count") != len(data["sample_id"])
    ):
        raise ValueError(f"{split} IMU feature hash differs from its receipt")
    with np.load(feature_path, allow_pickle=False) as archive:
        if set(archive.files) != {
            "sample_id",
            "features",
            "flattened_features",
            "metadata_json",
        }:
            raise ValueError("unexpected IMU feature archive fields")
        sample_ids = archive["sample_id"]
        features = archive["flattened_features"]
        sequences = archive["features"]
        metadata = json.loads(str(archive["metadata_json"].item()))
    if (
        sample_ids.shape != data["sample_id"].shape
        or not np.array_equal(sample_ids, data["sample_id"])
        or features.shape != (len(sample_ids), 30 * 21)
        or features.dtype != np.float32
        or sequences.shape != (len(sample_ids), 30, 21)
        or sequences.dtype != np.float32
        or not np.array_equal(features, sequences.reshape(len(sample_ids), -1))
        or not np.isfinite(features).all()
        or len(metadata) != len(sample_ids)
    ):
        raise ValueError(f"{split} IMU feature IDs/order/shape differ from AV cache")
    fields = (
        "split",
        "id",
        "session_id",
        "window_index",
        "target",
        "loss_mask",
        "media_sha256",
        "task_type",
    )
    sample_id_hash = hashlib.sha256("\n".join(map(str, sample_ids)).encode()).hexdigest()
    if expected_receipt.get("unique_sample_id_order_sha256") != sample_id_hash:
        raise ValueError(f"{split} IMU sample ID hash differs from its receipt")
    for index, row in enumerate(metadata):
        target = -1 if row["target"] is None else int(row["target"])
        expected = {
            "split": split,
            "id": str(data["sample_id"][index]),
            "session_id": str(data["session_id"][index]),
            "window_index": int(data["window_index"][index]),
            "target": int(data["target"][index]),
            "loss_mask": bool(data["loss_mask"][index]),
            "media_sha256": str(data["media_sha256"][index]),
            "task_type": str(data["task_type"][index]),
        }
        observed = {field: target if field == "target" else row[field] for field in fields}
        if observed != expected:
            raise ValueError(f"{split} IMU metadata differs from the AV corpus at row {index}")
    joined = dict(data)
    joined["sensor"] = features.copy()
    if joined["sensor"].shape[1] != 630:
        raise ValueError("IMU sensor feature dimension must be 630")
    return joined


def validate_feature_receipt(
    receipt: dict[str, Any],
    *,
    train_path: Path,
    validation_path: Path,
    train: dict[str, Any],
    validation: dict[str, Any],
    expected_encoders: dict[str, dict[str, Any]],
) -> None:
    if receipt.get("status") != "PASS_HOLOASSIST_FROZEN_FEATURE_CACHE":
        raise ValueError("a passing frozen feature-cache receipt is required")
    expected_hashes = {
        "train": sha256_file(train_path),
        "validation": sha256_file(validation_path),
    }
    if receipt.get("packed_feature_sha256") != expected_hashes:
        raise ValueError("packed feature files do not match their receipt hashes")
    for split, data in (("train", train), ("validation", validation)):
        ids_hash = hashlib.sha256("\n".join(map(str, data["sample_id"])).encode()).hexdigest()
        if receipt.get("report", {}).get(split, {}).get("ordered_sample_ids_sha256") != ids_hash:
            raise ValueError(f"{split} sample IDs do not match the feature receipt")
    if receipt.get("train_validation_session_overlap") != 0:
        raise ValueError("feature receipt reports train/validation session overlap")
    if receipt.get("train_validation_media_overlap") != 0:
        raise ValueError("feature receipt reports train/validation media overlap")
    if receipt.get("test_data_loaded") is not False:
        raise ValueError("test data must not be present in a development feature receipt")
    if set(receipt.get("input_feature_banks", {})) != set(expected_encoders):
        raise ValueError("receipt feature-bank set differs from the explicitly expected modalities")
    for modality, expected in expected_encoders.items():
        recorded = receipt["input_feature_banks"][modality]["model"]
        fields = (
            "model_id",
            "revision",
            "weights_sha256",
            "loaded_weights_sha256",
            "processor_sha256",
            "upstream_source_commit",
            "preprocessing",
        )
        for field in fields:
            if field in expected and recorded.get(field) != expected[field]:
                raise ValueError(f"{modality} model {field} differs from frozen config")


def collate_sessions(
    data: dict[str, Any],
    selected: list[int],
    device: torch.device,
    *,
    include_text: bool = True,
    include_sensor: bool = False,
) -> tuple[torch.Tensor, ...]:
    groups = [data["session_indices"][index] for index in selected]
    lengths = torch.tensor([len(group) for group in groups], dtype=torch.int64, device=device)
    steps = int(lengths.max().item())
    batch = len(groups)
    video = torch.zeros(batch, steps, 768, dtype=torch.float32, device=device)
    audio = torch.zeros(batch, steps, 384, dtype=torch.float32, device=device)
    text = torch.zeros(batch, 384, dtype=torch.float32, device=device)
    sensor = (
        torch.zeros(batch, steps, 630, dtype=torch.float32, device=device)
        if include_sensor
        else None
    )
    targets = torch.full((batch, steps), -1, dtype=torch.int64, device=device)
    loss_mask = torch.zeros(batch, steps, dtype=torch.bool, device=device)
    for row, group in enumerate(groups):
        length = len(group)
        video[row, :length] = torch.from_numpy(data["video"][group]).to(device)
        audio[row, :length] = torch.from_numpy(data["audio"][group]).to(device)
        if include_sensor:
            assert sensor is not None
            sensor[row, :length] = torch.from_numpy(data["sensor"][group]).to(device)
        if include_text:
            text[row] = torch.from_numpy(data["text"][group[0]]).to(device)
        targets[row, :length] = torch.from_numpy(data["target"][group]).to(device)
        loss_mask[row, :length] = torch.from_numpy(data["loss_mask"][group]).to(device)
        if include_text:
            repeated_text = np.repeat(data["text"][group[:1]], length, axis=0)
            if not np.array_equal(data["text"][group], repeated_text):
                raise ValueError(
                    f"text feature changes within session {data['session_id'][group[0]]}"
                )
    if include_sensor:
        assert sensor is not None
        return video, audio, text, sensor, lengths, targets, loss_mask
    return video, audio, text, lengths, targets, loss_mask


def metric_summary(
    logits: torch.Tensor, targets: torch.Tensor, bins: int = 15
) -> dict[str, float | int | None]:
    if logits.ndim != 2 or logits.shape != (targets.numel(), 2) or not targets.numel():
        raise ValueError("metrics require nonempty [N,2] logits and [N] targets")
    if bins < 1 or not torch.all((targets == 0) | (targets == 1)):
        raise ValueError("binary targets and a positive ECE bin count are required")
    probabilities = torch.softmax(logits.float(), dim=-1)
    if not torch.isfinite(probabilities).all():
        raise ValueError("non-finite evaluation probabilities")
    prediction = probabilities.argmax(dim=-1)
    confidence = probabilities.max(dim=-1).values
    correct = prediction.eq(targets)
    positive = targets == 1
    negative = targets == 0
    positive_recall = (
        float((prediction[positive] == 1).float().mean().item())
        if positive.any()
        else None
    )
    negative_recall = (
        float((prediction[negative] == 0).float().mean().item())
        if negative.any()
        else None
    )
    nll = F.cross_entropy(logits.float(), targets, reduction="none")
    one_hot = F.one_hot(targets, num_classes=2).float()
    brier = (probabilities - one_hot).square().sum(dim=-1)
    ece = torch.zeros((), dtype=torch.float32, device=logits.device)
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        selected = (confidence >= lower) & (
            (confidence < upper) if index + 1 < bins else (confidence <= upper)
        )
        if selected.any():
            ece += selected.float().mean() * (
                confidence[selected].mean() - correct[selected].float().mean()
            ).abs()
    return {
        "count": int(targets.numel()),
        "accuracy": float(correct.float().mean().item()),
        "positive_count": int(positive.sum().item()),
        "negative_count": int(negative.sum().item()),
        "positive_prevalence": float(positive.float().mean().item()),
        "sensitivity": positive_recall,
        "specificity": negative_recall,
        "balanced_accuracy": (
            (positive_recall + negative_recall) / 2
            if positive_recall is not None and negative_recall is not None
            else None
        ),
        "nll": float(nll.mean().item()),
        "brier": float(brier.mean().item()),
        "ece": float(ece.item()),
        "mean_confidence": float(confidence.mean().item()),
    }


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _evaluate(
    model: (
        CausalGatedTemporalDecision
        | TaskConditionedTemporalDecision
        | TaskConditionedAVSensorTemporalDecision
    ),
    data: dict[str, Any],
    device: torch.device,
    batch_sessions: int,
    ece_bins: int,
    task_type_to_id: dict[str, int] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    logits_rows, target_rows, task_type_rows, prediction_rows = [], [], [], []
    order = sorted(range(len(data["session_names"])), key=lambda i: str(data["session_names"][i]))
    with torch.inference_mode():
        for start in range(0, len(order), batch_sessions):
            selected = order[start : start + batch_sessions]
            batch = collate_sessions(
                data,
                selected,
                device,
                include_text=task_type_to_id is None,
                include_sensor="sensor" in data,
            )
            if "sensor" in data:
                video, audio, text, sensor, lengths, targets, mask = batch
            else:
                video, audio, text, lengths, targets, mask = batch
            if task_type_to_id is None:
                output = model(video, audio, text, lengths=lengths)
            else:
                task_ids = torch.tensor(
                    [
                        task_type_to_id[
                            str(data["task_type"][data["session_indices"][position][0]])
                        ]
                        for position in selected
                    ],
                    dtype=torch.int64,
                    device=device,
                )
                if isinstance(model, TaskConditionedAVSensorTemporalDecision):
                    output = model(video, audio, sensor, task_ids, lengths=lengths)
                else:
                    output = model(video, audio, task_ids, lengths=lengths)
            batch, steps = targets.shape
            for row in range(batch):
                valid_count = int(lengths[row].item())
                for step in range(valid_count):
                    if not bool(mask[row, step].item()):
                        continue
                    target = int(targets[row, step].item())
                    task_type = str(
                        data["task_type"][
                            data["session_indices"][selected[row]][step]
                        ]
                    )
                    scores = output.logits[row, step].detach().float().cpu()
                    probabilities = torch.softmax(scores, dim=-1).tolist()
                    sample_id = str(data["sample_id"][
                        data["session_indices"][selected[row]][step]
                    ])
                    logits_rows.append(scores)
                    target_rows.append(target)
                    task_type_rows.append(task_type)
                    prediction = {
                        "id": sample_id,
                        "session_id": str(data["session_names"][selected[row]]),
                        "task_type": task_type,
                        "window_index": step,
                        "target": target,
                        "prediction": int(np.argmax(probabilities)),
                        "probabilities": probabilities,
                    }
                    if isinstance(model, TaskConditionedAVSensorTemporalDecision):
                        prediction["modality_weights"] = (
                            output.modality_weights[row, step].detach().float().cpu().tolist()
                        )
                    prediction_rows.append(prediction)
    summary = metric_summary(
        torch.stack(logits_rows).to(device),
        torch.tensor(target_rows, dtype=torch.int64, device=device),
        bins=ece_bins,
    )
    task_type_metrics = {}
    for task_type in sorted(set(task_type_rows)):
        positions = [
            index for index, value in enumerate(task_type_rows) if value == task_type
        ]
        task_type_metrics[task_type] = metric_summary(
            torch.stack([logits_rows[index] for index in positions]).to(device),
            torch.tensor(
                [target_rows[index] for index in positions],
                dtype=torch.int64,
                device=device,
            ),
            bins=ece_bins,
        )
    summary["by_task_type"] = task_type_metrics
    summary["task_type_macro"] = {
        metric: float(np.mean([values[metric] for values in task_type_metrics.values()]))
        for metric in ("accuracy", "nll", "brier", "ece")
    }
    return summary, prediction_rows


def train_arm(
    *,
    arm: str,
    train: dict[str, np.ndarray],
    validation: dict[str, np.ndarray],
    config: dict[str, Any],
    device: torch.device,
    output: Path,
    task_type_to_id: dict[str, int] | None = None,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite readout attempt: {output}")
    sensor_model = "sensor" in train
    if sensor_model != ("sensor" in validation):
        raise ValueError("train and validation must both contain the same modality set")
    if sensor_model:
        for split_name, data in (("train", train), ("validation", validation)):
            sensor = data["sensor"]
            if (
                sensor.shape != (len(data["sample_id"]), 630)
                or sensor.dtype != np.float32
                or not np.isfinite(sensor).all()
            ):
                raise ValueError(f"invalid {split_name} IMU feature matrix")
    valid_arms = {"av", "sensor_only", "avs"} if sensor_model else {
        "video_only",
        "audio_only",
        "joint",
    }
    if arm not in valid_arms:
        raise ValueError(f"unknown modality arm: {arm}")
    if sensor_model and task_type_to_id is None:
        raise ValueError("AV/sensor readouts require typed task conditioning")
    output.mkdir(parents=True)
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    if sensor_model:
        model = TaskConditionedAVSensorTemporalDecision(
            task_count=len(task_type_to_id),
            hidden_size=int(config["hidden_size"]),
            modality_mode=arm,
        ).to(device)
    elif task_type_to_id is None:
        model = CausalGatedTemporalDecision(
            hidden_size=int(config["hidden_size"]), modality_mode=arm
        ).to(device)
    else:
        model = TaskConditionedTemporalDecision(
            task_count=len(task_type_to_id),
            hidden_size=int(config["hidden_size"]),
            modality_mode=arm,
        ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    batch_size = int(config["batch_sessions"])
    max_steps = int(config["max_optimizer_steps"])
    eval_interval = int(config["evaluation_interval_steps"])
    ece_bins = int(config["ece_bins"])
    step = 0
    best_nll = math.inf
    best_step: int | None = None
    history: list[dict[str, Any]] = []
    train_loss_window: list[float] = []
    train_accuracy_window: list[float] = []
    gradient_norm_window: list[float] = []
    consumed_sample_ids: set[str] = set()
    consumed_session_ids: set[str] = set()
    consumed_media_hashes: set[str] = set()
    consumed_task_type_occurrences: Counter[str] = Counter()
    consumed_session_occurrences = 0
    consumed_window_occurrences = 0
    observed_parameters_with_gradients: set[int] = set()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    train_order = list(range(len(train["session_names"])))
    initial_eval_started = time.perf_counter()
    validation0, predictions0 = _evaluate(
        model, validation, device, batch_size, ece_bins, task_type_to_id
    )
    best_nll = float(validation0["nll"])
    best_step = 0
    history.append(
        {
            "step": 0,
            "validation": validation0,
            "evaluation_seconds": time.perf_counter() - initial_eval_started,
            "elapsed_seconds": 0.0,
        }
    )
    _write_json(output / "predictions-step-000000.json", predictions0)
    torch.save(model.state_dict(), output / "best-checkpoint.pt")
    _write_json(
        output / "best-checkpoint.json",
        {"step": best_step, "validation_nll": best_nll},
    )
    _write_json(output / "training-history.json", history)
    while step < max_steps:
        epoch_order = torch.randperm(len(train_order), generator=generator).tolist()
        for start in range(0, len(epoch_order), batch_size):
            if step >= max_steps:
                break
            selected = epoch_order[start : start + batch_size]
            consumed_session_occurrences += len(selected)
            for session_position in selected:
                indices = train["session_indices"][session_position]
                consumed_window_occurrences += len(indices)
                consumed_sample_ids.update(map(str, train["sample_id"][indices]))
                consumed_session_ids.add(str(train["session_names"][session_position]))
                consumed_media_hashes.update(map(str, train["media_sha256"][indices]))
                consumed_task_type_occurrences.update(map(str, train["task_type"][indices]))
            batch = collate_sessions(
                train,
                selected,
                device,
                include_text=task_type_to_id is None,
                include_sensor=sensor_model,
            )
            if sensor_model:
                video, audio, text, sensor, lengths, targets, loss_mask = batch
            else:
                video, audio, text, lengths, targets, loss_mask = batch
            model.train()
            optimizer.zero_grad(set_to_none=True)
            if task_type_to_id is None:
                output_batch = model(video, audio, text, lengths=lengths)
            else:
                task_ids = torch.tensor(
                    [
                        task_type_to_id[
                            str(train["task_type"][train["session_indices"][position][0]])
                        ]
                        for position in selected
                    ],
                    dtype=torch.int64,
                    device=device,
                )
                if isinstance(model, TaskConditionedAVSensorTemporalDecision):
                    output_batch = model(video, audio, sensor, task_ids, lengths=lengths)
                else:
                    output_batch = model(video, audio, task_ids, lengths=lengths)
            mask = loss_mask & (targets >= 0)
            if not mask.any():
                raise ValueError("training batch has no labelled windows")
            selected_logits = output_batch.logits[mask]
            selected_targets = targets[mask]
            loss = F.cross_entropy(selected_logits, selected_targets)
            if not torch.isfinite(loss):
                raise ValueError(f"non-finite loss at optimizer step {step + 1}")
            loss.backward()
            observed_parameters_with_gradients.add(
                sum(
                    parameter.numel()
                    for parameter in model.parameters()
                    if parameter.grad is not None
                )
            )
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if not torch.isfinite(grad_norm):
                raise ValueError(f"non-finite gradient norm at optimizer step {step + 1}")
            optimizer.step()
            step += 1
            gradient_norm_window.append(float(grad_norm.detach().item()))
            train_loss_window.append(float(loss.detach().item()))
            train_accuracy_window.append(
                float(selected_logits.detach().argmax(dim=-1).eq(selected_targets).float().mean().item())
            )
            if step % eval_interval == 0 or step == max_steps:
                evaluation_started = time.perf_counter()
                validation_metrics, predictions = _evaluate(
                    model, validation, device, batch_size, ece_bins, task_type_to_id
                )
                record = {
                    "step": step,
                    "epoch": round(step / max(1, math.ceil(len(train_order) / batch_size)), 4),
                    "rolling_train_ce": float(np.mean(train_loss_window)),
                    "rolling_train_accuracy": float(np.mean(train_accuracy_window)),
                    "rolling_train_update_count": len(train_loss_window),
                    "rolling_gradient_norm_mean": float(np.mean(gradient_norm_window)),
                    "consumed_session_occurrences": consumed_session_occurrences,
                    "unique_train_sessions_consumed": len(consumed_session_ids),
                    "unique_train_assets_consumed": len(consumed_media_hashes),
                    "unique_train_windows_consumed": len(consumed_sample_ids),
                    "train_window_occurrences": consumed_window_occurrences,
                    "repeated_train_window_occurrences": consumed_window_occurrences
                    - len(consumed_sample_ids),
                    "consumed_task_type_window_occurrences": dict(
                        sorted(consumed_task_type_occurrences.items())
                    ),
                    "validation": validation_metrics,
                    "evaluation_seconds": time.perf_counter() - evaluation_started,
                    "elapsed_seconds": time.perf_counter() - started,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "parameters_with_gradients": max(observed_parameters_with_gradients),
                }
                history.append(record)
                train_loss_window.clear()
                train_accuracy_window.clear()
                gradient_norm_window.clear()
                _write_json(output / f"predictions-step-{step:06d}.json", predictions)
                _write_json(output / "training-history.json", history)
                if float(validation_metrics["nll"]) < best_nll:
                    best_nll = float(validation_metrics["nll"])
                    best_step = step
                    temporary = output / "best-checkpoint.pt.tmp"
                    torch.save(model.state_dict(), temporary)
                    temporary.replace(output / "best-checkpoint.pt")
                    _write_json(
                        output / "best-checkpoint.json",
                        {"step": best_step, "validation_nll": best_nll},
                    )
        # A round-robin over shuffled sessions exhausts each epoch exactly once.
    final_path = output / "final-checkpoint.pt.tmp"
    torch.save(model.state_dict(), final_path)
    final_path.replace(output / "final-checkpoint.pt")
    final_metrics, final_predictions = _evaluate(
        model, validation, device, batch_size, ece_bins, task_type_to_id
    )
    _write_json(output / f"predictions-step-{step:06d}-final.json", final_predictions)
    best_state = torch.load(
        output / "best-checkpoint.pt", map_location=device, weights_only=True
    )
    model.load_state_dict(best_state, strict=True)
    reloaded_metrics, reloaded_predictions = _evaluate(
        model, validation, device, batch_size, ece_bins, task_type_to_id
    )
    selected_predictions_path = output / f"predictions-step-{best_step:06d}.json"
    selected_predictions = json.loads(selected_predictions_path.read_text(encoding="utf-8"))
    if len(selected_predictions) != len(reloaded_predictions):
        raise ValueError("best-checkpoint reload changed prediction count")
    max_probability_delta = 0.0
    for before, after in zip(selected_predictions, reloaded_predictions, strict=True):
        identity_fields = ("id", "session_id", "task_type", "target", "prediction")
        if any(before[field] != after[field] for field in identity_fields):
            raise ValueError("best-checkpoint reload changed validation decisions")
        max_probability_delta = max(
            max_probability_delta,
            float(
                np.max(
                    np.abs(
                        np.asarray(before["probabilities"], dtype=np.float64)
                        - np.asarray(after["probabilities"], dtype=np.float64)
                    )
                )
            ),
        )
    if max_probability_delta > 1e-6:
        raise ValueError("best-checkpoint reload probability parity failed")
    result = {
        "status": "complete_fixed_budget_readout_run",
        "arm": arm,
        "seed": seed,
        "optimizer_steps": step,
        "best_step": best_step,
        "best_validation_nll": best_nll,
        "best_validation_reloaded": reloaded_metrics,
        "best_reload_max_probability_delta": max_probability_delta,
        "final_validation": final_metrics,
        "elapsed_seconds": time.perf_counter() - started,
        "train_session_count": len(train["session_names"]),
        "validation_session_count": len(validation["session_names"]),
        "train_window_count": len(train["sample_id"]),
        "validation_window_count": len(validation["sample_id"]),
        "unique_train_assets": len(set(map(str, train["media_sha256"]))),
        "unique_validation_assets": len(set(map(str, validation["media_sha256"]))),
        "train_task_type_window_counts": dict(
            sorted(Counter(map(str, train["task_type"])).items())
        ),
        "validation_task_type_window_counts": dict(
            sorted(Counter(map(str, validation["task_type"])).items())
        ),
        "consumed_session_occurrences": consumed_session_occurrences,
        "unique_train_sessions_consumed": len(consumed_session_ids),
        "unique_train_assets_consumed": len(consumed_media_hashes),
        "unique_train_windows_consumed": len(consumed_sample_ids),
        "train_window_occurrences": consumed_window_occurrences,
        "repeated_train_window_occurrences": consumed_window_occurrences
        - len(consumed_sample_ids),
        "trainable_parameters": model.trainable_parameter_count,
        "parameters_with_gradients": max(observed_parameters_with_gradients),
        "text_features_used": task_type_to_id is None,
        "sensor_features_used": arm in {"sensor_only", "avs"},
        "task_type_vocabulary": task_type_to_id,
        "peak_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
        "peak_cuda_reserved_bytes": (
            torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
        ),
        "best_checkpoint_sha256": sha256_file(output / "best-checkpoint.pt"),
        "final_checkpoint_sha256": sha256_file(output / "final-checkpoint.pt"),
        "early_stopping": False,
    }
    _write_json(output / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--validation-features", type=Path, required=True)
    parser.add_argument("--feature-receipt", type=Path, required=True)
    parser.add_argument("--imu-train-features", type=Path)
    parser.add_argument("--imu-validation-features", type=Path)
    parser.add_argument("--imu-receipt", type=Path)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite readout output: {args.output_dir}")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    require_text = config.get("conditioning", "text") != "task_type_id"
    train = load_feature_split(args.train_features, require_text=require_text)
    validation = load_feature_split(args.validation_features, require_text=require_text)
    validate_split_identity(train, validation)
    feature_receipt = json.loads(args.feature_receipt.read_text(encoding="utf-8"))
    validate_feature_receipt(
        feature_receipt,
        train_path=args.train_features,
        validation_path=args.validation_features,
        train=train,
        validation=validation,
        expected_encoders=(
            {name: config["encoders"][name] for name in ("video", "audio")}
            if not require_text
            else config["encoders"]
        ),
    )
    sensor_fusion = bool(config.get("sensor_fusion", False))
    sensor_normalization = None
    imu_receipt = None
    if sensor_fusion:
        if not all(
            path is not None
            for path in (
                args.imu_train_features,
                args.imu_validation_features,
                args.imu_receipt,
            )
        ):
            raise ValueError("sensor_fusion config requires both IMU splits and their receipt")
        imu_receipt = json.loads(args.imu_receipt.read_text(encoding="utf-8"))
        av_window_receipts = {
            bank["model"]["av_data_provenance"]["windows_receipt_sha256"]
            for bank in feature_receipt.get("input_feature_banks", {}).values()
        }
        if (
            imu_receipt.get("sealed_audit_loaded") is not False
            or imu_receipt.get("future_sensor_samples_used") is not False
            or av_window_receipts != {imu_receipt.get("window_receipt_sha256")}
            or any(
                imu_receipt.get("splits", {}).get(split, {}).get(
                    "unique_sample_id_order_sha256"
                )
                != feature_receipt.get("report", {}).get(split, {}).get(
                    "ordered_sample_ids_sha256"
                )
                for split in ("train", "validation")
            )
        ):
            raise ValueError("IMU feature receipt is not cleanly bound to this frozen window set")
        train = attach_imu_features(
            train, args.imu_train_features, split="train", receipt=imu_receipt
        )
        validation = attach_imu_features(
            validation,
            args.imu_validation_features,
            split="validation",
            receipt=imu_receipt,
        )
        mean = train["sensor"].mean(axis=0, dtype=np.float64)
        std = train["sensor"].std(axis=0, dtype=np.float64)
        std = np.maximum(std, 1e-6)
        train["sensor"] = np.clip((train["sensor"] - mean) / std, -10.0, 10.0).astype(
            np.float32
        )
        validation["sensor"] = np.clip(
            (validation["sensor"] - mean) / std, -10.0, 10.0
        ).astype(np.float32)
        sensor_normalization = {"mean": mean.tolist(), "std": std.tolist()}
        if set(config.get("arms", [])) != {"av", "sensor_only", "avs"}:
            raise ValueError(
                "sensor-fusion run must compare exactly AV, sensor-only, and AV+sensor"
            )
    elif any(
        path is not None
        for path in (args.imu_train_features, args.imu_validation_features, args.imu_receipt)
    ):
        raise ValueError("IMU features may only be supplied by an explicit sensor_fusion config")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    repository = REPOSITORY_ROOT
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repository, check=True, capture_output=True
    ).stdout
    if dirty.strip():
        raise ValueError("commit the feature/training pipeline before a task-quality run")
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True)
    if sensor_normalization is not None:
        _write_json(args.output_dir / "imu-normalization.json", sensor_normalization)
    metadata = {
        "status": "running",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "command": [str(item) for item in __import__("sys").argv],
        "source_commit": source_commit,
        "decision_module_path": str(
            Path(sys.modules[TaskConditionedAVSensorTemporalDecision.__module__].__file__).resolve()
        ),
        "python_executable": sys.executable,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
        "config": config,
        "config_sha256": sha256_file(args.config),
        "train_features_sha256": sha256_file(args.train_features),
        "validation_features_sha256": sha256_file(args.validation_features),
        "feature_receipt_sha256": sha256_file(args.feature_receipt),
        "imu_train_features_sha256": (
            sha256_file(args.imu_train_features) if args.imu_train_features else None
        ),
        "imu_validation_features_sha256": (
            sha256_file(args.imu_validation_features) if args.imu_validation_features else None
        ),
        "imu_receipt_sha256": sha256_file(args.imu_receipt) if args.imu_receipt else None,
        "imu_normalization_sha256": (
            sha256_file(args.output_dir / "imu-normalization.json")
            if sensor_normalization is not None
            else None
        ),
        "encoder_pins": feature_receipt["input_feature_banks"],
        "train_sample_ids_sha256": hashlib.sha256(
            "\n".join(map(str, train["sample_id"])).encode()
        ).hexdigest(),
        "validation_sample_ids_sha256": hashlib.sha256(
            "\n".join(map(str, validation["sample_id"])).encode()
        ).hexdigest(),
        "sealed_audit_loaded": False,
    }
    _write_json(args.output_dir / "run-metadata.json", metadata)
    results = []
    task_type_to_id = None
    if config.get("conditioning", "text") == "task_type_id":
        task_types = sorted(set(map(str, train["task_type"])))
        validation_task_types = set(map(str, validation["task_type"]))
        if not validation_task_types <= set(task_types):
            raise ValueError("validation task type is absent from the training vocabulary")
        task_type_to_id = {task_type: index for index, task_type in enumerate(task_types)}
        metadata["conditioning"] = (
            "typed task category ID; text feature values not passed to model"
        )
        metadata["task_type_vocabulary"] = task_type_to_id
        _write_json(args.output_dir / "run-metadata.json", metadata)
    for arm in config["arms"]:
        result = train_arm(
            arm=arm,
            train=train,
            validation=validation,
            config=config,
            device=device,
            output=args.output_dir / arm,
            task_type_to_id=task_type_to_id,
        )
        results.append(result)
        metadata["results"] = results
        metadata["completed_arms"] = len(results)
        _write_json(args.output_dir / "run-metadata.json", metadata)
    metadata["status"] = "complete"
    metadata["finished_at_utc"] = datetime.now(UTC).isoformat()
    _write_json(args.output_dir / "run-metadata.json", metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
