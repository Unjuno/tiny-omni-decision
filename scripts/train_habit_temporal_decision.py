"""Fit fixed-budget text/video/state candidate scorers on the HABIT pilot."""

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
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
import transformers
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tiny_omni_decision.decision import FrozenFeatureCandidateScorer
from tiny_omni_decision.video_sensor_temporal_decision import VideoSensorTemporalDecisionScorer

TEXT_REVISION = "4ca70771034acceecb2e72475f72050fcdde4ddc"
TEXT_WEIGHTS_SHA256 = "cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433"
VIDEO_REVISION = "be719a406d563b66f0ac969e7c94bab8e997c81a"
VIDEO_CHECKPOINT_SHA256 = "61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29"
VIDEOPRISM_PARAMETERS = 114_365_184
TEXT_PARAMETERS = 17_389_824
SEED = 17
MODES = ("text_only", "video_only", "state_only", "video_state")
CONFIG_PATH = Path("configs/pretrained_reuse/habit_temporal_decision.yaml")
MAX_EPOCHS = 80
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MIN_DELTA = 1e-4


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metrics(logits: list[Tensor], targets: list[int], groups: list[str]) -> dict[str, Any]:
    if not logits or len(logits) != len(targets) or len(groups) != len(targets):
        raise ValueError("metrics require aligned nonempty logits, targets and groups")
    probabilities = [torch.softmax(row.float(), dim=-1).cpu() for row in logits]
    correct = [int(int(prob.argmax()) == target) for prob, target in zip(
        probabilities, targets, strict=True
    )]
    nll = [-float(prob[target].clamp_min(1e-12).log()) for prob, target in zip(
        probabilities, targets, strict=True
    )]
    brier = [float((prob - F.one_hot(torch.tensor(target), prob.numel())).square().sum())
             for prob, target in zip(probabilities, targets, strict=True)]
    confidence = [float(prob.max()) for prob in probabilities]
    ece = 0.0
    for index in range(15):
        lower, upper = index / 15, (index + 1) / 15
        in_bin = [i for i, value in enumerate(confidence)
                  if value >= lower and (value < upper or index == 14 and value <= upper)]
        if in_bin:
            mean_confidence = sum(confidence[i] for i in in_bin) / len(in_bin)
            mean_accuracy = sum(correct[i] for i in in_bin) / len(in_bin)
            ece += len(in_bin) / len(correct) * abs(mean_confidence - mean_accuracy)
    indices_by_group: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        indices_by_group[group].append(index)
    group_metrics = {}
    for group, indices in sorted(indices_by_group.items()):
        group_metrics[group] = {
            "count": len(indices),
            "accuracy": sum(correct[i] for i in indices) / len(indices),
            "nll": sum(nll[i] for i in indices) / len(indices),
            "brier": sum(brier[i] for i in indices) / len(indices),
        }
    return {
        "count": len(targets),
        "group_count": len(group_metrics),
        "accuracy": sum(correct) / len(correct),
        "nll": sum(nll) / len(nll),
        "brier": sum(brier) / len(brier),
        "ece_15_bins": ece,
        "macro_group_accuracy": sum(row["accuracy"] for row in group_metrics.values())
        / len(group_metrics),
        "macro_group_nll": sum(row["nll"] for row in group_metrics.values())
        / len(group_metrics),
        "macro_group_brier": sum(row["brier"] for row in group_metrics.values())
        / len(group_metrics),
        "per_group": group_metrics,
        "probabilities": [prob.tolist() for prob in probabilities],
    }


def _validate_package(
    package_dir: Path, text_model: Path
) -> tuple[list[dict], dict[str, np.ndarray]]:
    report = json.loads((package_dir / "package-report.json").read_text(encoding="utf-8"))
    examples_path = package_dir / "transition-examples.json"
    cache_path = package_dir / "habit-transition-features.npz"
    if (report.get("status") != "complete_text_aligned_frozen_feature_package"
            or report.get("sealed_audit_accessed") is not False
            or report.get("action_columns_read") is not False
            or report.get("training_performed") is not False
            or sha256_file(examples_path) != report.get("transition_examples_sha256")
            or sha256_file(cache_path) != report.get("feature_cache_sha256")):
        raise ValueError("input package hashes or status do not match the frozen report")
    if sha256_file(text_model / "model.safetensors") != TEXT_WEIGHTS_SHA256:
        raise ValueError("MiniLM-L3 weight hash differs from the pinned candidate")
    example_manifest = json.loads(examples_path.read_text(encoding="utf-8"))
    if (example_manifest.get("source_revision") != "bae97d251907743f8fb6db7e70c61d361df4dc1e"
            or example_manifest.get("sealed_audit_accessed") is not False
            or example_manifest.get("action_columns_read") is not False):
        raise ValueError("text examples include an unexpected source or audit scope")
    examples = example_manifest["records"]
    with np.load(cache_path, allow_pickle=False) as values:
        arrays = {name: values[name].copy() for name in values.files}
    expected_ids = [row["id"] for row in examples]
    if arrays.get("ids", np.array([])).tolist() != expected_ids:
        raise ValueError("feature ID order differs from aligned text examples")
    if len(examples) != 278 or arrays["video_features"].shape != (278, 8, 768):
        raise ValueError("frozen feature cache has unexpected record/shape count")
    if arrays["sensor_sequence"].shape != (278, 8, 14):
        raise ValueError("sensor feature cache has unexpected shape")
    if len({row["id"] for row in examples}) != len(examples):
        raise ValueError("example IDs are not unique")
    for index, row in enumerate(examples):
        if (len(row["options"]) < 2 or len(row["options"]) != arrays["option_count"][index]
                or not 0 <= row["target_index"] < len(row["options"])
                or row["target_index"] != arrays["target_index"][index]
                or len(set(row["options"])) != len(row["options"])
                or not row["question"].strip()):
            raise ValueError(f"candidate or target alignment failed for {row['id']}")
    train_groups = {row["sid"] for row in examples if row["split"] == "train"}
    validation_groups = {row["sid"] for row in examples if row["split"] == "validation"}
    train_media = {row["video_sha256"] for row in examples if row["split"] == "train"}
    validation_media = {row["video_sha256"] for row in examples if row["split"] == "validation"}
    if train_groups & validation_groups or train_media & validation_media:
        raise ValueError("task group or media identity crosses train/validation")
    if sorted({row["split"] for row in examples}) != ["train", "validation"]:
        raise ValueError("both train and validation rows are required")
    return examples, arrays


def _validate_config(config_path: Path, package_dir: Path, text_model: Path) -> tuple[dict, str]:
    import yaml

    config_bytes = config_path.read_bytes()
    config = yaml.safe_load(config_bytes)
    expected = {
        "experiment_id": "habit-temporal-decision-cpu-seed17-v0",
        "seed": SEED,
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("experiment config differs from the reviewed fixed schedule")
    expected_training = {
        "epochs": MAX_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "min_delta": MIN_DELTA,
        "early_stopping": False,
        "seed_per_arm": SEED,
    }
    if any(config["training"].get(key) != value
           for key, value in expected_training.items()):
        raise ValueError("training config differs from the reviewed fixed schedule")
    expected_readout = {
        "type": "VideoSensorTemporalDecisionScorer",
        "sensor_channels": 14,
        "option_features": "question_option_pair_4way",
        "context_width": 384,
        "sensor_hidden_size": 32,
        "fusion_width": 64,
        "attention_heads": 4,
        "scorer_hidden_width": 64,
        "modes": list(MODES),
    }
    if config.get("readout") != expected_readout:
        raise ValueError("readout config differs from the fixed architecture")
    if config["data"]["package_report_sha256"] != sha256_file(
        package_dir / "package-report.json"
    ):
        raise ValueError("config references a different feature package")
    if config["text_encoder"]["weights_sha256"] != sha256_file(
        text_model / "model.safetensors"
    ):
        raise ValueError("config references different MiniLM weights")
    cache_path = package_dir / "habit-transition-features.npz"
    with np.load(cache_path, allow_pickle=False) as cache:
        if list(cache["video_features"].shape) != [278, 8, 768]:
            raise ValueError("feature cache shape differs from the fixed protocol")
    if (config["data"].get("revision") != "bae97d251907743f8fb6db7e70c61d361df4dc1e"
            or config["data"].get("feature_cache_sha256") != sha256_file(cache_path)
            or config["video_encoder"].get("revision") != VIDEO_REVISION
            or config["video_encoder"].get("checkpoint_sha256") != VIDEO_CHECKPOINT_SHA256
            or config["video_encoder"].get("parameters") != VIDEOPRISM_PARAMETERS
            or config["text_encoder"].get("revision") != TEXT_REVISION
            or config["text_encoder"].get("parameters") != TEXT_PARAMETERS
            or config["evaluation"].get("metrics")
            != ["accuracy", "nll", "brier", "ece_15_bins"]
            or config["evaluation"].get("aggregation")
            != ["sample_weighted", "macro_task_group"]
            or config["evaluation"].get("paired_controls") != [
                "selected_video_state", "paired_video_missing", "paired_sensor_missing",
                "paired_video_cross_group_shuffle", "paired_sensor_cross_group_shuffle",
            ]
            or config["evaluation"].get("paired_group_bootstrap_draws") != 2000
            or config["prohibitions"].get("sealed_audit_loading") is not False
            or config["prohibitions"].get("test_split_loading") is not False):
        raise ValueError("config dataset/model/evaluation gate differs from the fixed protocol")
    return config, hashlib.sha256(config_bytes).hexdigest()


def _encode_text(
    examples: list[dict], model_path: Path, device: torch.device
) -> tuple[Tensor, list[Tensor], float]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    encoder = AutoModel.from_pretrained(model_path, local_files_only=True).eval().to(device)
    if encoder.config.hidden_size != 384:
        raise ValueError("pinned MiniLM-L3 width changed")
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    contexts = [f"Question: {row['question']}" for row in examples]
    option_rows = [[f"Candidate answer: {option}" for option in row["options"]]
                   for row in examples]
    flat_options = [option for row in option_rows for option in row]
    texts = contexts + flat_options
    vectors: list[Tensor] = []
    started = time.perf_counter()
    encoder.eval()
    with torch.inference_mode():
        for start in range(0, len(texts), 64):
            batch = tokenizer(
                texts[start:start + 64], max_length=128, truncation=True,
                padding=True, return_tensors="pt",
            ).to(device)
            hidden = encoder(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            vectors.extend(pooled.float().cpu().unbind())
    context = torch.stack(vectors[:len(examples)])
    option_vectors = vectors[len(examples):]
    rows = []
    offset = 0
    for index, options in enumerate(option_rows):
        current = option_vectors[offset:offset + len(options)]
        q = context[index].expand(len(options), -1)
        rows.append(torch.stack([
            torch.cat([q[option_index], value, (q[option_index] - value).abs(),
                       q[option_index] * value])
            for option_index, value in enumerate(current)
        ]))
        offset += len(options)
    del encoder
    return context, rows, time.perf_counter() - started


def _new_model(mode: str) -> nn.Module:
    if mode == "text_only":
        return FrozenFeatureCandidateScorer(1536, hidden_size=128)
    return VideoSensorTemporalDecisionScorer(
        video_feature_width=768,
        sensor_channels=14,
        option_width=1536,
        context_width=384,
        sensor_hidden_size=32,
        width=64,
        heads=4,
        scorer_hidden_width=64,
        time_unit_ms=1000.0,
    )


def _batch(
    indices: list[int], examples: list[dict], arrays: dict[str, np.ndarray],
    context: Tensor, option_rows: list[Tensor], sensor_mean: Tensor, sensor_std: Tensor,
    mode: str, *, shuffle_options: bool, rng: random.Random,
    video_donors: list[int] | None = None,
    sensor_donors: list[int] | None = None,
) -> tuple[Tensor, ...]:
    batch = len(indices)
    counts = [len(examples[index]["options"]) for index in indices]
    max_choices = max(counts)
    options = torch.zeros(batch, max_choices, 1536)
    option_valid = torch.zeros(batch, max_choices, dtype=torch.bool)
    targets = torch.empty(batch, dtype=torch.long)
    for row_index, example_index in enumerate(indices):
        row = examples[example_index]
        order = list(range(counts[row_index]))
        if shuffle_options:
            rng.shuffle(order)
        options[row_index, :len(order)] = option_rows[example_index][order]
        option_valid[row_index, :len(order)] = True
        targets[row_index] = order.index(int(row["target_index"]))
    if mode == "text_only":
        return (options, option_valid, targets)

    video = np.stack([
        arrays["video_features"][
            video_donors[index] if video_donors is not None else index
        ] for index in indices
    ]).astype(np.float32, copy=False)
    sensor = np.stack([
        arrays["sensor_sequence"][
            sensor_donors[index] if sensor_donors is not None else index
        ] for index in indices
    ]).astype(np.float32, copy=False)
    video_tensor = torch.from_numpy(video)
    sensor_tensor = (torch.from_numpy(sensor) - sensor_mean) / sensor_std
    video_times, sensor_times = [], []
    for index in indices:
        row = examples[index]
        video_times.append([time - row["context_end_ms"] for time in row["video_frame_times_ms"]])
        sensor_times.append([time - row["context_end_ms"] for time in row["sensor_frame_times_ms"]])
    video_times_tensor = torch.tensor(video_times, dtype=torch.float32)
    sensor_times_tensor = torch.tensor(sensor_times, dtype=torch.float32)
    present_video = mode in {"video_only", "video_state"}
    present_sensor = mode in {"state_only", "video_state"}
    video_valid = torch.full((batch, 8), present_video, dtype=torch.bool)
    sensor_valid = torch.full((batch, 8), present_sensor, dtype=torch.bool)
    video_present = torch.full((batch,), present_video, dtype=torch.bool)
    sensor_present = torch.full((batch,), present_sensor, dtype=torch.bool)
    if not present_video:
        video_tensor.zero_()
    if not present_sensor:
        sensor_tensor.zero_()
    return (
        video_tensor, video_times_tensor, video_valid, video_present,
        sensor_tensor, sensor_times_tensor, sensor_valid, sensor_present,
        options, option_valid, context[indices], targets,
    )


def _forward(model: nn.Module, mode: str, batch: tuple[Tensor, ...]) -> tuple[Tensor, Tensor]:
    if mode == "text_only":
        options, valid, targets = batch
        logits = model(options.reshape(-1, 1536)).reshape(options.shape[:2])
        return logits.masked_fill(~valid, float("-inf")), targets
    (video, video_times, video_valid, video_present,
     sensor, sensor_times, sensor_valid, sensor_present,
     options, option_valid, context, targets) = batch
    logits = model(
        video, video_times, video_valid, video_present,
        sensor, sensor_times, sensor_valid, sensor_present,
        options, option_valid, context_features=context,
    )
    return logits, targets


@torch.inference_mode()
def _evaluate(
    model: nn.Module, mode: str, indices: list[int], examples: list[dict],
    arrays: dict[str, np.ndarray], context: Tensor, option_rows: list[Tensor],
    sensor_mean: Tensor, sensor_std: Tensor, *, video_donors: list[int] | None = None,
    sensor_donors: list[int] | None = None,
    drop_video: bool = False, drop_sensor: bool = False,
) -> dict[str, Any]:
    model.eval()
    rows_logits, targets, groups = [], [], []
    generator = random.Random(SEED)
    for start in range(0, len(indices), BATCH_SIZE):
        selected = indices[start:start + BATCH_SIZE]
        batch = _batch(
            selected, examples, arrays, context, option_rows, sensor_mean, sensor_std,
            mode, shuffle_options=False, rng=generator, video_donors=video_donors,
            sensor_donors=sensor_donors,
        )
        if mode != "text_only" and (drop_video or drop_sensor):
            mutable = list(batch)
            if drop_video:
                mutable[2].zero_()
                mutable[3].zero_()
                mutable[0].zero_()
            if drop_sensor:
                mutable[6].zero_()
                mutable[7].zero_()
                mutable[4].zero_()
            batch = tuple(mutable)
        logits, batch_targets = _forward(model, mode, batch)
        rows_logits.extend([logits[row, :len(examples[index]["options"])].cpu()
                            for row, index in enumerate(selected)])
        targets.extend(batch_targets.tolist())
        groups.extend(str(examples[index]["sid"]) for index in selected)
    return _metrics(rows_logits, targets, groups)


def _fit_arm(
    mode: str, train_indices: list[int], validation_indices: list[int],
    examples: list[dict], arrays: dict[str, np.ndarray], context: Tensor,
    option_rows: list[Tensor], sensor_mean: Tensor, sensor_std: Tensor, output: Path,
) -> dict[str, Any]:
    torch.manual_seed(SEED)
    model = _new_model(mode)
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    rng = random.Random(SEED)
    history = []
    best_score = float("inf")
    best_epoch = 0
    best_path = output / f"{mode}-best.pt"
    last_path = output / f"{mode}-last.pt"
    started = time.perf_counter()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        order = train_indices.copy()
        rng.shuffle(order)
        train_losses = []
        for start in range(0, len(order), BATCH_SIZE):
            indices = order[start:start + BATCH_SIZE]
            batch = _batch(
                indices, examples, arrays, context, option_rows, sensor_mean, sensor_std,
                mode, shuffle_options=True, rng=rng,
            )
            optimizer.zero_grad(set_to_none=True)
            logits, targets = _forward(model, mode, batch)
            loss = F.cross_entropy(logits.float(), targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite loss: {mode} epoch {epoch}")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(f"nonfinite gradient: {mode} epoch {epoch}")
            optimizer.step()
            train_losses.append(float(loss.detach()))

        validation_metrics = _evaluate(
            model, mode, validation_indices, examples, arrays, context, option_rows,
            sensor_mean, sensor_std,
        )
        train_metrics = _evaluate(
            model, mode, train_indices, examples, arrays, context, option_rows,
            sensor_mean, sensor_std,
        )
        score = validation_metrics["macro_group_nll"]
        history.append({
            "epoch": epoch,
            "optimizer_updates": epoch * math.ceil(len(train_indices) / BATCH_SIZE),
            "train_cross_entropy_mean_batch": sum(train_losses) / len(train_losses),
            "train_accuracy": train_metrics["accuracy"],
            "train_nll": train_metrics["nll"],
            "validation_accuracy": validation_metrics["accuracy"],
            "validation_nll": validation_metrics["nll"],
            "validation_brier": validation_metrics["brier"],
            "validation_ece_15_bins": validation_metrics["ece_15_bins"],
            "validation_macro_group_nll": score,
            "selection_delta_from_best": best_score - score if math.isfinite(best_score) else None,
        })
        torch.save(model.state_dict(), last_path)
        if score < best_score - MIN_DELTA:
            best_score = score
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)

    if best_epoch == 0:
        raise RuntimeError(f"no finite validation-selected checkpoint for {mode}")
    model.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=True))
    selected = _evaluate(
        model, mode, validation_indices, examples, arrays, context, option_rows,
        sensor_mean, sensor_std,
    )
    (output / f"{mode}-history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    result = {
        "mode": mode,
        "best_epoch": best_epoch,
        "max_epochs_completed": MAX_EPOCHS,
        "best_validation_macro_group_nll": best_score,
        "trainable_parameters": trainable,
        "fixed_encoder_parameters": VIDEOPRISM_PARAMETERS + TEXT_PARAMETERS,
        "total_unique_prequant_parameters": VIDEOPRISM_PARAMETERS + TEXT_PARAMETERS + trainable,
        "wall_seconds": time.perf_counter() - started,
        "best_checkpoint_sha256": sha256_file(best_path),
        "last_checkpoint_sha256": sha256_file(last_path),
        "validation": selected,
        "predictions": _prediction_rows(
            selected, validation_indices, examples
        ),
    }
    model.cpu()
    del model, optimizer
    return result


def _prediction_rows(
    metrics: dict[str, Any], indices: list[int], examples: list[dict]
) -> list[dict]:
    rows = []
    for position, index in enumerate(indices):
        example = examples[index]
        probabilities = metrics["probabilities"][position]
        rows.append({
            "sample_id": example["id"],
            "task_group": example["sid"],
            "question": example["question"],
            "options": example["options"],
            "target_index": example["target_index"],
            "prediction_index": int(np.argmax(probabilities)),
            "probabilities": probabilities,
        })
    return rows


def _video_derangement(
    indices: list[int], examples: list[dict], seed: int = SEED
) -> list[int]:
    rng = random.Random(seed)
    groups = {index: examples[index]["sid"] for index in indices}
    group_counts = Counter(groups.values())
    if max(group_counts.values()) > len(indices) // 2:
        raise ValueError(
            "cross-group derangement is impossible: one task group exceeds half "
            f"of the rows ({max(group_counts.values())}/{len(indices)})"
        )

    # Random permutations almost never satisfy all 76 pair constraints when
    # validation groups are imbalanced. Find a true bipartite matching instead.
    recipients = indices.copy()
    rng.shuffle(recipients)
    recipients.sort(key=lambda index: group_counts[groups[index]], reverse=True)
    donor_to_recipient: dict[int, int] = {}

    def match(recipient: int, seen: set[int]) -> bool:
        donors = indices.copy()
        rng.shuffle(donors)
        for donor in donors:
            if donor in seen or groups[recipient] == groups[donor]:
                continue
            seen.add(donor)
            previous = donor_to_recipient.get(donor)
            if previous is None or match(previous, seen):
                donor_to_recipient[donor] = recipient
                return True
        return False

    if not all(match(recipient, set()) for recipient in recipients):
        raise RuntimeError("could not find a cross-task-group validation matching")
    mapping = list(range(len(examples)))
    for donor, recipient in donor_to_recipient.items():
        mapping[recipient] = donor
    return mapping


def _group_bootstrap_deltas(arms: dict[str, dict[str, Any]], draws: int = 2000) -> dict[str, Any]:
    reference = arms["video_state"]["validation"]["per_group"]
    groups = sorted(reference)
    if len(groups) < 2:
        raise ValueError("paired task-group bootstrap requires at least two groups")
    rng = random.Random(SEED)
    comparisons = {}
    for candidate in ("text_only", "video_only", "state_only"):
        target = arms[candidate]["validation"]["per_group"]
        if set(target) != set(groups):
            raise ValueError("candidate and paired validation groups differ")
        metric_intervals = {}
        for metric in ("accuracy", "nll"):
            values = []
            for _ in range(draws):
                sample = [rng.choice(groups) for _ in groups]
                values.append(sum(
                    reference[group][metric] - target[group][metric] for group in sample
                ) / len(sample))
            values.sort()
            metric_intervals[metric] = {
                "mean_group_delta": sum(
                    reference[group][metric] - target[group][metric] for group in groups
                ) / len(groups),
                "percentile_95_interval": [
                    values[int(0.025 * draws)],
                    values[min(draws - 1, int(0.975 * draws))],
                ],
            }
        comparisons[f"video_state_minus_{candidate}"] = metric_intervals
    return {
        "method": "paired task-group percentile bootstrap with replacement",
        "seed": SEED,
        "draws": draws,
        "task_groups": groups,
        "selection_affected_validation": True,
        "comparisons": comparisons,
    }


def _control_metrics(
    validation_indices: list[int],
    examples: list[dict], arrays: dict[str, np.ndarray], context: Tensor,
    option_rows: list[Tensor], sensor_mean: Tensor, sensor_std: Tensor,
    output: Path,
) -> dict[str, Any]:
    model = _new_model("video_state")
    model.load_state_dict(torch.load(output / "video_state-best.pt", map_location="cpu",
                                    weights_only=True))
    shuffled = _video_derangement(validation_indices, examples)
    controls = {
        "selected_paired": _evaluate(
            model, "video_state", validation_indices, examples, arrays, context,
            option_rows, sensor_mean, sensor_std,
        ),
        "paired_video_missing": _evaluate(
            model, "video_state", validation_indices, examples, arrays, context,
            option_rows, sensor_mean, sensor_std, drop_video=True,
        ),
        "paired_sensor_missing": _evaluate(
            model, "video_state", validation_indices, examples, arrays, context,
            option_rows, sensor_mean, sensor_std, drop_sensor=True,
        ),
        "paired_video_cross_group_shuffle": _evaluate(
            model, "video_state", validation_indices, examples, arrays, context,
            option_rows, sensor_mean, sensor_std, video_donors=shuffled,
        ),
        "paired_sensor_cross_group_shuffle": _evaluate(
            model, "video_state", validation_indices, examples, arrays, context,
            option_rows, sensor_mean, sensor_std,
            sensor_donors=_video_derangement(validation_indices, examples, seed=SEED + 1),
        ),
    }
    return controls


def run_experiment(
    package_dir: Path, text_model: Path, output: Path, config_path: Path = CONFIG_PATH
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    run_started_at = datetime.now(UTC).isoformat()
    torch.manual_seed(SEED)
    random.seed(SEED)
    torch.set_num_threads(4)
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA initialized unexpectedly in this CPU-only process")
    config, config_sha256 = _validate_config(config_path, package_dir, text_model)
    examples, arrays = _validate_package(package_dir, text_model)
    text_started = time.perf_counter()
    context, option_rows, text_feature_seconds = _encode_text(
        examples, text_model, torch.device("cpu")
    )
    split = [row["split"] for row in examples]
    train_indices = [i for i, value in enumerate(split) if value == "train"]
    validation_indices = [i for i, value in enumerate(split) if value == "validation"]
    if (len(train_indices), len(validation_indices)) != (202, 76):
        raise ValueError("unexpected frozen split example counts")
    train_sensor = torch.from_numpy(arrays["sensor_sequence"][train_indices]).float()
    sensor_mean = train_sensor.mean(dim=(0, 1))
    sensor_std = train_sensor.std(dim=(0, 1), unbiased=False).clamp_min(1e-5)
    output.mkdir(parents=True)
    script_path = Path(__file__).resolve()
    shutil.copy2(script_path, output / "executed_script.py")
    shutil.copy2(config_path, output / "effective_config.yaml")
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=script_path.parents[1], text=True
    ).strip()
    torch.save({"context": context, "options": option_rows}, output / "frozen-text-features.pt")

    results = {}
    for mode in MODES:
        results[mode] = _fit_arm(
            mode, train_indices, validation_indices, examples, arrays, context,
            option_rows, sensor_mean, sensor_std, output,
        )

    controls = _control_metrics(
        validation_indices, examples, arrays, context,
        option_rows, sensor_mean, sensor_std, output,
    )
    split_group_counts = {
        name: len({examples[index]["sid"] for index in ids})
        for name, ids in (("train", train_indices), ("validation", validation_indices))
    }
    report = {
        "status": "completed_validation_selected_pilot",
        "experiment_id": "habit-temporal-decision-cpu-seed17-v0",
        "source_revision": "bae97d251907743f8fb6db7e70c61d361df4dc1e",
        "package_directory": str(package_dir),
        "package_report_sha256": sha256_file(package_dir / "package-report.json"),
        "features_sha256": sha256_file(package_dir / "habit-transition-features.npz"),
        "examples_sha256": sha256_file(package_dir / "transition-examples.json"),
        "text_model_revision": TEXT_REVISION,
        "text_model_weights_sha256": TEXT_WEIGHTS_SHA256,
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "source_commit": source_commit,
        "executed_script_sha256": sha256_file(script_path),
        "run_started_at_utc": run_started_at,
        "run_finished_at_utc": datetime.now(UTC).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu": platform.processor(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
            "torch_threads": torch.get_num_threads(),
            "device": "cpu",
            "cuda_initialized": torch.cuda.is_initialized(),
            "working_set_bytes_at_exit": psutil.Process().memory_info().rss,
        },
        "text_feature_seconds": text_feature_seconds,
        "experiment_wall_seconds": time.perf_counter() - text_started,
        "device": "cpu",
        "seed": SEED,
        "training_config": {
            "epochs": MAX_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "loss": "cross_entropy",
            "option_order_randomized_each_train_batch": True,
            "checkpoint_selection": "minimum_validation_macro_task_group_nll",
            "min_delta": MIN_DELTA,
            "early_stopping": False,
            "updates_per_arm": MAX_EPOCHS * math.ceil(len(train_indices) / BATCH_SIZE),
            "train_presentations_per_arm": MAX_EPOCHS * len(train_indices),
            "unique_train_examples_per_arm": len(train_indices),
            "repeat_presentations_after_first_per_arm": (MAX_EPOCHS - 1) * len(train_indices),
            "total_updates_all_arms": len(MODES) * MAX_EPOCHS
            * math.ceil(len(train_indices) / BATCH_SIZE),
            "total_train_presentations_all_arms": len(MODES) * MAX_EPOCHS
            * len(train_indices),
        },
        "split_counts": {
            "train_examples": len(train_indices),
            "validation_examples": len(validation_indices),
            "train_task_groups": split_group_counts["train"],
            "validation_task_groups": split_group_counts["validation"],
        },
        "sensor_normalization": {
            "fit_split": "train",
            "mean": sensor_mean.tolist(),
            "std": sensor_std.tolist(),
        },
        "arms": results,
        "paired_group_bootstrap": _group_bootstrap_deltas(results),
        "paired_controls": controls,
        "sealed_audit_accessed": False,
        "test_split_loaded": False,
        "training_performed": True,
        "quantization_performed": False,
        "interpretation": (
            "one seed and six held-out task groups on exposed development validation; "
            "pilot evidence only, not product quality or broad generalization"
        ),
    }
    (output / "predictions.json").write_text(json.dumps({
        mode: results[mode]["predictions"] for mode in MODES
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    report["predictions_sha256"] = sha256_file(output / "predictions.json")
    (output / "experiment-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def run_smoke(
    package_dir: Path, text_model: Path, output: Path, config_path: Path = CONFIG_PATH
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing smoke output: {output}")
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA initialized unexpectedly in this CPU-only process")
    _validate_config(config_path, package_dir, text_model)
    examples, arrays = _validate_package(package_dir, text_model)
    context, option_rows, text_seconds = _encode_text(
        examples, text_model, torch.device("cpu")
    )
    train_indices = [i for i, row in enumerate(examples) if row["split"] == "train"]
    train_sensor = torch.from_numpy(arrays["sensor_sequence"][train_indices]).float()
    sensor_mean = train_sensor.mean(dim=(0, 1))
    sensor_std = train_sensor.std(dim=(0, 1), unbiased=False).clamp_min(1e-5)
    indices = [0, 1]
    rows = {}
    for mode in MODES:
        model = _new_model(mode).eval()
        batch = _batch(
            indices, examples, arrays, context, option_rows, sensor_mean, sensor_std,
            mode, shuffle_options=False, rng=random.Random(SEED),
        )
        logits, _ = _forward(model, mode, batch)
        valid = batch[1] if mode == "text_only" else batch[9]
        if logits.shape != valid.shape or not torch.isfinite(logits[valid]).all():
            raise ValueError(f"invalid {mode} smoke logits")
        rows[mode] = {
            "logit_shape": list(logits.shape),
            "finite_valid_logits": True,
            "trainable_parameters": sum(
                parameter.numel() for parameter in model.parameters()
            ),
        }
    report = {
        "status": "PASS_CPU_HABIT_DECISION_READOUT_SMOKE",
        "records_checked": len(indices),
        "text_embedding_seconds": text_seconds,
        "device": "cpu",
        "arms": rows,
        "training_performed": False,
        "sealed_audit_accessed": False,
    }
    output.mkdir(parents=True)
    (output / "smoke-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--text-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    package_dir = args.package_dir.resolve(strict=True)
    text_model = args.text_model.resolve(strict=True)
    output = args.output_dir.resolve()
    if args.smoke_only:
        result = run_smoke(package_dir, text_model, output, args.config.resolve(strict=True))
        print(json.dumps(result, indent=2))
        return
    result = run_experiment(package_dir, text_model, output, args.config.resolve(strict=True))
    print(json.dumps({
        "status": result["status"],
        "experiment_id": result["experiment_id"],
        "arms": {name: {
            "best_epoch": row["best_epoch"],
            "validation": {key: row["validation"][key] for key in (
                "count", "group_count", "accuracy", "nll", "brier",
                "ece_15_bins", "macro_group_accuracy", "macro_group_nll",
            )},
        } for name, row in result["arms"].items()},
        "wall_seconds": result["experiment_wall_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
