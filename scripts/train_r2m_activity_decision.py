"""Train fixed-budget supplied-activity Decision readouts on frozen R2-M features."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from tiny_omni_decision.r2m_activity import (
    activity_metrics,
    same_participant_different_activity_sensor_donors,
    supported_activity_vocabulary,
    train_sensor_normalization,
)
from tiny_omni_decision.time_aware_gated_fusion import (
    TemporalFeatureStream,
    TimeAwareGatedFusion,
)

SEED = 17
SPLIT_SHA256 = "0f1539f3ca4fa59fb854f62def1c538e1f5851576444b678af4aa050a393a6f0"
FEATURE_INDEX_SHA256 = "61758bebaf53ad810737d31e7955820bf7094e2166460c5ce595ea0c475c6498"
PAIR_INDEX_SHA256 = "d1e1c74a1114fc0d071f71a50876b852e8d3fce8919fde9696368d4ebf365276"
METADATA_SHA256 = "faee790bfaa53016e7c92f0ac545f0a7dfe23cd126fa7e1287b10d8d8b0282c0"
CLASSIFIER_MODES = ("audio_only", "sensor_only", "audio_sensor_gated")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: dict[str, object]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def declared_feature_signal_splits(index: dict[str, object]) -> list[str]:
    splits = index.get("feature_signal_splits_loaded", ["train", "calibration", "validation"])
    allowed = {"train", "calibration", "validation"}
    if (
        not isinstance(splits, list)
        or not splits
        or any(not isinstance(split, str) for split in splits)
        or not set(splits) <= allowed
        or len(splits) != len(set(splits))
    ):
        raise ValueError("feature cache declares invalid signal-split coverage")
    return splits


def load_frozen_examples(
    cache_dir: Path,
    metadata_path: Path,
    *,
    expected_index_sha256: str = FEATURE_INDEX_SHA256,
) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    index_path = cache_dir / "feature-cache-index.json"
    if sha256_file(index_path) != expected_index_sha256:
        raise ValueError("feature-cache index differs from the frozen R2-M cache")
    metadata_bytes = metadata_path.read_bytes()
    if hashlib.sha256(metadata_bytes).hexdigest() != METADATA_SHA256:
        raise ValueError("allowed-trial metadata differs from the frozen pair index")
    metadata = json.loads(metadata_bytes)
    metadata_by_id = {
        f"{row['activity']}|participant={row['participant_id']}|trial={row['trial_id']}": row
        for row in metadata
    }
    if len(metadata_by_id) != len(metadata):
        raise ValueError("duplicate trial identity in allowed-trial metadata")

    index = json.loads(index_path.read_text(encoding="utf-8"))
    declared_feature_splits = declared_feature_signal_splits(index)
    archive_entries = index.get("archives")
    if not isinstance(archive_entries, list) or len(archive_entries) != 24:
        raise ValueError("frozen feature index must contain all 24 activity archives")
    indexed_by_name = {entry["activity_archive"][:-4]: entry for entry in archive_entries}
    features: dict[str, dict[str, object]] = {}
    feature_dir = cache_dir / "features"
    sidecars = sorted(feature_dir.glob("*.json"))
    if len(sidecars) != 24:
        raise ValueError("expected one sidecar per frozen activity archive")
    for sidecar_path in sidecars:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        activity = sidecar_path.stem
        if sidecar != indexed_by_name.get(activity):
            raise ValueError(f"feature sidecar differs from frozen index: {activity}")
        if sidecar.get("sealed_audit_accessed") is not False:
            raise ValueError("feature cache does not establish sealed-audit isolation")
        feature_path = sidecar_path.with_suffix(".npz")
        if sha256_file(feature_path) != sidecar.get("feature_sha256"):
            raise ValueError(f"feature data hash mismatch: {feature_path.name}")
        with np.load(feature_path, allow_pickle=False) as data:
            ids = data["sample_ids"].astype(str)
            audio_values = data["audio_values"]
            offsets = data["audio_offsets"]
            sensors = data["sensor_values"]
            durations = data["durations_seconds"]
            splits = data["splits"].astype(str)
            activities = data["activities"].astype(str)
            participant_ids = data["participant_ids"]
            trial_ids = data["trial_ids"]
            if not (len(ids) == len(splits) == len(activities) == len(sensors)
                    == len(durations) == len(participant_ids) == len(trial_ids)
                    == int(sidecar["samples"])):
                raise ValueError(f"feature arrays disagree on row count: {feature_path.name}")
            if offsets.shape != (len(ids) + 1,) or offsets[0] != 0:
                raise ValueError(f"invalid audio offsets: {feature_path.name}")
            if offsets[-1] != len(audio_values) or np.any(np.diff(offsets) <= 0):
                raise ValueError(f"invalid audio token spans: {feature_path.name}")
            for row_index, identifier in enumerate(ids):
                if identifier in features:
                    raise ValueError(f"duplicate feature ID {identifier}")
                metadata_row = metadata_by_id.get(identifier)
                if metadata_row is None:
                    raise ValueError(f"feature is absent from allowlisted metadata: {identifier}")
                split = str(splits[row_index])
                if split not in declared_feature_splits:
                    raise ValueError(f"feature signal split is outside cache scope: {split}")
                if (split != metadata_row["split"]
                        or str(activities[row_index]) != metadata_row["activity"]
                        or int(participant_ids[row_index]) != int(metadata_row["participant_id"])
                        or int(trial_ids[row_index]) != int(metadata_row["trial_id"])):
                    raise ValueError(f"feature identity/split mismatch: {identifier}")
                audio = audio_values[offsets[row_index]:offsets[row_index + 1]].copy()
                sensor = sensors[row_index].copy()
                duration = float(durations[row_index])
                if (audio.ndim != 2 or audio.shape[1] != 384 or sensor.shape != (128, 3)
                        or not math.isfinite(duration) or duration <= 0
                        or not np.isfinite(audio).all() or not np.isfinite(sensor).all()):
                    raise ValueError(f"invalid frozen feature tensors for {identifier}")
                features[identifier] = {
                    "sample_id": identifier,
                    "activity": str(activities[row_index]),
                    "split": split,
                    "participant_id": int(participant_ids[row_index]),
                    "trial_id": int(trial_ids[row_index]),
                    "duration": duration,
                    "audio": audio,
                    "sensor": sensor,
                }
    if len(features) != sum(int(row["samples"]) for row in archive_entries):
        raise ValueError("feature count differs from frozen feature index")
    return features, {
        "feature_cache_index_sha256": expected_index_sha256,
        "feature_assets": len(features),
        "feature_split_counts": dict(Counter(
            str(row["split"]) for row in features.values()
        )),
        "metadata_trials": len(metadata),
        "metadata_sha256": METADATA_SHA256,
        "unfeatured_allowed_trials": len(metadata_by_id.keys() - features.keys()),
        "pair_index_internal_sha256": PAIR_INDEX_SHA256,
        "sensor_tokenization": index.get("sensor_tokenization", "legacy-linear-points-128"),
        "feature_signal_splits_loaded": declared_feature_splits,
        "sealed_audit_loaded": False,
    }


def build_vocabulary(features: dict[str, dict[str, object]]) -> tuple[
    tuple[str, ...], dict[str, dict[str, int]], tuple[str, ...]
]:
    counts: dict[str, Counter[str]] = {
        split: Counter() for split in ("train", "calibration", "validation")
    }
    for row in features.values():
        counts[str(row["split"])][str(row["activity"])] += 1
    vocabulary = supported_activity_vocabulary(counts)
    all_labels = set().union(*(set(split_counts) for split_counts in counts.values()))
    excluded = tuple(sorted(all_labels - set(vocabulary)))
    serialized = {split: dict(sorted(row.items())) for split, row in counts.items()}
    return vocabulary, serialized, excluded


def make_model(mode: str, num_classes: int, device: torch.device):
    if mode == "audio_only":
        dimensions = {"audio": 384}
    elif mode == "sensor_only":
        dimensions = {"sensor": 3}
    elif mode == "audio_sensor_gated":
        dimensions = {"audio": 384, "sensor": 3}
    else:
        raise ValueError(f"unknown classifier mode {mode}")
    fusion = TimeAwareGatedFusion(dimensions, num_classes, width=96, heads=4)
    scorer = nn.Linear(96, 1)
    fusion.to(device)
    scorer.to(device)
    return fusion, scorer


def pack_batch(
    rows: list[dict[str, object]],
    vocabulary: tuple[str, ...],
    mode: str,
    *,
    class_order: np.ndarray | None = None,
    modality_condition: str = "normal",
    sensor_rows: list[dict[str, object]] | None = None,
    sensor_normalization: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[dict[str, TemporalFeatureStream], torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = len(rows)
    num_classes = len(vocabulary)
    if class_order is None:
        class_order = np.arange(num_classes, dtype=np.int64)
    class_to_index = {label: index for index, label in enumerate(vocabulary)}
    class_positions = {int(class_id): position for position, class_id in enumerate(class_order)}
    labels = torch.tensor([
        class_positions[class_to_index[str(row["activity"])]] for row in rows
    ], dtype=torch.long)
    query_eye = torch.eye(num_classes, dtype=torch.float32)[class_order]
    queries = query_eye.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
    option_valid = torch.ones(batch_size, num_classes, dtype=torch.bool)

    streams: dict[str, TemporalFeatureStream] = {}
    if "audio" in mode:
        audio_lengths = [len(np.asarray(row["audio"])) for row in rows]
        max_audio = max(audio_lengths)
        audio_values = torch.zeros(batch_size, max_audio, 384, dtype=torch.float32)
        audio_valid = torch.zeros(batch_size, max_audio, dtype=torch.bool)
        audio_times = torch.zeros(batch_size, max_audio, dtype=torch.float32)
        for index, row in enumerate(rows):
            audio = torch.from_numpy(np.asarray(row["audio"], dtype=np.float32))
            duration_ms = float(row["duration"]) * 1000.0
            audio_values[index, :len(audio)] = audio
            audio_valid[index, :len(audio)] = True
            audio_times[index, :len(audio)] = torch.linspace(0.0, duration_ms, len(audio))
        present = torch.ones(batch_size, dtype=torch.bool)
        if modality_condition == "without_audio":
            present.zero_()
            audio_valid.zero_()
        streams["audio"] = TemporalFeatureStream(
            audio_values, audio_times, audio_valid, present
        )

    if "sensor" in mode:
        sensor_source = sensor_rows if sensor_rows is not None else rows
        if len(sensor_source) != batch_size:
            raise ValueError("sensor replacement rows must match audio batch size")
        sensor_values = torch.stack([
            torch.from_numpy(np.asarray(row["sensor"], dtype=np.float32))
            for row in sensor_source
        ])
        if sensor_normalization is not None:
            mean = torch.as_tensor(sensor_normalization[0], dtype=torch.float32)
            std = torch.as_tensor(sensor_normalization[1], dtype=torch.float32)
            sensor_values = (sensor_values - mean) / std
        sensor_times = torch.stack([
            torch.linspace(0.0, float(row["duration"]) * 1000.0, 128)
            for row in sensor_source
        ])
        sensor_valid = torch.ones(batch_size, 128, dtype=torch.bool)
        present = torch.ones(batch_size, dtype=torch.bool)
        if modality_condition == "without_sensor":
            present.zero_()
            sensor_valid.zero_()
        streams["sensor"] = TemporalFeatureStream(
            sensor_values, sensor_times, sensor_valid, present
        )
    return streams, queries, labels, option_valid


def forward_batch(fusion, scorer, packed, device):
    streams, queries, labels, option_valid = packed
    device_streams = {
        name: TemporalFeatureStream(
            row.values.to(device), row.timestamps_ms.to(device),
            row.valid.to(device), row.present.to(device),
        )
        for name, row in streams.items()
    }
    states = fusion(device_streams, queries.to(device), option_valid.to(device))
    logits = scorer(states).squeeze(-1)
    return logits, labels.to(device)


def prediction_rows(rows, vocabulary, probabilities):
    predictions = probabilities.argmax(axis=1)
    return [
        {
            "sample_id": str(row["sample_id"]),
            "participant_id": int(row["participant_id"]),
            "trial_id": int(row["trial_id"]),
            "target": str(row["activity"]),
            "prediction": vocabulary[int(predictions[index])],
            "probabilities": {
                label: float(probabilities[index, class_index])
                for class_index, label in enumerate(vocabulary)
            },
        }
        for index, row in enumerate(rows)
    ]


def evaluate(
    fusion, scorer, rows, vocabulary, mode, device, batch_size,
    *, modality_condition="normal", sensor_replacement=None, candidate_order=None,
    sensor_normalization=None, allow_missing_classes=False,
):
    fusion.eval()
    scorer.eval()
    logits_parts: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start:start + batch_size]
            replacement_rows = (None if sensor_replacement is None else
                                sensor_replacement[start:start + len(batch_rows)])
            packed = pack_batch(
                batch_rows, vocabulary, mode, class_order=candidate_order,
                modality_condition=modality_condition,
                sensor_rows=replacement_rows,
                sensor_normalization=sensor_normalization,
            )
            logits, _ = forward_batch(fusion, scorer, packed, device)
            logits_parts.append(logits.detach().cpu().numpy())
    logits_np = np.concatenate(logits_parts)
    labels_np = np.asarray([vocabulary.index(str(row["activity"])) for row in rows],
                           dtype=np.int64)
    logits_np -= logits_np.max(axis=1, keepdims=True)
    probabilities = np.exp(logits_np)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    if candidate_order is not None:
        canonical_probabilities = np.empty_like(probabilities)
        canonical_probabilities[:, candidate_order] = probabilities
        probabilities = canonical_probabilities
    metrics = activity_metrics(
        probabilities, labels_np, vocabulary,
        allow_missing_classes=allow_missing_classes,
    )
    return metrics, probabilities, labels_np


def train_arm(
    mode, train_rows, calibration_rows, validation_rows, vocabulary, class_weights,
    output_dir, *, epochs, batch_size, learning_rate, weight_decay, device,
    sensor_normalization, modality_condition="normal",
    validation_allow_missing_classes=False,
):
    seed = SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    fusion, scorer = make_model(mode, len(vocabulary), device)
    parameter_count = sum(p.numel() for p in fusion.parameters()) + sum(
        p.numel() for p in scorer.parameters()
    )
    optimizer = torch.optim.AdamW(
        list(fusion.parameters()) + list(scorer.parameters()),
        lr=learning_rate, weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    history_path = output_dir / f"{mode}-training-history.jsonl"
    best_path = output_dir / f"{mode}-best-calibration-checkpoint.pt"
    final_path = output_dir / f"{mode}-final-checkpoint.pt"
    latest_path = output_dir / f"{mode}-latest-checkpoint.pt"
    best_macro_nll = float("inf")
    best_epoch = 0
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    updates = 0
    with history_path.open("w", encoding="utf-8", newline="\n") as history:
        for epoch in range(1, epochs + 1):
            fusion.train()
            scorer.train()
            permutation_rng = np.random.default_rng(seed + epoch * 1009)
            row_order = permutation_rng.permutation(len(train_rows))
            epoch_loss = 0.0
            epoch_correct = 0
            gradient_norm_sum = 0.0
            gradient_norm_max = 0.0
            epoch_updates = 0
            seen = 0
            for start in range(0, len(row_order), batch_size):
                indices = row_order[start:start + batch_size]
                batch_rows = [train_rows[int(index)] for index in indices]
                class_order = permutation_rng.permutation(len(vocabulary))
                packed = pack_batch(
                    batch_rows, vocabulary, mode, class_order=class_order,
                    modality_condition=modality_condition,
                    sensor_normalization=sensor_normalization,
                )
                logits, labels = forward_batch(fusion, scorer, packed, device)
                target_classes = torch.tensor([
                    vocabulary.index(str(row["activity"])) for row in batch_rows
                ], dtype=torch.long, device=device)
                per_example_loss = nn.functional.cross_entropy(
                    logits, labels, reduction="none"
                )
                sample_weights = weight_tensor[target_classes]
                loss = (per_example_loss * sample_weights).sum() / sample_weights.sum()
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at {mode} epoch {epoch}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    list(fusion.parameters()) + list(scorer.parameters()), 1.0
                )
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError(f"non-finite gradient at {mode} epoch {epoch}")
                gradient_norm_value = float(grad_norm.detach())
                gradient_norm_sum += gradient_norm_value
                gradient_norm_max = max(gradient_norm_max, gradient_norm_value)
                optimizer.step()
                updates += 1
                epoch_updates += 1
                epoch_loss += float(loss.detach()) * len(batch_rows)
                predicted_classes = class_order[logits.argmax(dim=1).detach().cpu().numpy()]
                expected_classes = np.asarray([
                    vocabulary.index(str(row["activity"])) for row in batch_rows
                ], dtype=np.int64)
                epoch_correct += int((predicted_classes == expected_classes).sum())
                seen += len(batch_rows)
            calibration_started = time.perf_counter()
            calibration, _, _ = evaluate(
                fusion, scorer, calibration_rows, vocabulary, mode, device, batch_size,
                modality_condition=modality_condition,
                sensor_normalization=sensor_normalization,
            )
            calibration_seconds = time.perf_counter() - calibration_started
            elapsed = time.perf_counter() - started
            row = {
                "epoch": epoch,
                "optimizer_updates": updates,
                "epoch_optimizer_updates": epoch_updates,
                "train_examples_consumed_cumulative": seen * epoch,
                "unique_train_examples_per_epoch": seen,
                "train_weighted_ce": epoch_loss / seen,
                "train_accuracy": epoch_correct / seen,
                "gradient_norm_mean": gradient_norm_sum / max(1, epoch_updates),
                "gradient_norm_max": gradient_norm_max,
                "calibration": calibration,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "calibration_seconds": calibration_seconds,
                "elapsed_seconds": elapsed,
            }
            history.write(json.dumps(row, sort_keys=True) + "\n")
            history.flush()
            temporary_latest = latest_path.with_suffix(".partial.pt")
            torch.save({
                "epoch": epoch,
                "mode": mode,
                "fusion": fusion.state_dict(),
                "scorer": scorer.state_dict(),
                "vocabulary": vocabulary,
                "seed": seed,
            }, temporary_latest)
            os.replace(temporary_latest, latest_path)
            if calibration["macro_nll"] < best_macro_nll:
                best_macro_nll = float(calibration["macro_nll"])
                best_epoch = epoch
                torch.save({
                    "epoch": epoch,
                    "mode": mode,
                    "fusion": fusion.state_dict(),
                    "scorer": scorer.state_dict(),
                    "calibration_metrics": calibration,
                    "vocabulary": vocabulary,
                    "seed": seed,
                }, best_path)
            scheduler.step()
    torch.save({
        "epoch": epochs,
        "mode": mode,
        "fusion": fusion.state_dict(),
        "scorer": scorer.state_dict(),
        "vocabulary": vocabulary,
        "seed": seed,
    }, final_path)

    selected = torch.load(best_path, map_location=device, weights_only=False)
    fusion.load_state_dict(selected["fusion"], strict=True)
    scorer.load_state_dict(selected["scorer"], strict=True)
    train_selected, _, _ = evaluate(
        fusion, scorer, train_rows, vocabulary, mode, device, batch_size,
        modality_condition=modality_condition,
        sensor_normalization=sensor_normalization,
    )
    calibration_selected, _, _ = evaluate(
        fusion, scorer, calibration_rows, vocabulary, mode, device, batch_size,
        modality_condition=modality_condition,
        sensor_normalization=sensor_normalization,
    )
    validation_selected, probabilities, labels = evaluate(
        fusion, scorer, validation_rows, vocabulary, mode, device, batch_size,
        modality_condition=modality_condition,
        sensor_normalization=sensor_normalization,
        allow_missing_classes=validation_allow_missing_classes,
    )
    if device.type == "cuda":
        peak_memory = int(torch.cuda.max_memory_allocated(device))
    else:
        peak_memory = 0
    summary = {
        "mode": mode,
        "modality_condition": modality_condition,
        "selected_epoch": best_epoch,
        "final_epoch": epochs,
        "optimizer_updates": updates,
        "parameter_count": parameter_count,
        "trainable_parameter_count": parameter_count,
        "train_examples_per_epoch": len(train_rows),
        "train_unique_assets": len({str(row["sample_id"]) for row in train_rows}),
        "validation_unique_assets": len({str(row["sample_id"]) for row in validation_rows}),
        "selected_train_metrics": train_selected,
        "selected_calibration_metrics": calibration_selected,
        "selected_validation_metrics": validation_selected,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_torch_cuda_bytes": peak_memory,
        "best_checkpoint_sha256": sha256_file(best_path),
        "final_checkpoint_sha256": sha256_file(final_path),
        "history_sha256": sha256_file(history_path),
    }
    predictions_path = output_dir / f"{mode}-validation-predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in prediction_rows(validation_rows, vocabulary, probabilities):
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    summary["validation_predictions_sha256"] = sha256_file(predictions_path)
    return summary, (fusion, scorer, probabilities, labels)


def run_smoke(
    rows, vocabulary, device, sensor_normalization, *,
    model_arms=CLASSIFIER_MODES, modality_condition="normal",
):
    if len(rows) < 2:
        raise ValueError("smoke requires at least two supported examples")
    results = {}
    for mode in model_arms:
        torch.manual_seed(SEED)
        fusion, scorer = make_model(mode, len(vocabulary), device)
        packed = pack_batch(
            rows[:2], vocabulary, mode, modality_condition=modality_condition,
            sensor_normalization=sensor_normalization,
        )
        logits, labels = forward_batch(fusion, scorer, packed, device)
        loss = nn.functional.cross_entropy(logits, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite smoke loss for {mode}")
        loss.backward()
        gradients = [parameter.grad for parameter in fusion.parameters()
                     if parameter.requires_grad]
        if not gradients or any(
            gradient is None or not torch.isfinite(gradient).all()
            for gradient in gradients
        ):
            raise FloatingPointError(f"missing/non-finite smoke gradients for {mode}")
        results[mode] = {
            "logits_shape": list(logits.shape),
            "loss": float(loss.detach()),
            "finite_gradients": True,
            "parameters": sum(parameter.numel() for parameter in fusion.parameters())
            + sum(parameter.numel() for parameter in scorer.parameters()),
        }
    return results


def source_provenance(repository: Path) -> dict[str, object]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repository, check=True,
        capture_output=True, text=True,
    ).stdout
    files = (
        Path(__file__),
        repository / "src/tiny_omni_decision/r2m_activity.py",
        repository / "src/tiny_omni_decision/time_aware_gated_fusion.py",
    )
    return {
        "base_commit": commit,
        "working_tree_clean": not bool(status.strip()),
        "working_tree_status_sha256": hashlib.sha256(status.encode()).hexdigest(),
        "source_file_sha256": {
            str(path.relative_to(repository)): sha256_file(path) for path in files
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--pair-index-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--model-arms", nargs="+", choices=CLASSIFIER_MODES,
                        default=list(CLASSIFIER_MODES))
    parser.add_argument("--modality-condition",
                        choices=("normal", "without_audio", "without_sensor"),
                        default="normal")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--preflight-output", type=Path)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    if (args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0
            or args.weight_decay < 0):
        raise SystemExit("epochs/batch size/LR must be positive; weight decay nonnegative")
    if not args.model_arms:
        raise SystemExit("at least one model arm is required")
    if args.modality_condition != "normal" and any(
        "audio" not in mode or "sensor" not in mode for mode in args.model_arms
    ):
        raise SystemExit("missing-modality conditions require the audio_sensor_gated arm")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit("refusing to overwrite a nonempty output directory")

    metadata_path = args.pair_index_dir / "allowed-trial-metadata.json"
    pair_index = json.loads((args.pair_index_dir / "pair-index.json").read_text())
    if pair_index.get("manifest_sha256") != PAIR_INDEX_SHA256:
        raise SystemExit("pair-index differs from the frozen R2-M pair index")
    if pair_index.get("sealed_audit_accessed") is not False:
        raise SystemExit("pair index does not establish sealed-audit isolation")
    pair_payload = {key: value for key, value in pair_index.items()
                    if key != "manifest_sha256"}
    if canonical_hash(pair_payload) != PAIR_INDEX_SHA256:
        raise SystemExit("pair-index internal digest mismatch")
    repository = Path(__file__).resolve().parents[1]
    split_manifest = json.loads((repository /
        "manifests/pretrained_reuse/r2m-audio-vibration-split-v2-20261120.json"
    ).read_text(encoding="utf-8"))
    if split_manifest.get("manifest_sha256") != SPLIT_SHA256:
        raise SystemExit("participant split differs from frozen R2-M split v2")
    split_payload = {key: value for key, value in split_manifest.items()
                     if key != "manifest_sha256"}
    if canonical_hash(split_payload) != SPLIT_SHA256:
        raise SystemExit("participant split internal digest mismatch")
    participants_to_split = {
        int(participant): split
        for split, participant_ids in split_manifest["splits"].items()
        for participant in participant_ids
    }
    metadata_rows = json.loads(metadata_path.read_text(encoding="utf-8"))
    for row in metadata_rows:
        if (participants_to_split.get(int(row["participant_id"])) != row["split"]
                or row["split"] not in {"train", "calibration", "validation"}):
            raise SystemExit("metadata violates frozen participant split boundaries")
    features, data_summary = load_frozen_examples(args.feature_cache, metadata_path)
    vocabulary, counts, excluded = build_vocabulary(features)
    rows_by_split: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in features.values():
        if str(row["activity"]) in vocabulary:
            rows_by_split[str(row["split"])].append(row)
    for rows in rows_by_split.values():
        rows.sort(key=lambda row: str(row["sample_id"]))
    if any(not rows_by_split[split] for split in ("train", "calibration", "validation")):
        raise SystemExit("train/calibration/validation must all contain supported classes")
    participants = {
        split: {int(row["participant_id"]) for row in features.values()
                if row["split"] == split}
        for split in ("train", "calibration", "validation")
    }
    if (participants["train"] & participants["calibration"]
            or participants["train"] & participants["validation"]
            or participants["calibration"] & participants["validation"]):
        raise SystemExit("participant overlap found in frozen development splits")
    train_counts = Counter(str(row["activity"]) for row in rows_by_split["train"])
    class_weights = np.asarray([
        len(rows_by_split["train"]) / (len(vocabulary) * train_counts[label])
        for label in vocabulary
    ], dtype=np.float32)
    class_weights /= class_weights.mean()
    sensor_mean, sensor_std = train_sensor_normalization([
        np.asarray(row["sensor"], dtype=np.float32)
        for row in rows_by_split["train"]
    ])
    sensor_normalization = (sensor_mean, sensor_std)
    effective_config = {
        "schema": "r2m_supplied_activity_decision_v1",
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repository, check=True,
            capture_output=True, text=True,
        ).stdout.strip(),
        "seed": SEED,
        "task": (
            "select the activity label matching paired audio and vibration "
            "among supplied options"
        ),
        "choices": list(vocabulary),
        "excluded_for_missing_split_coverage": list(excluded),
        "split_counts_all_labels": counts,
        "examples_by_split_used": {split: len(rows_by_split[split])
                                    for split in rows_by_split},
        "class_weights": {label: float(class_weights[index])
                          for index, label in enumerate(vocabulary)},
        "sensor_normalization": {
            "method": "per-axis z-score, population mean/std from supported train examples only",
            "training_examples": len(rows_by_split["train"]),
            "training_tokens_per_example": 128,
            "axis_order": ["x", "y", "z"],
            "mean": sensor_mean.tolist(),
            "standard_deviation": sensor_std.tolist(),
        },
        "model_arms": list(args.model_arms),
        "modality_condition": args.modality_condition,
        "fusion_width": 96,
        "fusion_heads": 4,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "optimizer": "AdamW",
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "scheduler": "CosineAnnealingLR over fixed epoch budget",
        "loss": "train-only inverse-frequency weighted cross entropy",
        "option_order": (
            "random candidate permutation per training batch; "
            "fixed canonical order in evaluation"
        ),
        "selector": "minimum calibration macro NLL across the fixed supported choice vocabulary",
        "early_stopping": False,
        "trainable_parameters": "per arm recorded after model construction",
        "checkpoint_initialization": (
            "fresh readout weights; frozen pretrained EfficientAT features"
        ),
        "data": data_summary,
        "participant_groups": {split: sorted(ids) for split, ids in participants.items()},
        "sealed_audit_loaded": False,
        "quantization": False,
        "source_provenance": source_provenance(repository),
    }
    encoded_config = json.dumps(effective_config, sort_keys=True, indent=2) + "\n"
    config_hash = hashlib.sha256(encoded_config.encode()).hexdigest()
    effective_config["effective_config_sha256_excluding_self"] = config_hash
    if args.preflight_only:
        preflight_text = json.dumps(effective_config, sort_keys=True, indent=2) + "\n"
        if args.preflight_output:
            args.preflight_output.parent.mkdir(parents=True, exist_ok=True)
            args.preflight_output.write_text(
                preflight_text, encoding="utf-8", newline="\n"
            )
            args.preflight_output.with_suffix(".sha256").write_text(
                hashlib.sha256(preflight_text.encode()).hexdigest() + "\n",
                encoding="ascii", newline="\n",
            )
        print(preflight_text, end="")
        return

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device)
        device_metadata = {
            "name": torch.cuda.get_device_name(device),
            "total_memory": torch.cuda.get_device_properties(device).total_memory,
        }
    else:
        device_metadata = {"name": "CPU"}
    if args.smoke_only:
        smoke = run_smoke(
            rows_by_split["train"], vocabulary, device, sensor_normalization,
            model_arms=args.model_arms,
            modality_condition=args.modality_condition,
        )
        print(json.dumps({
            "status": "PASS",
            "device": str(device),
            "device_metadata": device_metadata,
            "smoke": smoke,
            "output_directory_created": False,
            "training_started": False,
        }, sort_keys=True, indent=2))
        return
    effective_config["device"] = str(device)
    effective_config["device_metadata"] = device_metadata
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config_for_hash = dict(effective_config)
    config_for_hash.pop("effective_config_sha256_excluding_self", None)
    config_text = json.dumps(config_for_hash, sort_keys=True, indent=2) + "\n"
    effective_config["effective_config_sha256_excluding_self"] = hashlib.sha256(
        config_text.encode()
    ).hexdigest()
    config_text = json.dumps(effective_config, sort_keys=True, indent=2) + "\n"
    (args.output_dir / "effective-config.json").write_text(
        config_text, encoding="utf-8", newline="\n"
    )
    (args.output_dir / "effective-config.sha256").write_text(
        hashlib.sha256(config_text.encode()).hexdigest() + "\n",
        encoding="ascii", newline="\n",
    )
    (args.output_dir / "environment.json").write_text(json.dumps({
        "python": __import__("sys").version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "device_metadata": device_metadata,
    }, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    summaries = []
    started = time.perf_counter()
    validation_labels = np.asarray([
        vocabulary.index(str(row["activity"])) for row in rows_by_split["validation"]
    ], dtype=np.int64)
    uniform_probabilities = np.full(
        (len(validation_labels), len(vocabulary)), 1.0 / len(vocabulary),
        dtype=np.float64,
    )
    train_prior = np.asarray([train_counts[label] for label in vocabulary], dtype=np.float64)
    train_prior /= train_prior.sum()
    prior_probabilities = np.broadcast_to(
        train_prior, (len(validation_labels), len(vocabulary))
    ).copy()
    baselines = {
        "uniform_options": activity_metrics(
            uniform_probabilities, validation_labels, vocabulary
        ),
        "train_class_prior": activity_metrics(
            prior_probabilities, validation_labels, vocabulary
        ),
    }
    for mode in args.model_arms:
        summary, state = train_arm(
            mode, rows_by_split["train"], rows_by_split["calibration"],
            rows_by_split["validation"], vocabulary, class_weights,
            args.output_dir, epochs=args.epochs, batch_size=args.batch_size,
            learning_rate=args.learning_rate, weight_decay=args.weight_decay,
            device=device,
            sensor_normalization=sensor_normalization,
            modality_condition=args.modality_condition,
        )
        summaries.append(summary)
        fusion, scorer, normal_probabilities, val_labels = state
        if mode == "audio_sensor_gated" and args.modality_condition == "normal":
            # Test both explicit missing-input paths with the exact selected weights.
            for condition in ("without_audio", "without_sensor"):
                metrics, probabilities, _ = evaluate(
                    fusion, scorer, rows_by_split["validation"], vocabulary, mode,
                    device, args.batch_size, modality_condition=condition,
                    sensor_normalization=sensor_normalization,
                )
                path = args.output_dir / f"validation-control-{condition}.jsonl"
                with path.open("w", encoding="utf-8", newline="\n") as stream:
                    for row in prediction_rows(rows_by_split["validation"], vocabulary,
                                               probabilities):
                        stream.write(json.dumps(row, sort_keys=True) + "\n")
                summaries[-1].setdefault("validation_controls", {})[condition] = {
                    "metrics": metrics,
                    "predictions_sha256": sha256_file(path),
                }
            # Deterministic derangement checks whether the readout responds to the sensor stream.
            val_rows = rows_by_split["validation"]
            mismatched_sensor_rows = same_participant_different_activity_sensor_donors(
                val_rows, seed=SEED
            )
            metrics, probabilities, _ = evaluate(
                fusion, scorer, val_rows, vocabulary, mode, device, args.batch_size,
                sensor_replacement=mismatched_sensor_rows,
                sensor_normalization=sensor_normalization,
            )
            path = args.output_dir / "validation-control-cross-activity-sensor.jsonl"
            with path.open("w", encoding="utf-8", newline="\n") as stream:
                for row in prediction_rows(val_rows, vocabulary, probabilities):
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
            summaries[-1].setdefault("validation_controls", {})[
                "same_participant_different_activity_sensor"
            ] = {
                "metrics": metrics,
                "predictions_sha256": sha256_file(path),
                "replacement_derangement": (
                    "same-participant sensor donor permutation with a different activity "
                    "for every recipient; donors are shuffled within activity"
                ),
            }
            # Candidate permutation is a metamorphic runtime check, not a learned metric.
            perm = np.random.default_rng(SEED).permutation(len(vocabulary))
            _, permuted_probability, _ = evaluate(
                fusion, scorer, val_rows, vocabulary, mode, device, args.batch_size,
                candidate_order=perm, sensor_normalization=sensor_normalization,
            )
            permutation_delta = float(np.max(np.abs(
                normal_probabilities - permuted_probability
            )))
            if permutation_delta > 1e-6:
                raise RuntimeError(
                    f"candidate permutation changed probabilities by {permutation_delta}"
                )
            summaries[-1]["option_permutation_max_probability_delta"] = permutation_delta

    result = {
        "schema": "r2m_supplied_activity_decision_results_v1",
        "effective_config_sha256": hashlib.sha256(config_text.encode()).hexdigest(),
        "started_utc_epoch": time.time() - (time.perf_counter() - started),
        "elapsed_seconds_total": time.perf_counter() - started,
        "sealed_audit_loaded": False,
        "validation_baselines": baselines,
        "arms": summaries,
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
