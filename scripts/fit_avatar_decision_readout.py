"""Fit and compare small AVATAR frozen-feature Decision readouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

SEED = 17
SAMPLE_RATE = 32_000
AUDIO_WINDOW_SAMPLES = 2 * SAMPLE_RATE
HIDDEN_DIM = 64
MAX_EPOCHS = 50
PATIENCE = 6
MIN_DELTA = 0.001
BATCH_SIZE = 128
BOOTSTRAP_REPLICATES = 2_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _expected_audio_metadata(
    model_rows: list[dict[str, Any]],
) -> dict[str, tuple[float, tuple[int, int]]]:
    expected = {}
    for row in model_rows:
        key = row["audio_asset_key"]
        center = float(row["audio_window_center_seconds"])
        padding = (
            round(float(row["audio_window_left_padding_seconds"]) * SAMPLE_RATE),
            round(float(row["audio_window_right_padding_seconds"]) * SAMPLE_RATE),
        )
        previous = expected.setdefault(key, (center, padding))
        if previous != (center, padding):
            raise ValueError(f"conflicting frozen audio metadata for {key}")
    return expected


def _expected_video_timestamps(model_rows: list[dict[str, Any]]) -> dict[str, float]:
    expected = {}
    for row in model_rows:
        for option in row["options"]:
            key = option["video_asset_key"]
            timestamp = float(option["video_timestamp_seconds"])
            previous = expected.setdefault(key, timestamp)
            if abs(previous - timestamp) > 1e-9:
                raise ValueError(f"conflicting frozen video timestamp for {key}")
    return expected


def load_audio_feature_cache(
    path: Path, expected: dict[str, tuple[float, tuple[int, int]]]
) -> tuple[list[str], np.ndarray, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as archive:
        keys = archive["asset_keys"].tolist()
        features = np.asarray(archive["features"], dtype=np.float32)
        centers = np.asarray(archive["centers_seconds"], dtype=np.float64)
        actual_padding = np.asarray(archive["padding_samples"], dtype=np.int64)
        declared_padding = np.asarray(archive["declared_padding_samples"], dtype=np.int64)
    if (
        features.ndim != 2
        or features.shape != (len(keys), 384)
        or len(set(keys)) != len(keys)
        or not np.isfinite(features).all()
        or centers.shape != (len(keys),)
        or actual_padding.shape != (len(keys), 2)
        or declared_padding.shape != (len(keys), 2)
        or not np.isfinite(centers).all()
        or (actual_padding < 0).any()
        or (actual_padding.sum(axis=1) > AUDIO_WINDOW_SAMPLES).any()
        or (declared_padding < 0).any()
        or (declared_padding.sum(axis=1) > AUDIO_WINDOW_SAMPLES).any()
    ):
        raise ValueError(f"invalid frozen audio feature cache: {path}")
    if set(keys) != set(expected):
        raise ValueError("audio feature cache IDs differ from frozen input audio IDs")
    centers_by_key = dict(zip(keys, centers, strict=True))
    declared_by_key = dict(zip(keys, declared_padding.tolist(), strict=True))
    for key, (expected_center, expected_padding) in expected.items():
        if abs(centers_by_key[key] - expected_center) > 1e-9:
            raise ValueError(f"audio feature center differs from frozen input for {key}")
        if tuple(declared_by_key[key]) != expected_padding:
            raise ValueError(f"declared audio padding differs from frozen input for {key}")
    padding_delta = actual_padding - declared_padding
    diagnostics = {
        "asset_count": len(keys),
        "feature_shape": list(features.shape),
        "center_times_match_frozen_input": True,
        "declared_padding_matches_frozen_input": True,
        "actual_padding_mismatch_count": int(np.any(padding_delta != 0, axis=1).sum()),
        "audio_shorter_than_annotation_count": int((padding_delta[:, 1] > 0).sum()),
        "audio_longer_than_annotation_count": int((padding_delta[:, 1] < 0).sum()),
        "max_abs_padding_delta_samples": int(np.abs(padding_delta).max(initial=0)),
    }
    return keys, features, diagnostics


def load_video_feature_cache(
    path: Path, expected: dict[str, float]
) -> tuple[list[str], np.ndarray, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as archive:
        keys = archive["asset_keys"].tolist()
        features = np.asarray(archive["features"], dtype=np.float32)
        timestamps = np.asarray(archive["timestamps"], dtype=np.float64)
        frame_errors = np.asarray(archive["max_frame_time_error_seconds"], dtype=np.float64)
    if (
        features.ndim != 2
        or features.shape != (len(keys), 768)
        or len(set(keys)) != len(keys)
        or not np.isfinite(features).all()
        or timestamps.shape != (len(keys),)
        or frame_errors.shape != (len(keys),)
        or not np.isfinite(timestamps).all()
        or not np.isfinite(frame_errors).all()
        or (frame_errors < 0).any()
    ):
        raise ValueError(f"invalid frozen video feature cache: {path}")
    if set(keys) != set(expected):
        raise ValueError("video feature cache IDs differ from frozen input video IDs")
    timestamp_by_key = dict(zip(keys, timestamps, strict=True))
    for key, expected_timestamp in expected.items():
        if abs(timestamp_by_key[key] - expected_timestamp) > 1e-9:
            raise ValueError(f"video feature timestamp differs from frozen input for {key}")
    diagnostics = {
        "asset_count": len(keys),
        "feature_shape": list(features.shape),
        "timestamps_match_frozen_input": True,
        "frame_time_error_max_seconds": float(frame_errors.max(initial=0.0)),
        "frame_time_error_mean_seconds": float(frame_errors.mean()),
        "frame_time_error_p95_seconds": float(np.quantile(frame_errors, 0.95)),
    }
    return keys, features, diagnostics


def assemble_split(
    model_rows: list[dict[str, Any]],
    sidecar_rows: list[dict[str, Any]],
    audio_keys: list[str],
    audio_features: np.ndarray,
    video_keys: list[str],
    video_features: np.ndarray,
    *,
    expected_split: str,
) -> dict[str, Any]:
    if not model_rows or len(model_rows) != len(sidecar_rows):
        raise ValueError("model inputs and analysis sidecar must be nonempty and row-aligned")
    audio_index = {key: index for index, key in enumerate(audio_keys)}
    video_index = {key: index for index, key in enumerate(video_keys)}
    sidecar_by_id = {row["example_id"]: row for row in sidecar_rows}
    if len(sidecar_by_id) != len(sidecar_rows):
        raise ValueError("duplicate example ID in analysis sidecar")

    audio_rows = []
    video_rows = []
    metadata_rows = []
    targets = []
    example_ids = []
    groups = []
    target_categories = []
    option_categories = []
    audio_asset_ids = []
    video_option_asset_ids = []
    video_option_source_groups = []
    source_groups = set()
    all_assets = set()
    audio_asset_keys = set()
    video_asset_keys = set()
    for row in model_rows:
        if row.get("split") != expected_split:
            raise ValueError(f"unexpected split in example {row.get('example_id')}")
        example_id = row.get("example_id")
        sidecar = sidecar_by_id.pop(example_id, None)
        if sidecar is None:
            raise ValueError(f"analysis sidecar is missing {example_id}")
        if len(row.get("options", [])) != 4:
            raise ValueError(f"expected exactly four options for {example_id}")
        option_ids = [option.get("option_id") for option in row["options"]]
        if len(set(option_ids)) != 4 or row.get("target_option_id") not in option_ids:
            raise ValueError(f"invalid option IDs or target for {example_id}")
        target = option_ids.index(row["target_option_id"])
        audio_key = row.get("audio_asset_key")
        option_keys = [option.get("video_asset_key") for option in row["options"]]
        if len(set(option_keys)) != 4:
            raise ValueError(f"duplicate video option asset for {example_id}")
        if audio_key not in audio_index or any(key not in video_index for key in option_keys):
            raise ValueError(f"frozen features are missing assets for {example_id}")
        if sidecar.get("target_option_index") != target:
            raise ValueError(f"target index disagrees with analysis sidecar for {example_id}")
        if sidecar.get("option_asset_keys") != option_keys:
            raise ValueError(f"option order disagrees with analysis sidecar for {example_id}")
        option_source_groups = sidecar.get("option_source_groups", [])
        sidecar_option_categories = sidecar.get("option_categories", [])
        if len(option_source_groups) != 4 or len(sidecar_option_categories) != 4:
            raise ValueError(f"analysis sidecar group metadata malformed for {example_id}")
        if len(set(option_source_groups)) != 4:
            raise ValueError(f"options reuse an underlying source group for {example_id}")
        if len(set(sidecar_option_categories)) != 4:
            raise ValueError(f"options do not have four distinct categories for {example_id}")
        if option_source_groups[target] != sidecar.get("target_source_group"):
            raise ValueError(f"target source group disagrees with option groups for {example_id}")
        if sidecar_option_categories[target] != sidecar.get("target_category"):
            raise ValueError(f"target category disagrees with option categories for {example_id}")
        audio_source_group = str(
            sidecar.get("audio_source_group", sidecar["target_source_group"])
        )
        source_groups.add(audio_source_group)
        source_groups.update(str(group) for group in sidecar["option_source_groups"])
        audio_rows.append(audio_features[audio_index[audio_key]])
        video_rows.append(np.stack([video_features[video_index[key]] for key in option_keys]))
        metadata = np.asarray(
            [
                [
                    float(row["audio_window_center_seconds"]),
                    float(row["audio_window_duration_seconds"]),
                    float(row["audio_window_left_padding_seconds"]),
                    float(row["audio_window_right_padding_seconds"]),
                    float(option["video_timestamp_seconds"]),
                ]
                for option in row["options"]
            ],
            dtype=np.float32,
        )
        if (
            not np.isfinite(metadata).all()
            or metadata[:, 1].min() <= 0
            or (metadata[:, 2:4] < 0).any()
            or metadata[:, 4].min() < 0
        ):
            raise ValueError(f"invalid timing metadata for {example_id}")
        metadata_rows.append(metadata)
        targets.append(target)
        example_ids.append(example_id)
        groups.append(audio_source_group)
        target_categories.append(str(sidecar["target_category"]))
        option_categories.append([str(category) for category in sidecar_option_categories])
        audio_asset_ids.append(audio_key)
        video_option_asset_ids.append(option_keys)
        video_option_source_groups.append([str(group) for group in option_source_groups])
        all_assets.add(audio_key)
        all_assets.update(option_keys)
        audio_asset_keys.add(audio_key)
        video_asset_keys.update(option_keys)
    if sidecar_by_id:
        raise ValueError(f"unused analysis-sidecar examples: {len(sidecar_by_id)}")
    return {
        "audio": np.stack(audio_rows).astype(np.float32),
        "video": np.stack(video_rows).astype(np.float32),
        "metadata": np.stack(metadata_rows).astype(np.float32),
        "targets": np.asarray(targets, dtype=np.int64),
        "example_ids": example_ids,
        "groups": groups,
        "target_categories": target_categories,
        "option_categories": option_categories,
        "audio_asset_ids": audio_asset_ids,
        "video_option_asset_ids": video_option_asset_ids,
        "video_option_source_groups": video_option_source_groups,
        "source_groups": source_groups,
        "asset_keys": all_assets,
        "audio_asset_keys": audio_asset_keys,
        "video_asset_keys": video_asset_keys,
    }


def compute_metrics(
    probabilities: np.ndarray, targets: np.ndarray, bins: int = 15
) -> dict[str, float]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    if (
        probabilities.ndim != 2
        or probabilities.shape[0] != len(targets)
        or probabilities.shape[1] != 4
        or len(targets) == 0
        or not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    ):
        raise ValueError("invalid 4-choice probabilities")
    prediction = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = prediction == targets
    one_hot = np.eye(4, dtype=np.float64)[targets]
    clipped = np.clip(probabilities, 1e-12, 1.0)
    ece = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        mask = (confidence >= edges[index]) & (
            confidence < edges[index + 1] if index + 1 < bins else confidence <= edges[index + 1]
        )
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return {
        "count": int(len(targets)),
        "accuracy": float(correct.mean()),
        "nll": float(-np.log(clipped[np.arange(len(targets)), targets]).mean()),
        "brier": float(np.square(probabilities - one_hot).sum(axis=1).mean()),
        "ece_15": float(ece),
        "mean_confidence": float(confidence.mean()),
    }


def compute_grouped_metrics(
    probabilities: np.ndarray, targets: np.ndarray, groups: list[str]
) -> dict[str, Any]:
    if len(groups) != len(targets):
        raise ValueError("group labels must align with probability rows")
    group_values = np.asarray(groups)
    by_group = {
        group: compute_metrics(probabilities[group_values == group], targets[group_values == group])
        for group in sorted(set(groups))
    }
    metric_keys = ("accuracy", "nll", "brier", "ece_15", "mean_confidence")
    macro = {
        key: float(np.mean([metrics[key] for metrics in by_group.values()])) for key in metric_keys
    }
    return {"group_count": len(by_group), "macro_over_groups": macro, "by_group": by_group}


def group_disjoint_donor_indices(groups: list[str], seed: int = SEED) -> np.ndarray:
    if len(groups) < 2 or len(set(groups)) < 2:
        raise ValueError("group shuffle requires at least two source groups")
    rng = np.random.default_rng(seed)
    group_values = np.asarray(groups)
    unique_groups = np.asarray(sorted(set(groups)))
    rng.shuffle(unique_groups)
    donor_group = {
        group: unique_groups[(index + 1) % len(unique_groups)]
        for index, group in enumerate(unique_groups)
    }
    rows_by_group = {group: np.flatnonzero(group_values == group) for group in unique_groups}
    donors = np.empty(len(groups), dtype=np.int64)
    for index, group in enumerate(group_values):
        candidates = rows_by_group[donor_group[group]]
        donors[index] = candidates[int(rng.integers(0, len(candidates)))]
    if not np.all(group_values[donors] != group_values):
        raise ValueError("group-disjoint donor assignment failed")
    return donors


def group_bootstrap_difference(
    left_correct: np.ndarray,
    right_correct: np.ndarray,
    groups: list[str],
    *,
    seed: int = SEED,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, float]:
    if len(left_correct) != len(right_correct) or len(groups) != len(left_correct):
        raise ValueError("paired bootstrap inputs must be row-aligned")
    unique_groups = sorted(set(groups))
    if len(unique_groups) < 2:
        raise ValueError("paired bootstrap needs at least two source groups")
    group_indices = [np.flatnonzero(np.asarray(groups) == group) for group in unique_groups]
    rng = np.random.default_rng(seed)
    differences = np.empty(replicates, dtype=np.float64)
    for iteration in range(replicates):
        sampled = rng.integers(0, len(unique_groups), size=len(unique_groups))
        indices = np.concatenate([group_indices[int(index)] for index in sampled])
        differences[iteration] = (
            np.asarray(left_correct)[indices].mean() - np.asarray(right_correct)[indices].mean()
        )
    return {
        "mean_difference": float(
            np.asarray(left_correct).mean() - np.asarray(right_correct).mean()
        ),
        "ci_95_low": float(np.quantile(differences, 0.025)),
        "ci_95_high": float(np.quantile(differences, 0.975)),
        "bootstrap_replicates": replicates,
        "resampling_unit": "audio query source-video group, sampled with replacement",
        "seed": seed,
    }


def _metrics_for_logits(logits: Any, targets: Any) -> tuple[dict[str, float], np.ndarray]:
    import torch

    probabilities = torch.softmax(logits.detach().float(), dim=1).cpu().numpy()
    return compute_metrics(probabilities, targets.detach().cpu().numpy()), probabilities


def _model_class(torch: Any, input_dim: int):
    class PairwiseScorer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.Sequential(
                torch.nn.Linear(input_dim, HIDDEN_DIM),
                torch.nn.GELU(),
                torch.nn.Linear(HIDDEN_DIM, 1),
            )

        def forward(self, values):
            return self.layers(values).squeeze(-1)

    return PairwiseScorer


def _gated_model_class(torch: Any):
    class ConditionalGateScorer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.audio_projection = torch.nn.Linear(384, HIDDEN_DIM)
            self.video_projection = torch.nn.Linear(768, HIDDEN_DIM)
            self.gate_projection = torch.nn.Linear(2 * HIDDEN_DIM, HIDDEN_DIM)
            self.output = torch.nn.Linear(HIDDEN_DIM, 1)

        def gates(self, values):
            audio = torch.nn.functional.gelu(self.audio_projection(values[..., :384]))
            video = torch.nn.functional.gelu(self.video_projection(values[..., 384:]))
            return torch.sigmoid(self.gate_projection(torch.cat([audio, video], dim=-1)))

        def forward(self, values):
            audio = torch.nn.functional.gelu(self.audio_projection(values[..., :384]))
            video = torch.nn.functional.gelu(self.video_projection(values[..., 384:]))
            gate = torch.sigmoid(self.gate_projection(torch.cat([audio, video], dim=-1)))
            fused = gate * audio + (1.0 - gate) * video
            return self.output(fused).squeeze(-1)

    return ConditionalGateScorer


def _normalize(train: dict[str, Any], validation: dict[str, Any]) -> dict[str, Any]:
    train_audio = train["audio"]
    train_video = train["video"]
    train_metadata = train["metadata"]
    audio_mean = train_audio.mean(axis=0)
    audio_std = train_audio.std(axis=0).clip(1e-5)
    video_mean = train_video.reshape(-1, train_video.shape[-1]).mean(axis=0)
    video_std = train_video.reshape(-1, train_video.shape[-1]).std(axis=0).clip(1e-5)
    metadata_mean = train_metadata.reshape(-1, train_metadata.shape[-1]).mean(axis=0)
    metadata_std = train_metadata.reshape(-1, train_metadata.shape[-1]).std(axis=0).clip(1e-5)

    def apply(data):
        audio = (data["audio"] - audio_mean) / audio_std
        video = (data["video"] - video_mean) / video_std
        metadata = (data["metadata"] - metadata_mean) / metadata_std
        return {
            "audio": audio.astype(np.float32),
            "video": video.astype(np.float32),
            "metadata": metadata.astype(np.float32),
            "targets": data["targets"],
            "example_ids": data["example_ids"],
            "groups": data["groups"],
            "target_categories": data["target_categories"],
            "option_categories": data["option_categories"],
            "audio_asset_ids": data["audio_asset_ids"],
            "video_option_asset_ids": data["video_option_asset_ids"],
            "video_option_source_groups": data["video_option_source_groups"],
            "asset_keys": data["asset_keys"],
            "audio_asset_keys": data["audio_asset_keys"],
            "video_asset_keys": data["video_asset_keys"],
            "source_groups": data["source_groups"],
        }

    return {
        "train": apply(train),
        "validation": apply(validation),
        "normalization": {
            "audio_mean": audio_mean,
            "audio_std": audio_std,
            "video_mean": video_mean,
            "video_std": video_std,
            "metadata_mean": metadata_mean,
            "metadata_std": metadata_std,
        },
    }


def _joint_pairs(data: dict[str, Any]) -> np.ndarray:
    audio = np.repeat(data["audio"][:, None, :], 4, axis=1)
    return np.concatenate([audio, data["video"]], axis=2).astype(np.float32)


def _video_pairs(data: dict[str, Any]) -> np.ndarray:
    return data["video"].astype(np.float32)


def _metadata_pairs(data: dict[str, Any]) -> np.ndarray:
    return data["metadata"].astype(np.float32)


def _score_audio_category_candidates(
    category_logits: np.ndarray,
    option_categories: list[list[str]],
    category_to_index: dict[str, int],
) -> np.ndarray:
    known = np.asarray(
        [[category in category_to_index for category in row] for row in option_categories],
        dtype=bool,
    )
    indices = np.asarray(
        [[category_to_index.get(category, 0) for category in row] for row in option_categories],
        dtype=np.int64,
    )
    scores = np.take_along_axis(category_logits[:, None, :], indices[:, :, None], axis=2)[:, :, 0]
    scores[~known] = -30.0
    return scores


def _fit_candidate(
    torch: Any,
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    *,
    output_path: Path,
    device: Any,
    model_kind: str = "mlp",
) -> dict[str, Any]:
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    if model_kind == "conditional_gate":
        if train_x.shape[2] != 1152:
            raise ValueError("conditional AV gate expects 384-D audio + 768-D video features")
        model_type = _gated_model_class(torch)
    elif model_kind == "mlp":
        model_type = _model_class(torch, train_x.shape[2])
    else:
        raise ValueError(f"unsupported readout model kind: {model_kind}")
    model = model_type().to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_values = torch.from_numpy(train_x)
    train_targets = torch.from_numpy(train_y)
    validation_values = torch.from_numpy(validation_x).to(device)
    validation_targets = torch.from_numpy(validation_y).to(device)
    rng = np.random.default_rng(SEED)
    history = []
    best_nll = math.inf
    best_epoch = 0
    best_state = None
    last_state = None
    stale_epochs = 0
    started = time.perf_counter()
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        order = rng.permutation(len(train_y))
        train_loss_sum = 0.0
        train_correct = 0
        train_seen = 0
        gradient_norm_sum = 0.0
        gradient_norm_max = 0.0
        optimizer_steps = 0
        for first in range(0, len(order), BATCH_SIZE):
            indices = order[first : first + BATCH_SIZE]
            batch_x = train_values[indices].to(device)
            batch_y = train_targets[indices].to(device)
            logits = model(batch_x)
            loss = torch.nn.functional.cross_entropy(logits, batch_y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError("non-finite Decision readout gradient norm")
            gradient_norm_value = float(grad_norm)
            gradient_norm_sum += gradient_norm_value
            gradient_norm_max = max(gradient_norm_max, gradient_norm_value)
            optimizer_steps += 1
            optimizer.step()
            train_loss_sum += float(loss.detach()) * len(indices)
            train_correct += int((logits.argmax(dim=1) == batch_y).sum())
            train_seen += len(indices)
        model.eval()
        with torch.no_grad():
            train_eval_logits = []
            for first in range(0, len(train_y), BATCH_SIZE):
                train_eval_logits.append(
                    model(train_values[first : first + BATCH_SIZE].to(device)).cpu()
                )
            train_logits = torch.cat(train_eval_logits).to(device)
            train_metrics, _ = _metrics_for_logits(train_logits, torch.from_numpy(train_y))
            validation_logits = model(validation_values)
            validation_metrics, validation_probabilities = _metrics_for_logits(
                validation_logits, validation_targets
            )
        history.append(
            {
                "epoch": epoch,
                "train_cross_entropy": train_loss_sum / train_seen,
                "train_accuracy": train_correct / train_seen,
                "fixed_train_evaluation": train_metrics,
                "validation": validation_metrics,
                "optimizer_steps": optimizer_steps,
                "mean_gradient_norm": gradient_norm_sum / optimizer_steps,
                "max_gradient_norm": gradient_norm_max,
            }
        )
        last_state = {
            name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()
        }
        if validation_metrics["nll"] < best_nll - MIN_DELTA:
            best_nll = validation_metrics["nll"]
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()
            }
            stale_epochs = 0
            temporary = output_path.with_suffix(output_path.suffix + ".tmp")
            torch.save(best_state, temporary)
            temporary.replace(output_path)
        else:
            stale_epochs += 1
        if stale_epochs >= PATIENCE:
            break
    if best_state is None:
        raise RuntimeError("validation selector did not save a best checkpoint")
    if last_state is None:
        raise RuntimeError("training did not produce a final checkpoint")
    last_path = output_path.with_name("last-state.pt")
    temporary = last_path.with_suffix(last_path.suffix + ".tmp")
    torch.save(last_state, temporary)
    temporary.replace(last_path)
    saved_best_state = torch.load(output_path, map_location="cpu", weights_only=True)
    model.load_state_dict(saved_best_state)
    model.eval()
    with torch.no_grad():
        selected_logits = model(validation_values)
        gate_diagnostics = None
        if model_kind == "conditional_gate":
            gate_values = model.gates(validation_values).float()
            gate_diagnostics = {
                "audio_weight_mean": float(gate_values.mean().cpu()),
                "audio_weight_p10": float(torch.quantile(gate_values, 0.10).cpu()),
                "audio_weight_p90": float(torch.quantile(gate_values, 0.90).cpu()),
                "fraction_saturated_below_0_05": float((gate_values < 0.05).float().mean().cpu()),
                "fraction_saturated_above_0_95": float((gate_values > 0.95).float().mean().cpu()),
            }
    selected_metrics, selected_probabilities = _metrics_for_logits(
        selected_logits, validation_targets
    )
    return {
        "model": model,
        "parameter_count": parameter_count,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "best_validation_nll": best_nll,
        "selected_validation_metrics": selected_metrics,
        "selected_validation_probabilities": selected_probabilities,
        "history": history,
        "wall_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if torch.cuda.is_available() and device.type == "cuda"
            else 0
        ),
        "model_kind": model_kind,
        "gate_diagnostics": gate_diagnostics,
    }


def _fit_audio_category_classifier(
    torch: Any,
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    *,
    output_dir: Path,
    class_count: int,
    device: Any,
) -> dict[str, Any]:
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    class AudioCategoryClassifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.Sequential(
                torch.nn.Linear(train_x.shape[1], HIDDEN_DIM),
                torch.nn.GELU(),
                torch.nn.Linear(HIDDEN_DIM, class_count),
            )

        def forward(self, values):
            return self.layers(values)

    model = AudioCategoryClassifier().to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_values = torch.from_numpy(train_x)
    train_targets = torch.from_numpy(train_y)
    validation_values = torch.from_numpy(validation_x).to(device)
    validation_targets = torch.from_numpy(validation_y).to(device)
    rng = np.random.default_rng(SEED)
    history = []
    best_nll = math.inf
    best_epoch = 0
    best_state = None
    last_state = None
    stale_epochs = 0
    started = time.perf_counter()
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        order = rng.permutation(len(train_y))
        train_loss_sum = 0.0
        train_seen = 0
        optimizer_steps = 0
        gradient_norm_sum = 0.0
        gradient_norm_max = 0.0
        for first in range(0, len(order), BATCH_SIZE):
            indices = order[first : first + BATCH_SIZE]
            batch_x = train_values[indices].to(device)
            batch_y = train_targets[indices].to(device)
            logits = model(batch_x)
            loss = torch.nn.functional.cross_entropy(logits, batch_y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError("non-finite audio-category gradient norm")
            norm_value = float(grad_norm)
            gradient_norm_sum += norm_value
            gradient_norm_max = max(gradient_norm_max, norm_value)
            optimizer_steps += 1
            optimizer.step()
            train_loss_sum += float(loss.detach()) * len(indices)
            train_seen += len(indices)
        model.eval()
        with torch.no_grad():
            train_logits = []
            for first in range(0, len(train_y), BATCH_SIZE):
                train_logits.append(
                    model(train_values[first : first + BATCH_SIZE].to(device)).cpu()
                )
            fixed_train_logits = torch.cat(train_logits).to(device)
            fixed_train_nll = float(
                torch.nn.functional.cross_entropy(fixed_train_logits, train_targets.to(device))
            )
            fixed_train_accuracy = float(
                (fixed_train_logits.argmax(dim=1) == train_targets.to(device)).float().mean()
            )
            validation_logits = model(validation_values)
            validation_nll = float(
                torch.nn.functional.cross_entropy(validation_logits, validation_targets)
            )
            validation_accuracy = float(
                (validation_logits.argmax(dim=1) == validation_targets).float().mean()
            )
        history.append(
            {
                "epoch": epoch,
                "train_cross_entropy": train_loss_sum / train_seen,
                "fixed_train_nll": fixed_train_nll,
                "fixed_train_accuracy": fixed_train_accuracy,
                "validation_nll": validation_nll,
                "validation_accuracy": validation_accuracy,
                "optimizer_steps": optimizer_steps,
                "mean_gradient_norm": gradient_norm_sum / optimizer_steps,
                "max_gradient_norm": gradient_norm_max,
            }
        )
        last_state = {
            name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()
        }
        if validation_nll < best_nll - MIN_DELTA:
            best_nll = validation_nll
            best_epoch = epoch
            best_state = last_state
            stale_epochs = 0
            temporary = output_dir / "best-state.pt.tmp"
            torch.save(best_state, temporary)
            temporary.replace(output_dir / "best-state.pt")
        else:
            stale_epochs += 1
        if stale_epochs >= PATIENCE:
            break
    if best_state is None or last_state is None:
        raise RuntimeError("audio-category validation selector did not save checkpoints")
    temporary = output_dir / "last-state.pt.tmp"
    torch.save(last_state, temporary)
    temporary.replace(output_dir / "last-state.pt")
    saved_best_state = torch.load(
        output_dir / "best-state.pt", map_location="cpu", weights_only=True
    )
    model.load_state_dict(saved_best_state)
    model.eval()
    with torch.no_grad():
        best_validation_logits = model(validation_values).cpu().numpy()
    return {
        "model": model,
        "parameter_count": parameter_count,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "best_validation_nll": best_nll,
        "best_validation_accuracy": history[best_epoch - 1]["validation_accuracy"],
        "best_validation_logits": best_validation_logits,
        "history": history,
        "wall_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if torch.cuda.is_available() and device.type == "cuda"
            else 0
        ),
    }


def _set_metrics(model: Any, pair_values: np.ndarray, targets: np.ndarray, device: Any):
    import torch

    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(pair_values).to(device))
    return _metrics_for_logits(logits, torch.from_numpy(targets).to(device))


def _run(args: argparse.Namespace) -> dict[str, Any]:
    train_input = args.train_input.resolve(strict=True)
    validation_input = args.validation_input.resolve(strict=True)
    train_sidecar = args.train_sidecar.resolve(strict=True)
    validation_sidecar = args.validation_sidecar.resolve(strict=True)
    audio_features_path = args.audio_features.resolve(strict=True)
    video_features_path = args.video_features.resolve(strict=True)
    train_rows = read_jsonl(train_input)
    validation_rows = read_jsonl(validation_input)
    all_model_rows = train_rows + validation_rows
    audio_keys, audio_features, audio_cache_diagnostics = load_audio_feature_cache(
        audio_features_path, _expected_audio_metadata(all_model_rows)
    )
    video_keys, video_features, video_cache_diagnostics = load_video_feature_cache(
        video_features_path, _expected_video_timestamps(all_model_rows)
    )
    train = assemble_split(
        train_rows,
        read_jsonl(train_sidecar),
        audio_keys,
        audio_features,
        video_keys,
        video_features,
        expected_split="train",
    )
    validation = assemble_split(
        validation_rows,
        read_jsonl(validation_sidecar),
        audio_keys,
        audio_features,
        video_keys,
        video_features,
        expected_split="validation",
    )
    if train["source_groups"] & validation["source_groups"]:
        raise ValueError("train and validation source-video groups overlap")
    if train["asset_keys"] & validation["asset_keys"]:
        raise ValueError("train and validation media assets overlap")
    if (train["audio_asset_keys"] | validation["audio_asset_keys"]) != set(audio_keys):
        raise ValueError("audio feature cache contains unexpected or missing asset IDs")
    if (train["video_asset_keys"] | validation["video_asset_keys"]) != set(video_keys):
        raise ValueError("video feature cache contains unexpected or missing asset IDs")
    if set(train["example_ids"]) & set(validation["example_ids"]):
        raise ValueError("train and validation example IDs overlap")
    normalized = _normalize(train, validation)
    train_n, val_n = normalized["train"], normalized["validation"]
    audio_categories = sorted(set(train_n["target_categories"]))
    category_to_index = {category: index for index, category in enumerate(audio_categories)}
    unknown_validation_categories = sorted(set(val_n["target_categories"]) - set(category_to_index))
    if unknown_validation_categories:
        raise ValueError(
            "validation targets contain categories absent from train: "
            f"{unknown_validation_categories}"
        )
    torch = __import__("torch")
    requested_device = getattr(args, "device", "auto")
    if requested_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    else:
        device = torch.device(requested_device)
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite AVATAR readout output: {output}")
    output.mkdir(parents=True)
    run_started_utc = datetime.now(UTC).isoformat()
    config = {
        "repository_head": subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parent.parent), "rev-parse", "HEAD"],
            text=True,
        ).strip(),
        "readout_script_sha256": sha256_file(Path(__file__).resolve()),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "train_input_sha256": sha256_file(train_input),
        "validation_input_sha256": sha256_file(validation_input),
        "train_sidecar_sha256": sha256_file(train_sidecar),
        "validation_sidecar_sha256": sha256_file(validation_sidecar),
        "audio_feature_sha256": sha256_file(audio_features_path),
        "video_feature_sha256": sha256_file(video_features_path),
        "audio_cache_diagnostics": audio_cache_diagnostics,
        "video_cache_diagnostics": video_cache_diagnostics,
        "audio_encoder": {
            "name": "EfficientAT DyMN-0.4",
            "source_revision": "a425fdce92572e602a1d5634799bd9f1f2efa806",
            "checkpoint_sha256": "49b0e63577021994257b3de2ac5f1686d71554b93c8149f9d2568ad5760b904e",
            "parameters": 1_961_239,
            "feature_dim": 384,
        },
        "video_encoder": {
            "name": "VideoPrism-B",
            "source_revision": "be719a406d563b66f0ac969e7c94bab8e997c81a",
            "checkpoint_sha256": "61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29",
            "parameters": 114_365_184,
            "feature_dim": 768,
        },
        "seed": SEED,
        "requested_device": requested_device,
        "resolved_device": str(device),
        "device_name": (torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"),
        "task_definition": "select the candidate video whose category matches the query audio",
        "audio_category_labels_sorted": audio_categories,
        "audio_candidate_categories_unseen_in_train": sorted(
            set(category for row in val_n["option_categories"] for category in row)
            - set(category_to_index)
        ),
        "readout": {
            "joint": "per-option MLP 1152->64->1",
            "joint_parameters": 73_857,
            "joint_conditional_gate": (
                "per-option 384->64 audio and 768->64 video projections, "
                "feature-conditioned 128->64 sigmoid gate, weighted sum, 64->1"
            ),
            "joint_conditional_gate_parameters": 82_177,
            "video_only": "per-option MLP 768->64->1",
            "video_only_parameters": 49_281,
            "metadata_only": "per-option MLP 5->64->1",
            "metadata_only_parameters": 449,
            "audio_category_only": "audio MLP 384->64->80",
            "audio_category_only_parameters": 29_840,
            "option_only": "Laplace-smoothed training target-position prior",
            "optimizer": "AdamW(lr=1e-3, weight_decay=1e-4)",
        },
        "hidden_dim": HIDDEN_DIM,
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": PATIENCE,
        "min_delta": MIN_DELTA,
        "batch_size": BATCH_SIZE,
        "selection_metric": "validation NLL",
        "sealed_audit_loaded": False,
        "model_input_policy": {
            "joint": "frozen query-audio and candidate-video features",
            "joint_conditional_gate": "the same frozen query-audio and candidate-video features",
            "video_only": "candidate-video features only",
            "metadata_only": "audio-window center/duration/padding and candidate timestamps only",
            "audio_category_only": (
                "query-audio features only; target category is a training label and candidate "
                "categories are used only by the offline option scorer, making this an oracle "
                "diagnostic rather than a deployable option predictor"
            ),
            "analysis_sidecar_as_feature_input": False,
        },
        "split_integrity": {
            "all_target_and_option_source_groups_must_be_disjoint": True,
            "all_media_feature_ids_must_be_disjoint": True,
        },
        "audio_shuffle_control": (
            "validation audio donors are assigned from different audio asset IDs; donor rows "
            "are sampled with replacement from a deterministic cycle of distinct assets"
        ),
    }
    write_json_atomic(output / "effective-config.json", config)
    effective_config_sha256 = sha256_file(output / "effective-config.json")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    joint_dir = output / "joint"
    video_only_dir = output / "video-only"
    joint_dir.mkdir()
    video_only_dir.mkdir()
    train_joint = _joint_pairs(train_n)
    validation_joint = _joint_pairs(val_n)
    joint = _fit_candidate(
        torch,
        train_joint,
        train_n["targets"],
        validation_joint,
        val_n["targets"],
        output_path=joint_dir / "best-state.pt",
        device=device,
    )
    gated_joint_dir = output / "joint-conditional-gate"
    gated_joint_dir.mkdir()
    gated_joint = _fit_candidate(
        torch,
        train_joint,
        train_n["targets"],
        validation_joint,
        val_n["targets"],
        output_path=gated_joint_dir / "best-state.pt",
        device=device,
        model_kind="conditional_gate",
    )
    video_only = _fit_candidate(
        torch,
        _video_pairs(train_n),
        train_n["targets"],
        _video_pairs(val_n),
        val_n["targets"],
        output_path=video_only_dir / "best-state.pt",
        device=device,
    )
    metadata_only_dir = output / "metadata-only"
    metadata_only_dir.mkdir()
    metadata_only = _fit_candidate(
        torch,
        _metadata_pairs(train_n),
        train_n["targets"],
        _metadata_pairs(val_n),
        val_n["targets"],
        output_path=metadata_only_dir / "best-state.pt",
        device=device,
    )
    train_category_targets = np.asarray(
        [category_to_index[category] for category in train_n["target_categories"]],
        dtype=np.int64,
    )
    validation_category_targets = np.asarray(
        [category_to_index[category] for category in val_n["target_categories"]],
        dtype=np.int64,
    )
    audio_only_dir = output / "audio-category-only"
    audio_only_dir.mkdir()
    audio_category = _fit_audio_category_classifier(
        torch,
        train_n["audio"],
        train_category_targets,
        val_n["audio"],
        validation_category_targets,
        output_dir=audio_only_dir,
        class_count=len(audio_categories),
        device=device,
    )

    uniform = np.full((len(val_n["targets"]), 4), 0.25, dtype=np.float64)
    uniform_metrics = compute_metrics(uniform, val_n["targets"])
    uniform_metrics["accuracy"] = 0.25
    position_counts = np.bincount(train_n["targets"], minlength=4).astype(np.float64)
    position_prior = (position_counts + 1.0) / (position_counts.sum() + 4.0)
    position_prior_probabilities = np.repeat(position_prior[None, :], len(val_n["targets"]), axis=0)
    position_prior_metrics = compute_metrics(position_prior_probabilities, val_n["targets"])
    joint_metrics, joint_probabilities = (
        joint["selected_validation_metrics"],
        joint["selected_validation_probabilities"],
    )
    video_metrics, video_probabilities = (
        video_only["selected_validation_metrics"],
        video_only["selected_validation_probabilities"],
    )
    metadata_metrics, metadata_probabilities = (
        metadata_only["selected_validation_metrics"],
        metadata_only["selected_validation_probabilities"],
    )
    audio_option_logits = _score_audio_category_candidates(
        audio_category["best_validation_logits"],
        val_n["option_categories"],
        category_to_index,
    )
    audio_only_probabilities = torch.softmax(torch.from_numpy(audio_option_logits), dim=1).numpy()
    audio_only_metrics = compute_metrics(audio_only_probabilities, val_n["targets"])
    joint_missing_audio_metrics, joint_missing_audio_probs = _set_metrics(
        joint["model"],
        np.concatenate(
            [np.zeros_like(val_n["audio"][:, None, :]).repeat(4, axis=1), val_n["video"]], axis=2
        ),
        val_n["targets"],
        device,
    )
    joint_missing_video_values = np.concatenate(
        [np.repeat(val_n["audio"][:, None, :], 4, axis=1), np.zeros_like(val_n["video"])],
        axis=2,
    )
    joint_missing_video_metrics, joint_missing_video_probs = _set_metrics(
        joint["model"], joint_missing_video_values, val_n["targets"], device
    )
    permutation = group_disjoint_donor_indices(val_n["audio_asset_ids"], seed=SEED)
    shuffled_audio_values = np.concatenate(
        [
            np.repeat(val_n["audio"][permutation, None, :], 4, axis=1),
            val_n["video"],
        ],
        axis=2,
    )
    shuffled_audio_metrics, shuffled_audio_probs = _set_metrics(
        joint["model"], shuffled_audio_values, val_n["targets"], device
    )
    shuffle_option_indices = np.random.default_rng(SEED).permutation(4)
    shuffled_video = val_n["video"][:, shuffle_option_indices, :]
    shuffled_video_values = np.concatenate(
        [np.repeat(val_n["audio"][:, None, :], 4, axis=1), shuffled_video], axis=2
    )
    shuffled_video_metrics, shuffled_video_probs = _set_metrics(
        joint["model"], shuffled_video_values, val_n["targets"], device
    )
    shuffled_video_targets = np.asarray(
        [int(np.flatnonzero(shuffle_option_indices == target)[0]) for target in val_n["targets"]]
    )
    shuffled_video_metrics = compute_metrics(shuffled_video_probs, shuffled_video_targets)

    gated_joint_metrics = gated_joint["selected_validation_metrics"]
    gated_joint_probabilities = gated_joint["selected_validation_probabilities"]
    gated_missing_audio_values = np.concatenate(
        [np.zeros_like(val_n["audio"][:, None, :]).repeat(4, axis=1), val_n["video"]],
        axis=2,
    )
    gated_missing_audio_metrics, gated_missing_audio_probs = _set_metrics(
        gated_joint["model"], gated_missing_audio_values, val_n["targets"], device
    )
    gated_missing_video_values = np.concatenate(
        [np.repeat(val_n["audio"][:, None, :], 4, axis=1), np.zeros_like(val_n["video"])],
        axis=2,
    )
    gated_missing_video_metrics, gated_missing_video_probs = _set_metrics(
        gated_joint["model"], gated_missing_video_values, val_n["targets"], device
    )
    gated_shuffled_audio_values = np.concatenate(
        [
            np.repeat(val_n["audio"][permutation, None, :], 4, axis=1),
            val_n["video"],
        ],
        axis=2,
    )
    gated_shuffled_audio_metrics, gated_shuffled_audio_probs = _set_metrics(
        gated_joint["model"], gated_shuffled_audio_values, val_n["targets"], device
    )
    gated_shuffled_video_values = np.concatenate(
        [np.repeat(val_n["audio"][:, None, :], 4, axis=1), shuffled_video], axis=2
    )
    gated_shuffled_video_metrics, gated_shuffled_video_probs = _set_metrics(
        gated_joint["model"], gated_shuffled_video_values, val_n["targets"], device
    )
    gated_shuffled_video_metrics = compute_metrics(
        gated_shuffled_video_probs, shuffled_video_targets
    )

    predictions = []
    conditions = {
        "uniform": uniform,
        "option_position_prior": position_prior_probabilities,
        "joint": joint_probabilities,
        "joint_conditional_gate": gated_joint_probabilities,
        "audio_only_oracle_category_match": audio_only_probabilities,
        "video_only": video_probabilities,
        "metadata_only": metadata_probabilities,
        "joint_missing_audio": joint_missing_audio_probs,
        "joint_missing_video": joint_missing_video_probs,
        "joint_group_shuffled_audio": shuffled_audio_probs,
        "joint_shuffled_video_option_order": shuffled_video_probs,
        "joint_gate_missing_audio": gated_missing_audio_probs,
        "joint_gate_missing_video": gated_missing_video_probs,
        "joint_gate_group_shuffled_audio": gated_shuffled_audio_probs,
        "joint_gate_shuffled_video_option_order": gated_shuffled_video_probs,
    }
    category_metrics = {
        name: compute_grouped_metrics(probabilities, val_n["targets"], val_n["target_categories"])
        for name, probabilities in conditions.items()
    }
    source_group_metrics = {
        name: compute_grouped_metrics(probabilities, val_n["targets"], val_n["groups"])
        for name, probabilities in conditions.items()
    }
    for index, example_id in enumerate(val_n["example_ids"]):
        row = {
            "example_id": example_id,
            "target_option_index": int(val_n["targets"][index]),
            "audio_source_group": val_n["groups"][index],
            "target_source_group": val_n["video_option_source_groups"][index][
                int(val_n["targets"][index])
            ],
            "target_category": val_n["target_categories"][index],
            "audio_asset_id": val_n["audio_asset_ids"][index],
            "video_option_asset_ids": val_n["video_option_asset_ids"][index],
        }
        for name, probabilities in conditions.items():
            row[name] = {
                "prediction_index": (
                    None if name == "uniform" else int(probabilities[index].argmax())
                ),
                "probabilities": probabilities[index].tolist(),
            }
        predictions.append(row)
    write_json_atomic(output / "validation-predictions.json", predictions)
    joint_correct = joint_probabilities.argmax(axis=1) == val_n["targets"]
    gated_joint_correct = gated_joint_probabilities.argmax(axis=1) == val_n["targets"]
    video_correct = video_probabilities.argmax(axis=1) == val_n["targets"]
    metadata_correct = metadata_probabilities.argmax(axis=1) == val_n["targets"]
    audio_only_correct = audio_only_probabilities.argmax(axis=1) == val_n["targets"]
    position_prior_correct = position_prior_probabilities.argmax(axis=1) == val_n["targets"]
    report = {
        "status": "PASS",
        "device": str(device),
        "train_examples": len(train_n["targets"]),
        "validation_examples": len(val_n["targets"]),
        "train_unique_source_groups": len(set(train_n["groups"])),
        "validation_unique_source_groups": len(set(val_n["groups"])),
        "train_unique_all_source_groups": len(train_n["source_groups"]),
        "validation_unique_all_source_groups": len(val_n["source_groups"]),
        "train_unique_assets": len(train_n["asset_keys"]),
        "validation_unique_assets": len(val_n["asset_keys"]),
        "train_validation_group_overlap": 0,
        "train_validation_asset_overlap": 0,
        "audio_shuffle_uses_distinct_asset_ids": bool(
            all(
                val_n["audio_asset_ids"][index] != val_n["audio_asset_ids"][donor]
                for index, donor in enumerate(permutation)
            )
        ),
        "trainable_parameters": {
            "joint": joint["parameter_count"],
            "video_only": video_only["parameter_count"],
            "metadata_only": metadata_only["parameter_count"],
            "audio_category_only": audio_category["parameter_count"],
        },
        "feature_cache_validation": {
            "audio": audio_cache_diagnostics,
            "video": video_cache_diagnostics,
        },
        "option_position_prior": {
            "train_counts": position_counts.astype(int).tolist(),
            "probabilities": position_prior.tolist(),
        },
        "checkpoint_selection": "best validation NLL with min_delta=0.001 and patience=6",
        "audio_category_classifier_diagnostic": {
            "class_count": len(audio_categories),
            "best_epoch": audio_category["best_epoch"],
            "epochs_run": audio_category["epochs_run"],
            "validation_nll": audio_category["best_validation_nll"],
            "validation_accuracy": audio_category["best_validation_accuracy"],
            "selection_metric": "validation category NLL",
        },
        "candidates": {
            "uniform_options_only": uniform_metrics,
            "option_position_prior": position_prior_metrics,
            "audio_only_oracle_category_match": {
                **audio_only_metrics,
                "interpretation": (
                    "audio-only category classifier; candidate category labels are used only "
                    "by the offline scorer to map class probabilities to choices, so this is "
                    "a diagnostic oracle control, not a deployable option predictor"
                ),
            },
            "video_only": video_metrics,
            "metadata_only": metadata_metrics,
            "joint_audio_video": joint_metrics,
            "joint_audio_video_conditional_gate": gated_joint_metrics,
            "joint_missing_audio": joint_missing_audio_metrics,
            "joint_missing_video": joint_missing_video_metrics,
            "joint_group_shuffled_audio": shuffled_audio_metrics,
            "joint_shuffled_video_option_order": shuffled_video_metrics,
            "joint_gate_missing_audio": gated_missing_audio_metrics,
            "joint_gate_missing_video": gated_missing_video_metrics,
            "joint_gate_group_shuffled_audio": gated_shuffled_audio_metrics,
            "joint_gate_shuffled_video_option_order": gated_shuffled_video_metrics,
        },
        "validation_category_metrics": category_metrics,
        "validation_source_group_metrics": source_group_metrics,
        "train_validation_gap": {
            "joint_fixed_train_nll": joint["history"][joint["best_epoch"] - 1][
                "fixed_train_evaluation"
            ]["nll"],
            "joint_validation_nll": joint_metrics["nll"],
            "joint_validation_minus_train_nll": joint_metrics["nll"]
            - joint["history"][joint["best_epoch"] - 1]["fixed_train_evaluation"]["nll"],
            "joint_gate_fixed_train_nll": gated_joint["history"][
                gated_joint["best_epoch"] - 1
            ]["fixed_train_evaluation"]["nll"],
            "joint_gate_validation_nll": gated_joint_metrics["nll"],
            "joint_gate_validation_minus_train_nll": gated_joint_metrics["nll"]
            - gated_joint["history"][gated_joint["best_epoch"] - 1][
                "fixed_train_evaluation"
            ]["nll"],
            "video_only_fixed_train_nll": video_only["history"][video_only["best_epoch"] - 1][
                "fixed_train_evaluation"
            ]["nll"],
            "video_only_validation_nll": video_metrics["nll"],
            "video_only_validation_minus_train_nll": video_metrics["nll"]
            - video_only["history"][video_only["best_epoch"] - 1]["fixed_train_evaluation"]["nll"],
            "metadata_only_fixed_train_nll": metadata_only["history"][
                metadata_only["best_epoch"] - 1
            ]["fixed_train_evaluation"]["nll"],
            "metadata_only_validation_nll": metadata_metrics["nll"],
            "metadata_only_validation_minus_train_nll": metadata_metrics["nll"]
            - metadata_only["history"][metadata_only["best_epoch"] - 1]["fixed_train_evaluation"][
                "nll"
            ],
            "audio_category_train_nll": audio_category["history"][audio_category["best_epoch"] - 1][
                "fixed_train_nll"
            ],
            "audio_category_validation_nll": audio_category["best_validation_nll"],
        },
        "group_paired_bootstrap": {
            "conditional_gate_minus_joint_accuracy": group_bootstrap_difference(
                gated_joint_correct, joint_correct, val_n["groups"]
            ),
            "conditional_gate_minus_video_only_accuracy": group_bootstrap_difference(
                gated_joint_correct, video_correct, val_n["groups"]
            ),
            "joint_minus_video_only_accuracy": group_bootstrap_difference(
                joint_correct, video_correct, val_n["groups"]
            ),
            "joint_minus_uniform_accuracy": group_bootstrap_difference(
                joint_correct,
                np.full(len(val_n["targets"]), 0.25, dtype=np.float64),
                val_n["groups"],
            ),
            "joint_minus_option_position_prior_accuracy": group_bootstrap_difference(
                joint_correct, position_prior_correct, val_n["groups"]
            ),
            "joint_minus_metadata_only_accuracy": group_bootstrap_difference(
                joint_correct, metadata_correct, val_n["groups"]
            ),
            "joint_minus_audio_only_oracle_category_accuracy": group_bootstrap_difference(
                joint_correct, audio_only_correct, val_n["groups"]
            ),
        },
        "training": {
            "joint": {
                key: value
                for key, value in joint.items()
                if key not in {"model", "selected_validation_probabilities"}
            },
            "joint_conditional_gate": {
                key: value
                for key, value in gated_joint.items()
                if key not in {"model", "selected_validation_probabilities"}
            },
            "video_only": {
                key: value
                for key, value in video_only.items()
                if key not in {"model", "selected_validation_probabilities"}
            },
            "metadata_only": {
                key: value
                for key, value in metadata_only.items()
                if key not in {"model", "selected_validation_probabilities"}
            },
            "audio_category_only": {
                key: value
                for key, value in audio_category.items()
                if key not in {"model", "best_validation_logits"}
            },
        },
        "prediction_sha256": sha256_file(output / "validation-predictions.json"),
        "effective_config_sha256": effective_config_sha256,
        "run_started_utc": run_started_utc,
        "run_completed_utc": datetime.now(UTC).isoformat(),
        "sealed_audit_loaded": False,
        "product_promotion": False,
    }
    write_json_atomic(output / "report.json", report)
    write_json_atomic(
        output / "normalization.json",
        {key: value.tolist() for key, value in normalized["normalization"].items()},
    )
    result = {
        "status": "PASS",
        "report_sha256": sha256_file(output / "report.json"),
        "prediction_sha256": report["prediction_sha256"],
        "joint_checkpoint_sha256": sha256_file(joint_dir / "best-state.pt"),
        "joint_last_checkpoint_sha256": sha256_file(joint_dir / "last-state.pt"),
        "joint_conditional_gate_checkpoint_sha256": sha256_file(
            gated_joint_dir / "best-state.pt"
        ),
        "joint_conditional_gate_last_checkpoint_sha256": sha256_file(
            gated_joint_dir / "last-state.pt"
        ),
        "video_only_checkpoint_sha256": sha256_file(video_only_dir / "best-state.pt"),
        "video_only_last_checkpoint_sha256": sha256_file(video_only_dir / "last-state.pt"),
        "metadata_only_checkpoint_sha256": sha256_file(metadata_only_dir / "best-state.pt"),
        "metadata_only_last_checkpoint_sha256": sha256_file(metadata_only_dir / "last-state.pt"),
        "audio_category_checkpoint_sha256": sha256_file(audio_only_dir / "best-state.pt"),
        "audio_category_last_checkpoint_sha256": sha256_file(audio_only_dir / "last-state.pt"),
        "sealed_audit_loaded": False,
    }
    write_json_atomic(output / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-input", type=Path, required=True)
    parser.add_argument("--validation-input", type=Path, required=True)
    parser.add_argument("--train-sidecar", type=Path, required=True)
    parser.add_argument("--validation-sidecar", type=Path, required=True)
    parser.add_argument("--audio-features", type=Path, required=True)
    parser.add_argument("--video-features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    print(json.dumps(_run(args), indent=2), flush=True)


if __name__ == "__main__":
    main()
