"""Train and evaluate a frozen-feature STARSS23 spatial Decision readout."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from scripts.extract_frozen_videoprism_features import sha256_file
from scripts.extract_starss23_corpus_features import read_manifest
from scripts.fetch_starss23_frozen_media import write_json
from scripts.preflight_starss23_spatial_decision import EVENT_CLASSES, EVENT_NAMES, SECTORS
from scripts.starss23_readout_common import binary_metrics, different_recording_permutation
from tiny_omni_decision.decision import FrozenFeatureCandidateScorer

EVENTS = [EVENT_NAMES[event_id] for event_id in EVENT_CLASSES]
EXPECTED_WINDOWS = 2720
SEED = 17
HIDDEN_SIZE = 128
BATCH_SIZE = 1024
MAX_EPOCHS = 40
PATIENCE = 6
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001


def candidate_features(
    observations: torch.Tensor, event_ids: torch.Tensor, sector_ids: torch.Tensor
) -> torch.Tensor:
    """Construct two supplied-option feature rows per event-sector question."""
    event = F.one_hot(event_ids, len(EVENTS)).to(observations.dtype)
    sector = F.one_hot(sector_ids, len(SECTORS)).to(observations.dtype)
    context = torch.cat((observations, event, sector), dim=-1)
    options = torch.eye(2, dtype=observations.dtype, device=observations.device)
    options = options.unsqueeze(0).expand(len(observations), -1, -1)
    context = context.unsqueeze(1).expand(-1, 2, -1)
    return torch.cat((context, options), dim=-1)


def balanced_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Average positive and negative CE separately to resist the no-label prior."""
    losses = F.cross_entropy(logits, targets, reduction="none")
    positive = targets == 1
    negative = targets == 0
    if not positive.any() or not negative.any():
        return losses.mean()
    return (losses[positive].mean() + losses[negative].mean()) / 2


def standardize_by_train_windows(
    features: np.ndarray, train_window_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if features.ndim != 2 or not len(train_window_indices):
        raise ValueError("training feature rows are required for standardization")
    train = features[np.unique(train_window_indices)]
    mean = train.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std > 1e-6, std, 1.0).astype(np.float32)
    normalized = ((features - mean) / std).astype(np.float32)
    if not np.isfinite(normalized).all():
        raise ValueError("nonfinite normalized features")
    return normalized, mean, std


def same_room_video_permutation(
    unique_windows: np.ndarray,
    recordings: list[str],
    rooms: list[str],
    seed: int = SEED,
) -> np.ndarray:
    if len(unique_windows) != len(recordings) or len(recordings) != len(rooms):
        raise ValueError("window, recording, and room arrays must have equal length")
    replacements = np.empty(len(unique_windows), dtype=np.int64)
    room_values = np.asarray(rooms)
    recording_values = np.asarray(recordings)
    for room in sorted(set(rooms)):
        indices = np.flatnonzero(room_values == room)
        local_order = different_recording_permutation(recording_values[indices].tolist(), seed=seed)
        replacements[indices] = unique_windows[indices[local_order]]
    if any(
        recordings[index] == recordings[np.flatnonzero(unique_windows == replacement)[0]]
        or rooms[index] != rooms[np.flatnonzero(unique_windows == replacement)[0]]
        for index, replacement in enumerate(replacements)
    ):
        raise ValueError("video permutation must preserve room and change recording")
    return replacements


def _feature_matrix(
    directory: Path,
    modality: str,
    manifest_sha: str,
    expected_shape: tuple[int, ...],
    manifest_rows: list[dict],
) -> tuple[np.ndarray, dict]:
    report = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    matrix_path = directory / "features.npy"
    if report.get("status") != "PASS" or report.get("modality") != modality:
        raise ValueError(f"{modality} feature extraction is incomplete")
    if report.get("manifest_result_sha256", report.get("manifest_sha256")) != manifest_sha:
        raise ValueError(f"{modality} features use a different window manifest")
    feature_rows = report.get("windows", report.get("features"))
    expected_rows = [
        (row["id"], row["recording"], row["split"], row["media"][modality]["sha256"])
        for row in manifest_rows
    ]
    actual_rows = [
        (row["id"], row["recording"], row["split"], row["media_sha256"])
        for row in feature_rows or []
    ]
    if actual_rows != expected_rows:
        raise ValueError(f"{modality} feature order or media identity differs from manifest")
    if sha256_file(matrix_path) != report.get("feature_sha256"):
        raise ValueError(f"{modality} feature matrix hash mismatch")
    matrix = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
    if matrix.shape != expected_shape or not np.isfinite(matrix).all():
        raise ValueError(f"invalid {modality} feature matrix shape or values")
    return np.asarray(matrix, dtype=np.float32).reshape(len(matrix), -1), report


def load_queries(
    corpus_dir: Path, windows: list[dict], manifest_sha: str
) -> tuple[dict[str, dict[str, np.ndarray]], dict]:
    result = json.loads((corpus_dir / "result.json").read_text(encoding="utf-8"))
    if result.get("status") != "PASS" or result.get("windows") != len(windows):
        raise ValueError("spatial Decision corpus is incomplete or mismatched")
    if result.get("windows_manifest_sha256") != manifest_sha:
        raise ValueError("spatial Decision corpus uses a different window manifest")
    index_by_window = {row["id"]: i for i, row in enumerate(windows)}
    manifest_by_id = {row["id"]: row for row in windows}
    if len(index_by_window) != len(windows):
        raise ValueError("duplicate fixed-window IDs")
    splits = {}
    for split in ("train", "calibration", "validation"):
        path = corpus_dir / f"{split}.jsonl"
        expected_hash = result["splits"][split]["sha256"]
        if sha256_file(path) != expected_hash:
            raise ValueError(f"{split} query corpus hash mismatch")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if len(rows) != result["splits"][split]["questions"]:
            raise ValueError(f"{split} query count mismatch")
        ids = [row["sample_id"] for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate {split} sample IDs")
        arrays = {
            "ids": np.asarray(ids),
            "window": np.asarray(
                [index_by_window[row["window_id"]] for row in rows], dtype=np.int64
            ),
            "event": np.asarray([EVENTS.index(row["event"]) for row in rows], dtype=np.int64),
            "sector": np.asarray([SECTORS.index(row["sector"]) for row in rows], dtype=np.int64),
            "target": np.asarray([row["target"] for row in rows], dtype=np.int64),
            "recording": np.asarray([row["recording"] for row in rows]),
            "room": np.asarray([row["room"] for row in rows]),
            "audio_sha256": np.asarray(
                [manifest_by_id[row["window_id"]]["media"]["audio"]["sha256"] for row in rows]
            ),
            "video_sha256": np.asarray(
                [manifest_by_id[row["window_id"]]["media"]["video"]["sha256"] for row in rows]
            ),
        }
        if not np.isin(arrays["target"], [0, 1]).all():
            raise ValueError(f"invalid {split} targets")
        if any(row["options"] != ["no", "yes"] for row in rows):
            raise ValueError(f"unexpected {split} option order")
        splits[split] = arrays
    split_recordings = {name: set(arrays["recording"].tolist()) for name, arrays in splits.items()}
    split_rooms = {name: set(arrays["room"].tolist()) for name, arrays in splits.items()}
    split_audio = {name: set(arrays["audio_sha256"].tolist()) for name, arrays in splits.items()}
    split_video = {name: set(arrays["video_sha256"].tolist()) for name, arrays in splits.items()}
    if any(
        split_recordings[a] & split_recordings[b]
        or split_rooms[a] & split_rooms[b]
        or split_audio[a] & split_audio[b]
        or split_video[a] & split_video[b]
        for a, b in (
            ("train", "calibration"),
            ("train", "validation"),
            ("calibration", "validation"),
        )
    ):
        raise ValueError("recording identity overlaps splits")
    return splits, result


def _predict(
    model, matrix: np.ndarray, rows: dict[str, np.ndarray], device: torch.device
) -> torch.Tensor:
    output = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(rows["target"]), 2048):
            end = min(start + 2048, len(rows["target"]))
            indices = slice(start, end)
            observations = torch.as_tensor(matrix[rows["window"][indices]], device=device)
            event_ids = torch.as_tensor(rows["event"][indices], device=device)
            sector_ids = torch.as_tensor(rows["sector"][indices], device=device)
            output.append(
                model(candidate_features(observations, event_ids, sector_ids)).float().cpu()
            )
    return torch.cat(output, dim=0)


def _macro_event_sector_nll(
    logits: torch.Tensor, targets: torch.Tensor, events: np.ndarray, sectors: np.ndarray
) -> float:
    losses = F.cross_entropy(logits, targets, reduction="none").numpy()
    values = [
        float(losses[(events == event_id) & (sectors == sector_id)].mean())
        for event_id in range(len(EVENTS))
        for sector_id in range(len(SECTORS))
        if ((events == event_id) & (sectors == sector_id)).any()
    ]
    if len(values) != len(EVENTS) * len(SECTORS):
        raise ValueError("calibration is missing one or more event-sector groups")
    return float(np.mean(values))


def _calibrate(
    logits: torch.Tensor,
    targets: np.ndarray,
    events: np.ndarray,
    sectors: np.ndarray,
) -> tuple[float, float, list[dict]]:
    """Fit a tiny temperature/intercept grid on calibration only."""
    y = torch.as_tensor(targets, dtype=torch.long)
    temperatures = np.geomspace(0.5, 4.0, 15)
    biases = np.linspace(-8.0, 0.0, 33)
    candidates = []
    for temperature in temperatures:
        for bias in biases:
            adjusted = logits / float(temperature)
            adjusted[:, 1] += float(bias)
            macro_nll = _macro_event_sector_nll(adjusted, y, events, sectors)
            natural_nll = float(F.cross_entropy(adjusted, y))
            candidates.append(
                {
                    "temperature": float(temperature),
                    "yes_bias": float(bias),
                    "macro_event_sector_nll": macro_nll,
                    "natural_nll": natural_nll,
                }
            )
    selected = min(
        candidates,
        key=lambda row: (
            row["macro_event_sector_nll"],
            row["temperature"],
            row["yes_bias"],
        ),
    )
    return selected["temperature"], selected["yes_bias"], candidates


def _metric_summary(
    targets: np.ndarray,
    probabilities: np.ndarray,
    events: np.ndarray,
    sectors: np.ndarray,
    rooms: np.ndarray | None = None,
) -> dict:
    overall = binary_metrics(targets, probabilities)
    event_metrics = {
        name: binary_metrics(targets[events == i], probabilities[events == i])
        for i, name in enumerate(EVENTS)
    }
    sector_metrics = {
        name: binary_metrics(targets[sectors == i], probabilities[sectors == i])
        for i, name in enumerate(SECTORS)
    }
    pair_metrics = {
        f"{event}|{sector}": binary_metrics(
            targets[(events == ei) & (sectors == si)],
            probabilities[(events == ei) & (sectors == si)],
        )
        for ei, event in enumerate(EVENTS)
        for si, sector in enumerate(SECTORS)
    }
    macro_keys = ("balanced_accuracy", "nll", "brier", "ece")

    def macro(values: dict[str, dict], key: str) -> float | None:
        observed = [value[key] for value in values.values() if value[key] is not None]
        return float(np.mean(observed)) if observed else None

    result = {
        "overall": overall,
        "macro_event": {key: macro(event_metrics, key) for key in macro_keys},
        "macro_sector": {key: macro(sector_metrics, key) for key in macro_keys},
        "macro_event_sector": {key: macro(pair_metrics, key) for key in macro_keys},
        "per_event": event_metrics,
        "per_sector": sector_metrics,
        "per_event_sector": pair_metrics,
    }
    if rooms is not None:
        result["per_room"] = {
            str(room): binary_metrics(targets[rooms == room], probabilities[rooms == room])
            for room in sorted(set(rooms.tolist()))
        }
    return result


def _fit_arm(
    name: str,
    matrix: np.ndarray,
    splits: dict[str, dict[str, np.ndarray]],
    config: dict,
    out: Path,
    device: torch.device,
    normalization: tuple[np.ndarray, np.ndarray],
    control_matrices: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]] | None = None,
) -> dict:
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = FrozenFeatureCandidateScorer(
        matrix.shape[1] + len(EVENTS) + len(SECTORS) + 2, HIDDEN_SIZE
    ).to(device)
    np.savez_compressed(out / "normalization.npz", mean=normalization[0], std=normalization[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    train, calibration, validation = (splits[key] for key in ("train", "calibration", "validation"))
    targets = torch.as_tensor(train["target"], dtype=torch.long, device=device)
    order_generator = torch.Generator(device="cpu").manual_seed(SEED)
    best_nll = math.inf
    best_epoch = 0
    best_state = None
    stale = 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_started = time.perf_counter()
        model.train()
        order = torch.randperm(len(targets), generator=order_generator)
        losses = []
        for batch in order.split(BATCH_SIZE):
            ids = batch.to(device)
            observations = torch.as_tensor(matrix[train["window"][batch.numpy()]], device=device)
            event_ids = torch.as_tensor(train["event"][batch.numpy()], device=device)
            sector_ids = torch.as_tensor(train["sector"][batch.numpy()], device=device)
            logits = model(candidate_features(observations, event_ids, sector_ids))
            loss = balanced_cross_entropy(logits, targets[ids])
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(
                parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            ):
                raise ValueError("nonfinite gradient")
            optimizer.step()
            losses.append(float(loss.detach()))
        calibration_logits = _predict(model, matrix, calibration, device)
        calibration_nll = float(
            F.cross_entropy(
                calibration_logits, torch.as_tensor(calibration["target"], dtype=torch.long)
            )
        )
        selector_nll = _macro_event_sector_nll(
            calibration_logits,
            torch.as_tensor(calibration["target"], dtype=torch.long),
            calibration["event"],
            calibration["sector"],
        )
        history.append(
            {
                "epoch": epoch,
                "rolling_train_balanced_ce": float(np.mean(losses)),
                "calibration_nll": calibration_nll,
                "calibration_macro_event_sector_nll": selector_nll,
                "epoch_seconds": time.perf_counter() - epoch_started,
            }
        )
        if selector_nll < best_nll - 1e-5:
            best_nll = selector_nll
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            torch.save(best_state, out / "best.pt")
            stale = 0
        else:
            stale += 1
        write_json(out / "history.json", history)
        if stale >= PATIENCE:
            break
    training_seconds = time.perf_counter() - started
    evaluation_started = time.perf_counter()
    if best_state is None:
        raise ValueError("no valid checkpoint selected")
    torch.save(model.state_dict(), out / "final.pt")
    model.load_state_dict(torch.load(out / "best.pt", map_location=device, weights_only=True))
    if any(
        not torch.equal(value.detach().cpu(), best_state[key])
        for key, value in model.state_dict().items()
    ):
        raise ValueError("best checkpoint reload mismatch")
    calibration_logits = _predict(model, matrix, calibration, device)
    temperature, yes_bias, calibration_grid = _calibrate(
        calibration_logits,
        calibration["target"],
        calibration["event"],
        calibration["sector"],
    )
    training_logits = _predict(model, matrix, train, device)
    calibrated_training_logits = training_logits / temperature
    calibrated_training_logits[:, 1] += yes_bias
    raw_training_probabilities = training_logits.softmax(-1)[:, 1].numpy()
    calibrated_training_probabilities = calibrated_training_logits.softmax(-1)[:, 1].numpy()
    validation_logits = _predict(model, matrix, validation, device)
    calibrated_logits = validation_logits / temperature
    calibrated_logits[:, 1] += yes_bias
    raw_probabilities = validation_logits.softmax(-1)[:, 1].numpy()
    calibrated_probabilities = calibrated_logits.softmax(-1)[:, 1].numpy()
    controls = {}
    control_predictions = {}
    for control_name, (control_matrix, control_rows) in (control_matrices or {}).items():
        control_logits = _predict(model, control_matrix, control_rows, device)
        control_logits[:, 1] += yes_bias * temperature
        control_probabilities = (control_logits / temperature).softmax(-1)[:, 1].numpy()
        controls[control_name] = _metric_summary(
            control_rows["target"],
            control_probabilities,
            control_rows["event"],
            control_rows["sector"],
            control_rows["room"],
        )
        control_predictions[f"{control_name}_positive_probabilities"] = control_probabilities
    np.savez_compressed(
        out / "validation_predictions.npz",
        sample_ids=validation["ids"],
        recording=validation["recording"],
        room=validation["room"],
        targets=validation["target"],
        event_ids=validation["event"],
        sector_ids=validation["sector"],
        raw_positive_probabilities=raw_probabilities,
        calibrated_positive_probabilities=calibrated_probabilities,
        **control_predictions,
    )
    return {
        "arm": name,
        "feature_dimension": int(matrix.shape[1]),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "best_epoch": best_epoch,
        "stopped_epoch": len(history),
        "selection": "minimum raw calibration macro event-sector NLL; patience 6; min_delta 1e-5",
        "calibration": {"temperature": temperature, "yes_bias": yes_bias, "grid": calibration_grid},
        "train_fixed_raw": _metric_summary(
            train["target"],
            raw_training_probabilities,
            train["event"],
            train["sector"],
            train["room"],
        ),
        "train_fixed_calibrated": _metric_summary(
            train["target"],
            calibrated_training_probabilities,
            train["event"],
            train["sector"],
            train["room"],
        ),
        "validation_raw": _metric_summary(
            validation["target"],
            raw_probabilities,
            validation["event"],
            validation["sector"],
            validation["room"],
        ),
        "validation_calibrated": _metric_summary(
            validation["target"],
            calibrated_probabilities,
            validation["event"],
            validation["sector"],
            validation["room"],
        ),
        "validation_input_controls": controls,
        "best_sha256": sha256_file(out / "best.pt"),
        "final_sha256": sha256_file(out / "final.pt"),
        "prediction_sha256": sha256_file(out / "validation_predictions.npz"),
        "normalization_sha256": sha256_file(out / "normalization.npz"),
        "training_wall_seconds": time.perf_counter() - started,
        "training_seconds": training_seconds,
        "evaluation_and_save_seconds": time.perf_counter() - evaluation_started,
        "peak_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
        "peak_cuda_reserved_bytes": (
            torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
        ),
        "history": history,
    }


def train(args: argparse.Namespace) -> dict:
    manifest_rows = read_manifest(args.manifest_dir)
    if len(manifest_rows) != EXPECTED_WINDOWS:
        raise ValueError("fixed STARSS23 window manifest count changed")
    manifest_sha = sha256_file(args.manifest_dir / "result.json")
    audio, audio_report = _feature_matrix(
        args.audio_features,
        "audio",
        manifest_sha,
        (EXPECTED_WINDOWS, 4, 384),
        manifest_rows,
    )
    video, video_report = _feature_matrix(
        args.video_features,
        "video",
        manifest_sha,
        (EXPECTED_WINDOWS, 8, 768),
        manifest_rows,
    )
    splits, corpus_report = load_queries(args.corpus_dir, manifest_rows, manifest_sha)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    train_windows = splits["train"]["window"]
    joint_raw = np.concatenate((audio, video), axis=1)
    audio, audio_mean, audio_std = standardize_by_train_windows(audio, train_windows)
    video, video_mean, video_std = standardize_by_train_windows(video, train_windows)
    joint, joint_mean, joint_std = standardize_by_train_windows(joint_raw, train_windows)
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = {
        "seed": SEED,
        "hidden_size": HIDDEN_SIZE,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "loss": ("equal positive/negative CE; calibration and validation use natural frequencies"),
        "feature_normalization": "mean/std fit on unique train windows only",
        "arms": ["audio_only", "video_only", "audio_video_joint"],
        "device": str(args.device),
    }
    effective = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    write_json(args.output_dir / "effective_config.json", config)
    write_json(
        args.output_dir / "run_inputs.json",
        {
            "manifest_result_sha256": manifest_sha,
            "runner_sha256": sha256_file(Path(__file__)),
            "corpus_result_sha256": sha256_file(args.corpus_dir / "result.json"),
            "audio_result_sha256": sha256_file(args.audio_features / "result.json"),
            "audio_feature_sha256": audio_report["feature_sha256"],
            "audio_checkpoint_sha256": audio_report.get("checkpoint_sha256"),
            "audio_source_commit": audio_report.get("source_commit"),
            "video_result_sha256": sha256_file(args.video_features / "result.json"),
            "video_feature_sha256": video_report["feature_sha256"],
            "video_checkpoint_sha256": video_report.get("checkpoint_sha256"),
            "video_source_commit": video_report.get("videoprism_source_commit"),
            "config_sha256": hashlib.sha256(effective).hexdigest(),
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
            ).strip(),
            "command": [sys.executable, *sys.argv],
            "working_directory": str(Path.cwd()),
            "splits": {
                name: {
                    "questions": len(rows["target"]),
                    "positives": int(rows["target"].sum()),
                    "recordings": len(set(rows["recording"].tolist())),
                    "rooms": sorted(set(rows["room"].tolist())),
                    "unique_windows": len(set(rows["window"].tolist())),
                    "unique_audio_assets": len(set(rows["audio_sha256"].tolist())),
                    "unique_video_assets": len(set(rows["video_sha256"].tolist())),
                }
                for name, rows in splits.items()
            },
            "training": True,
            "official_dev_test_or_sealed_audit_read": False,
        },
    )
    results = {}
    write_json(
        args.output_dir / "progress.json",
        {"status": "RUNNING", "current_arm": "audio_only", "completed_arms": []},
    )
    validation = splits["validation"]
    unique_windows = np.unique(validation["window"])
    recording_by_window = [
        validation["recording"][np.flatnonzero(validation["window"] == window)[0]]
        for window in unique_windows
    ]
    room_by_window = [
        validation["room"][np.flatnonzero(validation["window"] == window)[0]]
        for window in unique_windows
    ]
    replacement_windows = same_room_video_permutation(
        unique_windows, recording_by_window, room_by_window, seed=SEED
    )
    replacement = {
        int(window): int(replacement_windows[index]) for index, window in enumerate(unique_windows)
    }
    permuted_validation = dict(validation)
    permuted_validation["window"] = np.asarray(
        [replacement[int(window)] for window in validation["window"]], dtype=np.int64
    )
    joint_controls = {
        "video_zeroed": (np.concatenate((audio, np.zeros_like(video)), axis=1), validation),
        "audio_zeroed": (np.concatenate((np.zeros_like(audio), video), axis=1), validation),
        "video_from_different_recording": (joint, permuted_validation),
    }
    for arm, matrix, normalization in (
        ("audio_only", audio, (audio_mean, audio_std)),
        ("video_only", video, (video_mean, video_std)),
        ("audio_video_joint", joint, (joint_mean, joint_std)),
    ):
        arm_dir = args.output_dir / arm
        arm_dir.mkdir()
        results[arm] = _fit_arm(
            arm,
            matrix,
            splits,
            config,
            arm_dir,
            device,
            normalization,
            joint_controls if arm == "audio_video_joint" else None,
        )
        write_json(
            args.output_dir / "progress.json",
            {
                "status": "RUNNING",
                "current_arm": "complete" if arm == "audio_video_joint" else "next",
                "completed_arms": list(results),
            },
        )
    no_positive = {split: float((rows["target"] == 0).mean()) for split, rows in splits.items()}
    baselines = {
        split: {
            "always_no": _metric_summary(
                rows["target"],
                np.zeros(len(rows["target"]), dtype=np.float32),
                rows["event"],
                rows["sector"],
                rows["room"],
            ),
            "uniform": _metric_summary(
                rows["target"],
                np.full(len(rows["target"]), 0.5, dtype=np.float32),
                rows["event"],
                rows["sector"],
                rows["room"],
            ),
        }
        for split, rows in splits.items()
    }
    result = {
        "status": "PASS",
        "source": "STARSS23 DOI 10.5281/zenodo.7880637; pinned dev-train metadata only",
        "summary": (
            "room-disjoint STARSS23 event-sector binary Decision development validation; "
            "not a blind audit or product quality claim"
        ),
        "class_prior_baseline_always_no_accuracy": no_positive,
        "baselines": baselines,
        "arms": results,
        "runtime": {
            "torch": torch.__version__,
            "torch_cuda_runtime": torch.version.cuda,
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "logical_cpu_count": os.cpu_count(),
            "cuda_available": torch.cuda.is_available(),
            "device": str(device),
            "cuda_device_name": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None,
            "cuda_device_total_bytes": (
                torch.cuda.get_device_properties(device).total_memory
                if device.type == "cuda"
                else None
            ),
        },
        "training_examples_by_split": {
            name: int(len(rows["target"])) for name, rows in splits.items()
        },
        "training_examples_unique_queries": {
            name: int(len(set(rows["ids"].tolist()))) for name, rows in splits.items()
        },
        "frozen_encoder_parameters": {
            "audio_dymn": audio_report.get("parameters"),
            "video_videoprism": video_report.get("parameters"),
        },
        "corpus_summary": corpus_report["splits"],
        "official_dev_test_or_sealed_audit_read": False,
    }
    write_json(args.output_dir / "result.json", result)
    write_json(
        args.output_dir / "progress.json", {"status": "PASS", "completed_arms": list(results)}
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest-dir", "audio-features", "video-features", "corpus-dir", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    print(json.dumps(train(args), indent=2))


if __name__ == "__main__":
    main()
