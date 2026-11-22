"""Fit one frozen-encoder, supplied-option audio+image Decision pilot."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch.nn import functional as F

from scripts.spoken_palette_decision_data import COLORS, POSITIONS
from scripts.train_miniwob_dino_path_a import (
    encode_screen_features,
    load_encoder,
)
from tiny_omni_decision.decision import FrozenFeatureCandidateScorer
from tiny_omni_decision.miniwob_dino_readout import permute_candidate_axis

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/pretrained_reuse/spoken_palette_decision_v1.json"
SCREEN_CONFIG_PATH = ROOT / "configs/pretrained_reuse/miniwob_dino_path_a_v1.json"
FEATURE_WIDTH = 1538
VISUAL_WIDTH = 1154
HIDDEN_WIDTH = 64
TRAINABLE_PARAMETERS = 98_561


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_jsonl(path: Path, expected_sha256: str) -> list[dict[str, Any]]:
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"frozen JSONL hash mismatch: {path.name}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite number cannot enter a JSON artifact")
    return value


def write_json(path: Path, value: Any) -> str:
    payload = (
        json.dumps(json_safe(value), ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode()
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    payload = "".join(
        json.dumps(json_safe(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    ).encode("utf-8")
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def sample_peak_rss(process: Any) -> int:
    return int(process.memory_info().rss)


def unique_screens(
    rows: list[dict[str, Any]], root: Path
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    screens: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = row["screenshot_sha256"]
        existing = screens.get(key)
        if existing is not None and existing["screenshot_path"] != row["screenshot_path"]:
            raise ValueError("one screenshot hash resolves to different relative paths")
        screens[key] = row
    ordered = sorted(screens.values(), key=lambda row: row["screenshot_sha256"])
    index_by_hash = {row["screenshot_sha256"]: index for index, row in enumerate(ordered)}
    for row in ordered:
        image_path = (root / row["screenshot_path"]).resolve(strict=True)
        if (
            root.resolve() not in image_path.parents
            or sha256_file(image_path) != row["screenshot_sha256"]
        ):
            raise ValueError("generated screenshot path/hash binding failed")
    return ordered, index_by_hash


def canonical_screen_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    canonical = []
    for index, row in enumerate(rows):
        options = [
            {"option_id": f"position-{position}", "x_norm": x, "y_norm": y}
            for position, (x, y) in enumerate(POSITIONS)
        ]
        canonical.append(
            {
                "sample_id": f"screen-{index}-{row['screenshot_sha256'][:12]}",
                "episode_group_id": row["screenshot_sha256"],
                "task_template": "spoken-color-to-screen-action-v1",
                "seed": 17,
                "screenshot_path": row["screenshot_path"],
                "screenshot_sha256": row["screenshot_sha256"],
                "screenshot_shape": row["screenshot_shape"],
                "option_order_seed": 17,
                "options": options,
                "target_index": 0,
            }
        )
    return canonical


def encode_screens(
    rows: list[dict[str, Any]], corpus_root: Path, config: dict[str, Any], process: Any
):
    unique, index_by_hash = unique_screens(rows, corpus_root)
    model, model_dir, hashes = load_encoder(config)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("DINOv2 screen encoder must remain frozen")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != config["encoder"]["parameters"]:
        raise ValueError("DINOv2 loaded parameter count differs from the pinned config")
    start = time.perf_counter()
    screen_features = encode_screen_features(canonical_screen_rows(unique), corpus_root, model)
    elapsed = time.perf_counter() - start
    values = screen_features.detach().cpu().numpy().astype(np.float32, copy=True)
    if values.shape != (len(unique), 4, VISUAL_WIDTH) or not np.isfinite(values).all():
        raise ValueError("DINOv2 frozen option-feature shape/value check failed")
    process._spoken_palette_peak_rss = max(
        getattr(process, "_spoken_palette_peak_rss", 0), sample_peak_rss(process)
    )
    return (
        unique,
        index_by_hash,
        values,
        {
            "model_dir": str(model_dir),
            "model_parameter_count": parameter_count,
            "model_file_hashes": hashes,
            "unique_screens": len(unique),
            "screen_features_shape": list(values.shape),
            "feature_bytes_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
            "feature_extraction_seconds": elapsed,
        },
    )


def arrays_for_rows(
    rows: list[dict[str, Any]],
    screen_index_by_hash: dict[str, int],
    color_by_source: dict[str, str],
) -> dict[str, Any]:
    audio_indices = np.asarray([row["audio_feature_index"] for row in rows], dtype=np.int64)
    screen_indices = np.asarray(
        [screen_index_by_hash[row["screenshot_sha256"]] for row in rows], dtype=np.int64
    )
    option_positions = np.asarray(
        [
            [POSITIONS.index((option["x_norm"], option["y_norm"])) for option in row["options"]]
            for row in rows
        ],
        dtype=np.int64,
    )
    targets = np.asarray([row["target_index"] for row in rows], dtype=np.int64)
    colors = np.asarray([color_by_source[row["source_id"]] for row in rows], dtype=object)
    if (
        option_positions.shape != (len(rows), 4)
        or not np.all(np.sort(option_positions, axis=1) == np.arange(4))
        or np.any(targets < 0)
        or np.any(targets > 3)
    ):
        raise ValueError("candidate positions or supplied targets are invalid")
    return {
        "rows": rows,
        "audio_indices": audio_indices,
        "screen_indices": screen_indices,
        "option_positions": option_positions,
        "targets": targets,
        "colors": colors,
        "source_ids": np.asarray([row["source_id"] for row in rows], dtype=object),
        "group_ids": np.asarray([row["speaker_group"] for row in rows], dtype=object),
    }


def tensor_batch(
    indices: np.ndarray,
    arrays: dict[str, Any],
    audio_bank: np.ndarray,
    visual_bank: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    missing_audio: bool = False,
    missing_visual: bool = False,
    audio_donor: np.ndarray | None = None,
    screen_donor: np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected = indices
    audio_indices = arrays["audio_indices"][selected]
    screen_indices = arrays["screen_indices"][selected]
    if audio_donor is not None:
        audio_indices = audio_donor[selected]
    if screen_donor is not None:
        screen_indices = screen_donor[selected]
    option_positions = arrays["option_positions"][selected]
    audio = np.asarray(audio_bank[audio_indices], dtype=np.float32)
    visual = np.asarray(visual_bank[screen_indices[:, None], option_positions], dtype=np.float32)
    if missing_audio:
        audio = np.broadcast_to(mean[:384], audio.shape).copy()
    if missing_visual:
        visual = visual.copy()
        visual[:, :, :1152] = mean[384 : 384 + 1152]
        # Preserve candidate geometry while removing screen-derived visual content.
    repeated_audio = np.broadcast_to(audio[:, None, :], (len(selected), 4, 384))
    features = np.concatenate((repeated_audio, visual), axis=-1)
    features = np.asarray((features - mean) / std, dtype=np.float32)
    return torch.from_numpy(features), torch.from_numpy(arrays["targets"][selected].copy())


def metrics(probabilities: np.ndarray, targets: np.ndarray, colors: np.ndarray) -> dict[str, Any]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    if probabilities.shape != (len(targets), 4) or not np.isfinite(probabilities).all():
        raise ValueError("probability matrix must be finite [N,4]")
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predictions == targets
    one_hot = np.eye(4, dtype=np.float64)[targets]
    ece = 0.0
    for index in range(15):
        lower, upper = index / 15, (index + 1) / 15
        mask = (confidence >= lower) & (confidence <= upper if index == 14 else confidence < upper)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    per_color = {}
    for color in COLORS:
        mask = colors == color
        if not mask.any():
            raise ValueError(f"evaluation split has no examples for color {color}")
        per_color[color] = {
            "count": int(mask.sum()),
            "accuracy": float(correct[mask].mean()),
            "nll": float(-np.log(np.clip(probabilities[mask, targets[mask]], 1e-12, 1)).mean()),
        }
    return {
        "count": int(len(targets)),
        "accuracy": float(correct.mean()),
        "macro_color_accuracy": float(np.mean([per_color[c]["accuracy"] for c in COLORS])),
        "nll": float(
            -np.log(np.clip(probabilities[np.arange(len(targets)), targets], 1e-12, 1)).mean()
        ),
        "macro_color_nll": float(np.mean([per_color[c]["nll"] for c in COLORS])),
        "brier": float(np.square(probabilities - one_hot).sum(axis=1).mean()),
        "ece_15": float(ece),
        "mean_confidence": float(confidence.mean()),
        "per_color": per_color,
    }


def infer(
    model: torch.nn.Module,
    arrays: dict[str, Any],
    audio_bank: np.ndarray,
    visual_bank: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    temperature: float = 1.0,
    missing_audio: bool = False,
    missing_visual: bool = False,
    audio_donor: np.ndarray | None = None,
    screen_donor: np.ndarray | None = None,
    batch_size: int = 512,
) -> np.ndarray:
    model.eval()
    all_probabilities = []
    with torch.inference_mode():
        for start in range(0, len(arrays["targets"]), batch_size):
            indices = np.arange(start, min(start + batch_size, len(arrays["targets"])))
            features, _ = tensor_batch(
                indices,
                arrays,
                audio_bank,
                visual_bank,
                mean,
                std,
                missing_audio=missing_audio,
                missing_visual=missing_visual,
                audio_donor=audio_donor,
                screen_donor=screen_donor,
            )
            logits = model(features).float() / temperature
            all_probabilities.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(all_probabilities, axis=0)


def fit(args) -> dict[str, Any]:
    import psutil

    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config_sha = sha256_file(CONFIG_PATH)
    corpus_root = Path(args.corpus_root).resolve(strict=True)
    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite fit output: {output_root}")
    if (
        os.path.splitdrive(str(output_root))[0].lower() == "c:"
        and __import__("shutil").disk_usage(output_root.parent).free < 2 * 1024**3
    ):
        raise RuntimeError("C: must retain at least 2 GiB free")
    manifest_path = corpus_root / "manifest.json"
    manifest_sha = sha256_file(manifest_path)
    corpus = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        corpus.get("status") != "TRAIN_ONLY_FACTORIAL_PREFLIGHT_PASS_NO_MODEL_FIT"
        or corpus.get("model_fit_started") is not False
        or corpus.get("source", {}).get("source_manifest_sha256")
        != cfg["source"]["source_manifest_sha256"]
    ):
        raise ValueError("the frozen training-only factorial corpus manifest is not admitted")
    row_files = corpus["split_row_sha256"]
    # Evaluation labels/assets stay closed until model and calibration freeze.
    train_rows = read_jsonl(corpus_root / "train.jsonl", row_files["train"])
    selection_rows = read_jsonl(corpus_root / "selection.jsonl", row_files["selection"])
    calibration_rows = read_jsonl(corpus_root / "calibration.jsonl", row_files["calibration"])
    train_original = json.loads(
        (
            Path(cfg["source"]["audio_feature_path"]).parent
            / "lombard-grid-layout-02-media-gated"
            / "train.json"
        ).read_text(encoding="utf-8")
    )
    color_by_source = {row["source_id"]: row["targets"]["color"] for row in train_original}
    process = psutil.Process()
    process._spoken_palette_peak_rss = sample_peak_rss(process)

    audio_root = Path(cfg["source"]["audio_feature_path"]).resolve(strict=True)
    audio_bank_path = audio_root / "pooled.npy"
    if sha256_file(audio_bank_path) != cfg["source"]["audio_pooled_sha256"]:
        raise ValueError("frozen Whisper feature bank changed")
    audio_info = json.loads((audio_root / "result.json").read_text(encoding="utf-8"))
    if audio_info.get("status") != "PASS" or audio_info.get("training") is not False:
        raise ValueError("frozen Whisper bank lacks a verified inference-only receipt")
    audio_bank = np.load(audio_bank_path, mmap_mode="r", allow_pickle=False)
    if audio_bank.shape != (len(audio_info["rows"]), 384) or audio_bank.dtype != np.float32:
        raise ValueError("frozen Whisper feature bank shape/dtype differs")

    screen_cfg_path = ROOT / cfg["source"]["screen_encoder_config"]
    if sha256_file(screen_cfg_path) != cfg["source"]["screen_encoder_config_sha256"]:
        raise ValueError("pinned DINOv2 encoder config changed")
    screen_cfg = json.loads(screen_cfg_path.read_text(encoding="utf-8"))

    output_root.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=False, exist_ok=False)
    # Encode only the fitting/selection/calibration screens before freezing the candidate.
    dev_rows = train_rows + selection_rows + calibration_rows
    screen_encoder_start = time.perf_counter()
    _, dev_screen_indices, dev_visual, dev_screen_info = encode_screens(
        dev_rows, corpus_root, screen_cfg, process
    )
    screen_encoder_seconds = time.perf_counter() - screen_encoder_start
    dev_feature_path = output_root / "dev-screen-features.npy"
    np.save(dev_feature_path, dev_visual, allow_pickle=False)
    dev_feature_sha = sha256_file(dev_feature_path)

    train_arrays = arrays_for_rows(train_rows, dev_screen_indices, color_by_source)
    selection_arrays = arrays_for_rows(selection_rows, dev_screen_indices, color_by_source)
    calibration_arrays = arrays_for_rows(calibration_rows, dev_screen_indices, color_by_source)
    # Normalizer excludes all non-training assets, even though they share the dev feature matrix.
    train_screen_indices = np.unique(train_arrays["screen_indices"])
    train_audio_indices = np.unique(train_arrays["audio_indices"])
    train_visual = dev_visual[train_screen_indices].reshape(-1, VISUAL_WIDTH)
    unique_audio = np.asarray(audio_bank[train_audio_indices], dtype=np.float32)
    mean = np.concatenate((unique_audio.mean(axis=0), train_visual.mean(axis=0))).astype(np.float32)
    std = np.concatenate((unique_audio.std(axis=0), train_visual.std(axis=0))).astype(np.float32)
    std[std <= 1e-6] = 1.0

    effective = {
        **cfg,
        "config_path": str(CONFIG_PATH),
        "config_sha256": config_sha,
        "corpus_root": str(corpus_root),
        "corpus_manifest_sha256": manifest_sha,
        "train_jsonl_sha256": row_files["train"],
        "selection_jsonl_sha256": row_files["selection"],
        "calibration_jsonl_sha256": row_files["calibration"],
        "dev_screen_feature_sha256": dev_feature_sha,
        "dev_screen_feature_info": dev_screen_info,
        "screen_encoder_seconds": screen_encoder_seconds,
    }
    write_json(output_root / "effective-config.json", effective)
    np.savez(output_root / "train-normalizer.npz", mean=mean, std=std)
    normalizer_sha = sha256_file(output_root / "train-normalizer.npz")

    torch.set_num_threads(4)
    torch.manual_seed(17)
    np.random.seed(17)
    random.seed(17)
    model = FrozenFeatureCandidateScorer(FEATURE_WIDTH, hidden_size=HIDDEN_WIDTH).cpu()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != TRAINABLE_PARAMETERS:
        raise ValueError(f"scorer parameter count mismatch: {parameter_count}")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    max_epochs = cfg["training"]["max_epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    batch_size = cfg["training"]["batch_size"]
    min_delta = cfg["training"]["selector_min_delta"]
    patience = cfg["training"]["early_stopping_patience"]
    rng = np.random.default_rng(17)
    best_score = float("inf")
    best_epoch = None
    bad_epochs = 0
    history: list[dict[str, Any]] = []
    started = datetime.now(UTC)
    fit_start = time.perf_counter()
    peak_rss = process._spoken_palette_peak_rss
    for epoch in range(1, max_epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        order = rng.permutation(len(train_rows))
        train_loss_sum = 0.0
        train_correct = 0
        train_count = 0
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            features, targets = tensor_batch(
                indices, train_arrays, audio_bank, dev_visual, mean, std
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = F.cross_entropy(logits, targets)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite training loss; safe stop")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if not torch.isfinite(grad_norm):
                raise RuntimeError("non-finite gradient norm; safe stop")
            optimizer.step()
            train_loss_sum += float(loss.detach()) * len(indices)
            train_correct += int((logits.detach().argmax(dim=1) == targets).sum())
            train_count += len(indices)
        scheduler.step()
        selection_probabilities = infer(model, selection_arrays, audio_bank, dev_visual, mean, std)
        selection_metrics = metrics(
            selection_probabilities,
            selection_arrays["targets"],
            selection_arrays["colors"],
        )
        train_probabilities = infer(model, train_arrays, audio_bank, dev_visual, mean, std)
        train_metrics = metrics(
            train_probabilities, train_arrays["targets"], train_arrays["colors"]
        )
        score = selection_metrics["macro_color_nll"]
        improved = score < best_score - min_delta
        if improved:
            best_score = score
            best_epoch = epoch
            bad_epochs = 0
            save_file(
                {
                    key: value.detach().cpu().contiguous()
                    for key, value in model.state_dict().items()
                },
                str(output_root / "checkpoint-best.safetensors"),
            )
        else:
            bad_epochs += 1
        history_row = {
            "epoch": epoch,
            "optimizer_updates": math.ceil(len(train_rows) / batch_size) * epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_fit_rolling_ce": train_loss_sum / max(train_count, 1),
            "train_accuracy": train_correct / max(train_count, 1),
            "train_fixed_accuracy": train_metrics["accuracy"],
            "train_fixed_macro_color_accuracy": train_metrics["macro_color_accuracy"],
            "train_fixed_macro_color_nll": train_metrics["macro_color_nll"],
            "selection": selection_metrics,
            "selected": improved,
            "best_epoch_so_far": best_epoch,
            "best_selection_macro_color_nll": best_score,
            "early_stopping_wait": bad_epochs,
            "seconds": time.perf_counter() - epoch_start,
            "gradient_norm_last_batch": float(grad_norm.detach()),
        }
        history.append(history_row)
        write_jsonl(output_root / "training-history.jsonl", history)
        peak_rss = max(peak_rss, sample_peak_rss(process))
        if bad_epochs >= patience:
            break
    if best_epoch is None:
        raise RuntimeError("no finite selection checkpoint was produced")

    save_file(
        {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()},
        str(output_root / "checkpoint-final.safetensors"),
    )
    selected = FrozenFeatureCandidateScorer(FEATURE_WIDTH, hidden_size=HIDDEN_WIDTH).cpu()
    selected.load_state_dict(
        load_file(output_root / "checkpoint-best.safetensors", device="cpu"), strict=True
    )
    selected.eval()
    calibration_probabilities_raw = infer(
        selected, calibration_arrays, audio_bank, dev_visual, mean, std
    )
    temperatures = np.geomspace(0.25, 4.0, 81)
    calibration_logits = np.log(np.clip(calibration_probabilities_raw, 1e-12, 1.0))
    calibration_scores = []
    for temperature in temperatures:
        probabilities = torch.softmax(
            torch.from_numpy(calibration_logits / float(temperature)), dim=-1
        ).numpy()
        calibration_scores.append(
            metrics(probabilities, calibration_arrays["targets"], calibration_arrays["colors"])[
                "macro_color_nll"
            ]
        )
    selected_temperature = float(temperatures[int(np.argmin(calibration_scores))])
    best_sha = sha256_file(output_root / "checkpoint-best.safetensors")
    calibration_record = {
        "temperature": selected_temperature,
        "grid_min": float(temperatures[0]),
        "grid_max": float(temperatures[-1]),
        "grid_count": len(temperatures),
        "calibration_rows": len(calibration_rows),
        "calibration_groups": sorted({row["speaker_group"] for row in calibration_rows}),
        "best_checkpoint_sha256": best_sha,
        "selector": "minimum macro-color calibration NLL; earliest grid temperature on ties",
    }
    calibration_sha = write_json(output_root / "calibration.json", calibration_record)
    freeze = {
        "status": "CANDIDATE_AND_TEMPERATURE_FROZEN_BEFORE_EVALUATION_SPLIT_ACCESS",
        "frozen_utc": datetime.now(UTC).isoformat(),
        "best_epoch": best_epoch,
        "best_checkpoint_sha256": best_sha,
        "final_checkpoint_sha256": sha256_file(output_root / "checkpoint-final.safetensors"),
        "normalizer_sha256": normalizer_sha,
        "calibration_sha256": calibration_sha,
        "temperature": selected_temperature,
        "training_history_sha256": sha256_file(output_root / "training-history.jsonl"),
        "config_sha256": effective["config_sha256"],
        "evaluation_jsonl_sha256": row_files["evaluation"],
    }
    freeze_sha = write_json(output_root / "frozen-candidate.json", freeze)

    # Evaluation split is first opened only after selected weights and calibration are frozen.
    evaluation_rows = read_jsonl(corpus_root / "evaluation.jsonl", row_files["evaluation"])
    evaluation_screen_start = time.perf_counter()
    eval_screens, eval_screen_indices, eval_visual, eval_screen_info = encode_screens(
        evaluation_rows, corpus_root, screen_cfg, process
    )
    evaluation_screen_seconds = time.perf_counter() - evaluation_screen_start
    eval_feature_path = output_root / "evaluation-screen-features.npy"
    np.save(eval_feature_path, eval_visual, allow_pickle=False)
    eval_arrays = arrays_for_rows(evaluation_rows, eval_screen_indices, color_by_source)
    matched_probabilities = infer(
        selected,
        eval_arrays,
        audio_bank,
        eval_visual,
        mean,
        std,
        temperature=selected_temperature,
    )
    matched = metrics(matched_probabilities, eval_arrays["targets"], eval_arrays["colors"])
    missing_audio_probabilities = infer(
        selected,
        eval_arrays,
        audio_bank,
        eval_visual,
        mean,
        std,
        temperature=selected_temperature,
        missing_audio=True,
    )
    missing_visual_probabilities = infer(
        selected,
        eval_arrays,
        audio_bank,
        eval_visual,
        mean,
        std,
        temperature=selected_temperature,
        missing_visual=True,
    )
    audio_donor = eval_arrays["audio_indices"].copy()
    source_colors: dict[str, str] = {}
    for row in evaluation_rows:
        source_colors[row["source_id"]] = color_by_source[row["source_id"]]
    donor_by_source: dict[str, str] = {}
    sources_by_color = {
        color: sorted(
            {
                row["source_id"]
                for row in evaluation_rows
                if source_colors[row["source_id"]] == color
            }
        )
        for color in COLORS
    }
    for color in COLORS:
        other_colors = [candidate for candidate in COLORS if candidate != color]
        candidates = sorted(source for other in other_colors for source in sources_by_color[other])
        for source in sources_by_color[color]:
            donor_by_source[source] = candidates[
                int(canonical_hash([17, source, "wrong-color-donor"])[:8], 16) % len(candidates)
            ]
    source_to_feature = {row["source_id"]: row["audio_feature_index"] for row in evaluation_rows}
    audio_donor = np.asarray(
        [source_to_feature[donor_by_source[source]] for source in eval_arrays["source_ids"]],
        dtype=np.int64,
    )
    screen_donor = eval_arrays["screen_indices"].copy()
    # Generate a deterministic wrong-layout donor by source and original layout ID.
    screen_indices_for_key = {
        row["screenshot_sha256"]: index for index, row in enumerate(eval_screens)
    }
    wrong_screen_by_key: dict[str, int] = {}
    for row in eval_screens:
        options = [
            candidate for candidate in eval_screens if candidate["layout_id"] != row["layout_id"]
        ]
        if not options:
            raise ValueError("evaluation screens need at least two distinct layouts")
        selected_screen = options[
            int(canonical_hash([17, row["screenshot_sha256"], "wrong-screen"])[:8], 16)
            % len(options)
        ]
        wrong_screen_by_key[row["screenshot_sha256"]] = screen_indices_for_key[
            selected_screen["screenshot_sha256"]
        ]
    screen_donor = np.asarray(
        [wrong_screen_by_key[row["screenshot_sha256"]] for row in evaluation_rows],
        dtype=np.int64,
    )
    audio_mismatch_probabilities = infer(
        selected,
        eval_arrays,
        audio_bank,
        eval_visual,
        mean,
        std,
        temperature=selected_temperature,
        audio_donor=audio_donor,
    )
    screen_mismatch_probabilities = infer(
        selected,
        eval_arrays,
        audio_bank,
        eval_visual,
        mean,
        std,
        temperature=selected_temperature,
        screen_donor=screen_donor,
    )
    controls = {
        "matched": matched,
        "audio_mean_replaced": metrics(
            missing_audio_probabilities, eval_arrays["targets"], eval_arrays["colors"]
        ),
        "visual_content_mean_replaced_coordinates_retained": metrics(
            missing_visual_probabilities, eval_arrays["targets"], eval_arrays["colors"]
        ),
        "wrong_color_audio_donor": metrics(
            audio_mismatch_probabilities, eval_arrays["targets"], eval_arrays["colors"]
        ),
        "wrong_layout_screen_donor": metrics(
            screen_mismatch_probabilities, eval_arrays["targets"], eval_arrays["colors"]
        ),
    }
    prediction_rows = []
    for row, probabilities, color in zip(
        evaluation_rows, matched_probabilities, eval_arrays["colors"], strict=True
    ):
        option_ids = [option["option_id"] for option in row["options"]]
        target_id = option_ids[row["target_index"]]
        predicted_index = int(probabilities.argmax())
        prediction_rows.append(
            {
                "sample_id": row["sample_id"],
                "source_id": row["source_id"],
                "speaker_group": row["speaker_group"],
                "layout_id": row["layout_id"],
                "color_group_for_metrics_only": str(color),
                "option_ids": option_ids,
                "target_option_id": target_id,
                "predicted_option_id": option_ids[predicted_index],
                "probabilities_by_option_id": {
                    option_id: float(probability)
                    for option_id, probability in zip(option_ids, probabilities, strict=True)
                },
            }
        )
    prediction_sha = write_jsonl(output_root / "evaluation-predictions.jsonl", prediction_rows)

    row0 = evaluation_rows[0]
    first_indices = np.asarray([0], dtype=np.int64)
    first_features, _ = tensor_batch(first_indices, eval_arrays, audio_bank, eval_visual, mean, std)
    first_permutation = torch.tensor([2, 0, 3, 1], dtype=torch.long)
    permuted, inverse = permute_candidate_axis(first_features, first_permutation)
    with torch.inference_mode():
        probability_original = torch.softmax(
            selected(first_features)[0] / selected_temperature, dim=-1
        )
        probability_permuted = torch.softmax(
            selected(permuted)[0] / selected_temperature, dim=-1
        ).index_select(0, inverse)
    option_order_max_delta = float((probability_original - probability_permuted).abs().max())
    if option_order_max_delta > 1e-6:
        raise RuntimeError("candidate scorer failed supplied-option permutation invariance")
    decision_json = selected.decide(
        first_features[0], temperature=selected_temperature
    ).to_decision_json(
        request_id=row0["sample_id"],
        decision_type="choice",
        option_ids=[option["option_id"] for option in row0["options"]],
        option_texts=[
            f"action at x={option['x_norm']:.2f}, y={option['y_norm']:.2f}"
            for option in row0["options"]
        ],
    )
    write_json(output_root / "example-decision.json", json.loads(decision_json))
    elapsed = time.perf_counter() - fit_start
    peak_rss = max(peak_rss, sample_peak_rss(process))
    result = {
        "status": "PASS_EXECUTION_WITH_LIMITED_DEVELOPMENT_EVALUATION",
        "experiment_id": cfg["experiment_id"],
        "source_commit": os.environ.get("GIT_COMMIT", "unknown"),
        "config_sha256": effective["config_sha256"],
        "corpus_manifest_sha256": manifest_sha,
        "frozen_candidate_sha256": freeze_sha,
        "start_utc": started.isoformat(),
        "end_utc": datetime.now(UTC).isoformat(),
        "execution": {
            "device": "cpu",
            "cuda_used": False,
            "cloud_cost_usd": 0,
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cpu_threads": torch.get_num_threads(),
            "screen_feature_extract_seconds_dev": screen_encoder_seconds,
            "screen_feature_extract_seconds_evaluation": evaluation_screen_seconds,
            "fit_plus_evaluation_seconds": elapsed,
            "peak_sampled_rss_bytes": peak_rss,
            "optimizer_updates": sum(math.ceil(len(train_rows) / batch_size) for _ in history),
            "epochs_completed": len(history),
            "best_epoch": best_epoch,
            "early_stopping_reason": (
                "selection patience reached"
                if len(history) < max_epochs
                else "fixed epoch budget reached"
            ),
        },
        "dataset": {
            "split_counts": {
                "train": len(train_rows),
                "selection": len(selection_rows),
                "calibration": len(calibration_rows),
                "evaluation": len(evaluation_rows),
            },
            "speaker_groups": corpus["speaker_group_split"]["group_counts"],
            "source_rows": corpus["speaker_group_split"]["source_row_counts"],
            "audio_source_overlap": False,
            "normalized_content_overlap": False,
            "screenshot_hash_overlap": False,
            "all_modalities_required_by_factorial_contract": True,
            "evaluation_is_development_only": True,
        },
        "parameters": {
            "whisper_encoder_frozen": cfg["source"]["audio_encoder_parameters"],
            "dinov2_small_frozen": cfg["source"]["screen_encoder_parameters"],
            "trainable_readout": parameter_count,
            "resident_total_prequantization": (
                cfg["source"]["audio_encoder_parameters"]
                + cfg["source"]["screen_encoder_parameters"]
                + parameter_count
            ),
            "updated_encoder_parameters": 0,
        },
        "selected_checkpoint": {
            "epoch": best_epoch,
            "sha256": best_sha,
            "calibration_temperature": selected_temperature,
            "calibration_file_sha256": calibration_sha,
            "final_checkpoint_sha256": sha256_file(output_root / "checkpoint-final.safetensors"),
        },
        "evaluation": controls,
        "evaluation_prediction_sha256": prediction_sha,
        "candidate_order_permutation_max_probability_delta": option_order_max_delta,
        "example_typed_choice_json": json.loads(decision_json),
        "overfitting_summary": {
            "last_train_macro_color_nll": history[-1]["train_fixed_macro_color_nll"],
            "last_selection_macro_color_nll": history[-1]["selection"]["macro_color_nll"],
            "best_selection_macro_color_nll": best_score,
        },
        "limitations": [
            "Constructed four-color voice-guided screen-action task, not a general Omni benchmark.",
            "Four evaluation speaker groups from source TRAIN; development evidence only.",
            "Generated still screens; natural video temporal reasoning is not tested.",
            "No text-only, general audio, natural-image, or video quality is inferred.",
            "Full four-modality <=200M inventory and 95-98% task gates remain open.",
        ],
    }
    write_json(output_root / "result.json", result)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus-root",
        type=Path,
        default=Path(
            r"C:\CodexArtifacts\pretrained-reuse-r2\spoken-palette-av-decision-v1-attempt-02"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            r"C:\CodexArtifacts\pretrained-reuse-r2\spoken-palette-av-decision-v1-fit-attempt-01"
        ),
    )
    print(
        json.dumps(
            json_safe(fit(parser.parse_args())), ensure_ascii=False, sort_keys=True, indent=2
        )
    )
