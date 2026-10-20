"""Train matched small AV timing readouts over frozen VideoPrism/MiniLM features."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from scripts.hotgpu_preflight import validate_gpu_preflight
from tiny_omni_decision.temporal_decision import CausalGatedTemporalDecision, probability_json

CONFIG_DEFAULT = Path("configs/pretrained_reuse/clevrer_synthetic_av_temporal_decision_v1.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_DEFAULT)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--feature-report", type=Path, required=True)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--smoke-only", action="store_true")
    return parser.parse_args()


def _read_rows(path: Path, expected_split: str) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows or any(row.get("split") != expected_split for row in rows):
        raise ValueError(f"{path} must contain nonempty {expected_split} rows")
    return rows


def _ordered_ids_sha256(rows: list[dict[str, Any]]) -> str:
    return hashlib.sha256("\n".join(str(row["id"]) for row in rows).encode()).hexdigest()


def _validate_pairs(rows: list[dict[str, Any]]) -> dict[str, list[int]]:
    indexes: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        indexes[str(row["pair_id"])].append(index)
    for pair_id, members in indexes.items():
        if len(members) != 2:
            raise ValueError(f"expected exactly two variants for {pair_id}")
        left, right = (rows[index] for index in members)
        if (
            {left["target_index"], right["target_index"]} != {0, 1}
            or left["question"] != right["question"]
            or left["scene_group_id"] != right["scene_group_id"]
            or left["media_sha256"] != right["media_sha256"]
        ):
            raise ValueError(f"inconsistent paired variants: {pair_id}")
    return dict(indexes)


def _load_inputs(
    config: dict[str, Any], feature_dir: Path, feature_report_path: Path, sample_dir: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, torch.Tensor], dict[str, Any]]:
    report = json.loads(feature_report_path.read_text(encoding="utf-8"))
    if (
        report.get("status") != "complete_clevrer_synthetic_av_timing_feature_cache"
        or report.get("validation_test_or_sealed_data_opened") is not False
        or report.get("scene_overlap") != 0
        or report.get("media_hash_overlap") != 0
    ):
        raise ValueError("feature report is incomplete or violates split isolation")
    feature_path = feature_dir / "decision-features.npz"
    if _sha256(feature_path) != report.get("feature_cache_sha256"):
        raise ValueError("feature cache hash differs from its report")
    train_rows = _read_rows(sample_dir / "train.jsonl", "train")
    validation_rows = _read_rows(sample_dir / "validation.jsonl", "validation")
    manifest_hashes = {
        name: _sha256(sample_dir / name)
        for name in ("train.jsonl", "validation.jsonl", "fetch-report.json")
    }
    if manifest_hashes != report.get("sample_manifest_sha256"):
        raise ValueError("sample manifests differ from the feature builder receipt")
    if _ordered_ids_sha256(train_rows) != report.get(
        "train_sample_id_order_sha256"
    ) or _ordered_ids_sha256(validation_rows) != report.get("validation_sample_id_order_sha256"):
        raise ValueError("sample ID order differs from the frozen feature report")
    train_pairs = _validate_pairs(train_rows)
    validation_pairs = _validate_pairs(validation_rows)
    train_scenes = {str(row["scene_group_id"]) for row in train_rows}
    validation_scenes = {str(row["scene_group_id"]) for row in validation_rows}
    train_media = {str(row["media_sha256"]) for row in train_rows}
    validation_media = {str(row["media_sha256"]) for row in validation_rows}
    if train_scenes & validation_scenes or train_media & validation_media:
        raise ValueError("train and validation scenes/media overlap")

    with np.load(feature_path, allow_pickle=False) as archive:
        sample_ids = archive["sample_ids"].astype(str).tolist()
        splits = archive["splits"].astype(str).tolist()
        scene_ids = archive["scene_ids"].astype(str).tolist()
        pair_ids = archive["pair_ids"].astype(str).tolist()
        media_hashes = archive["media_sha256"].astype(str).tolist()
        arrays = {
            "video": archive["videos"].astype(np.float32, copy=False),
            "audio": archive["audios"].astype(np.float32, copy=False),
            "text": archive["texts"].astype(np.float32, copy=False),
            "targets": archive["targets"].astype(np.int64, copy=False),
        }
    rows = train_rows + validation_rows
    if (
        sample_ids != [str(row["id"]) for row in rows]
        or splits != [str(row["split"]) for row in rows]
        or scene_ids != [str(row["scene_group_id"]) for row in rows]
        or pair_ids != [str(row["pair_id"]) for row in rows]
        or media_hashes != [str(row["media_sha256"]) for row in rows]
        or arrays["targets"].tolist() != [int(row["target_index"]) for row in rows]
    ):
        raise ValueError("feature rows do not match the frozen sample manifests")
    if (
        arrays["video"].shape != (len(rows), 16, 768)
        or arrays["audio"].shape != (len(rows), 16, 1)
        or arrays["text"].shape != (len(rows), 384)
        or arrays["targets"].shape != (len(rows),)
        or not all(np.isfinite(array).all() for array in arrays.values())
    ):
        raise ValueError("feature tensors have unexpected shapes or nonfinite values")
    text_model = config["frozen_encoders"]["text"]
    video_model = config["frozen_encoders"]["video"]
    if (
        report["video_features"]["revision"] != video_model["revision"]
        or report["text_features"]["revision"] != text_model["revision"]
        or report["text_features"]["weight_sha256"] != text_model["checkpoint_sha256"]
        or report["video_features"]["checkpoint_sha256"] != video_model["checkpoint_sha256"]
    ):
        raise ValueError("feature encoder provenance differs from the frozen experiment config")
    if (
        len(train_pairs) != report["train_sample_count"] // 2
        or len(validation_pairs) != report["validation_sample_count"] // 2
    ):
        raise ValueError("paired event count differs from the frozen feature report")
    tensors = {name: torch.from_numpy(value.copy()) for name, value in arrays.items()}
    boundaries = {
        "train_count": len(train_rows),
        "validation_count": len(validation_rows),
        "train_id_order_sha256": _ordered_ids_sha256(train_rows),
        "validation_id_order_sha256": _ordered_ids_sha256(validation_rows),
        "train_scenes": sorted(train_scenes),
        "validation_scenes": sorted(validation_scenes),
        "train_pair_count": len(train_pairs),
        "validation_pair_count": len(validation_pairs),
        "sample_manifest_sha256": manifest_hashes,
        "feature_builder_sha256": report["feature_builder_sha256"],
        "split_overlap_scenes": 0,
        "split_overlap_media_hashes": 0,
    }
    return train_rows, validation_rows, tensors, {"feature_report": report, "split": boundaries}


def _gpu_gate(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        return {"mode": "cpu"}
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,utilization.gpu,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    used, utilization, total = (
        int(value.strip()) for value in result.stdout.splitlines()[0].split(",")
    )
    pids_output = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    active_pids = {
        int(line.strip())
        for line in pids_output.splitlines()
        if line.strip() and line.strip().casefold() != "no running processes found"
    }
    return validate_gpu_preflight(
        memory_used_mib=used,
        utilization_pct=utilization,
        total_mib=total,
        active_pids=active_pids,
        own_pid=os.getpid(),
        environment=os.environ,
    )


def _metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float | int]:
    with torch.inference_mode():
        probabilities = torch.softmax(logits.float(), dim=-1)
        prediction = probabilities.argmax(dim=-1)
        confidence = probabilities.max(dim=-1).values
        correct = prediction.eq(targets)
        one_hot = F.one_hot(targets, num_classes=2).float()
        nll = F.cross_entropy(logits.float(), targets, reduction="none")
        brier = (probabilities - one_hot).square().sum(dim=-1)
        ece = torch.zeros((), dtype=torch.float32, device=logits.device)
        for index in range(15):
            mask = (confidence >= index / 15) & (
                confidence < (index + 1) / 15 if index < 14 else confidence <= 1
            )
            if bool(mask.any()):
                ece += (
                    mask.float().mean()
                    * (correct[mask].float().mean() - confidence[mask].mean()).abs()
                )
        return {
            "count": int(len(targets)),
            "accuracy": float(correct.float().mean().item()),
            "nll": float(nll.mean().item()),
            "brier": float(brier.mean().item()),
            "ece_15_bins": float(ece.item()),
            "mean_confidence": float(confidence.mean().item()),
        }


def _model_logits(
    model: CausalGatedTemporalDecision,
    video: torch.Tensor,
    audio: torch.Tensor,
    text: torch.Tensor,
    *,
    mean_pooled: bool,
) -> torch.Tensor:
    if mean_pooled:
        video = video.mean(dim=1, keepdim=True).expand_as(video)
        audio = audio.mean(dim=1, keepdim=True).expand_as(audio)
    return model(video, audio, text).logits[:, -1, :]


def _scene_pair_metrics(
    rows: list[dict[str, Any]], logits: torch.Tensor, targets: torch.Tensor
) -> dict[str, Any]:
    pair_indexes = _validate_pairs(rows)
    predictions = logits.argmax(dim=-1)
    pairs = []
    for pair_id, indexes in pair_indexes.items():
        is_correct = [int(predictions[index].item() == targets[index].item()) for index in indexes]
        predicted = [int(predictions[index].item()) for index in indexes]
        pairs.append(
            {
                "pair_id": pair_id,
                "scene_group_id": rows[indexes[0]]["scene_group_id"],
                "both_correct": int(all(is_correct)),
                "mean_row_accuracy": float(np.mean(is_correct)),
                "prediction_flip": int(predicted[0] != predicted[1]),
            }
        )
    return {
        "event_pair_count": len(pairs),
        "both_variants_correct_rate": float(np.mean([row["both_correct"] for row in pairs])),
        "mean_row_accuracy_per_pair": float(np.mean([row["mean_row_accuracy"] for row in pairs])),
        "prediction_flip_rate": float(np.mean([row["prediction_flip"] for row in pairs])),
    }


def _pair_donor_indices(rows: list[dict[str, Any]]) -> torch.Tensor:
    pairs = _validate_pairs(rows)
    donors = torch.arange(len(rows), dtype=torch.long)
    for indexes in pairs.values():
        donors[indexes[0]] = indexes[1]
        donors[indexes[1]] = indexes[0]
    return donors


def _prediction_rows(
    rows: list[dict[str, Any]], logits: torch.Tensor, targets: torch.Tensor
) -> list[dict[str, Any]]:
    values = logits.detach().cpu()
    output = []
    for row, target, logit in zip(rows, targets.detach().cpu(), values, strict=True):
        result = probability_json(
            logit,
            row["options"],
            str(row["id"]),
        )
        output.append(
            {
                **result,
                "target": row["target"],
                "target_index": int(target.item()),
                "scene_group_id": row["scene_group_id"],
                "pair_id": row["pair_id"],
            }
        )
    return output


def _paired_scene_bootstrap(
    rows: list[dict[str, Any]],
    left_logits: torch.Tensor,
    right_logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    scenes = sorted({str(row["scene_group_id"]) for row in rows})
    scene_to_indexes: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        scene_to_indexes[str(row["scene_group_id"])].append(index)
    targets_cpu = targets.detach().cpu()
    left = left_logits.detach().cpu().float()
    right = right_logits.detach().cpu().float()
    left_prob = torch.softmax(left, dim=-1)
    right_prob = torch.softmax(right, dim=-1)
    left_correct = (left.argmax(-1) == targets_cpu).float().numpy()
    right_correct = (right.argmax(-1) == targets_cpu).float().numpy()
    left_nll = F.cross_entropy(left, targets_cpu, reduction="none").numpy()
    right_nll = F.cross_entropy(right, targets_cpu, reduction="none").numpy()
    one_hot = F.one_hot(targets_cpu, num_classes=2).float()
    left_brier = (left_prob - one_hot).square().sum(-1).numpy()
    right_brier = (right_prob - one_hot).square().sum(-1).numpy()
    outcomes = {
        "accuracy": left_correct - right_correct,
        "nll": right_nll - left_nll,
        "brier": right_brier - left_brier,
    }
    scene_deltas: dict[str, dict[str, float]] = {scene: {} for scene in scenes}
    for name, values in outcomes.items():
        for scene in scenes:
            scene_deltas[scene][name] = float(np.mean([values[i] for i in scene_to_indexes[scene]]))
    rng = np.random.default_rng(seed)
    sample = rng.integers(0, len(scenes), size=(repetitions, len(scenes)))
    result = {}
    for name, _values in outcomes.items():
        per_scene = np.asarray([scene_deltas[scene][name] for scene in scenes])
        bootstrap = per_scene[sample].mean(axis=1)
        result[name] = {
            "difference_direction": (
                "joint minus audio-only" if name == "accuracy" else "audio-only minus joint"
            ),
            "point_difference": float(per_scene.mean()),
            "percentile_95_ci": [
                float(np.quantile(bootstrap, 0.025, method="linear")),
                float(np.quantile(bootstrap, 0.975, method="linear")),
            ],
        }
    return {
        "method": "paired bootstrap over validation scenes; per-scene mean example difference",
        "cluster_count": len(scenes),
        "repetitions": repetitions,
        "seed": seed,
        "comparison": ("joint_lstm versus audio_only_lstm; positive differences favor joint"),
        "metrics": result,
    }


def _run_candidate(
    *,
    name: str,
    mode: str,
    mean_pooled: bool,
    config: dict[str, Any],
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    tensors: dict[str, torch.Tensor],
    train_count: int,
    device: torch.device,
    output_dir: Path,
    epoch_orders: list[torch.Tensor],
) -> dict[str, Any]:
    settings = config["training"]
    readout = config["readout"]
    torch.manual_seed(int(config["seed"]))
    random.seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(config["seed"]))
        torch.cuda.reset_peak_memory_stats(device)
    model = CausalGatedTemporalDecision(
        video_size=int(readout["video_size"]),
        audio_size=int(readout["audio_size"]),
        text_size=int(readout["text_size"]),
        hidden_size=int(readout["hidden_size"]),
        modality_mode=mode,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    values = {key: tensor.to(device) for key, tensor in tensors.items()}
    history_path = output_dir / f"training-history-{name}.jsonl"
    best_score = math.inf
    best_step = 0
    best_state: dict[str, torch.Tensor] | None = None
    training_seconds = 0.0
    evaluation_seconds = 0.0
    losses: list[float] = []
    grad_norms: list[float] = []
    consumed_examples = 0
    seen_train_indexes: set[int] = set()
    total_updates = int(settings["max_optimizer_updates_per_candidate"])
    evaluation_interval = int(settings["evaluation_interval"])
    batches_per_epoch = math.ceil(train_count / int(settings["batch_size"]))

    def evaluate(step: int) -> dict[str, Any]:
        nonlocal best_score, best_step, best_state, evaluation_seconds
        started = time.perf_counter()
        model.eval()
        with torch.inference_mode():
            train_logits = _model_logits(
                model,
                values["video"][:train_count],
                values["audio"][:train_count],
                values["text"][:train_count],
                mean_pooled=mean_pooled,
            ).float()
            validation_logits = _model_logits(
                model,
                values["video"][train_count:],
                values["audio"][train_count:],
                values["text"][train_count:],
                mean_pooled=mean_pooled,
            ).float()
            train_metrics = _metrics(train_logits, values["targets"][:train_count])
            validation_metrics = _metrics(validation_logits, values["targets"][train_count:])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        evaluation_seconds += time.perf_counter() - started
        score = float(validation_metrics["nll"])
        is_best = score < best_score
        if is_best:
            best_score = score
            best_step = step
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            torch.save(
                {
                    "candidate": name,
                    "step": step,
                    "selector": "minimum validation NLL",
                    "validation_nll": score,
                    "model_state": best_state,
                    "config": config,
                },
                output_dir / f"best-{name}.pt",
            )
        record = {
            "candidate": name,
            "optimizer_updates": step,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "training_cross_entropy_recent_mean": (
                float(np.mean(losses[-evaluation_interval:])) if losses else None
            ),
            "training_gradient_norm_recent_mean": (
                float(np.mean(grad_norms[-evaluation_interval:])) if grad_norms else None
            ),
            "cumulative_consumed_examples": consumed_examples,
            "unique_train_examples_consumed": len(seen_train_indexes),
            "unique_train_scenes_consumed": len(
                {str(train_rows[index]["scene_group_id"]) for index in seen_train_indexes}
            ),
            "unique_train_media_assets_consumed": len(
                {str(train_rows[index]["media_sha256"]) for index in seen_train_indexes}
            ),
            "train_full_set": train_metrics,
            "validation": validation_metrics,
            "best_step": best_step,
            "best_validation_nll": best_score,
            "selected_now": is_best,
            "elapsed_training_seconds": training_seconds,
            "elapsed_evaluation_seconds": evaluation_seconds,
            "peak_vram_bytes": (
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            ),
        }
        with history_path.open("a", encoding="utf-8", newline="\n") as history:
            history.write(json.dumps(record, sort_keys=True) + "\n")
        return {
            "record": record,
            "train_logits": train_logits,
            "validation_logits": validation_logits,
        }

    start_time = datetime.now(UTC).isoformat()
    latest: dict[str, Any] | None = None
    latest = evaluate(0)
    for update in range(total_updates):
        model.train()
        order = epoch_orders[update // batches_per_epoch]
        batch_index = update % batches_per_epoch
        batch_indexes = order[
            batch_index * int(settings["batch_size"]) : (batch_index + 1)
            * int(settings["batch_size"])
        ]
        if len(batch_indexes) == 0:
            raise RuntimeError("empty deterministic training batch")
        seen_train_indexes.update(int(index) for index in batch_indexes.tolist())
        consumed_examples += len(batch_indexes)
        batch_indexes = batch_indexes.to(device)
        batch_video = values["video"].index_select(0, batch_indexes)
        batch_audio = values["audio"].index_select(0, batch_indexes)
        batch_text = values["text"].index_select(0, batch_indexes)
        batch_targets = values["targets"].index_select(0, batch_indexes)
        optimizer.zero_grad(set_to_none=True)
        update_started = time.perf_counter()
        logits = _model_logits(model, batch_video, batch_audio, batch_text, mean_pooled=mean_pooled)
        loss = F.cross_entropy(logits.float(), batch_targets)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"nonfinite loss in {name} update {update + 1}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(settings["gradient_clip_norm"])
        )
        if not bool(torch.isfinite(grad_norm)) or any(
            parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        ):
            raise FloatingPointError(f"nonfinite gradient in {name} update {update + 1}")
        optimizer.step()
        losses.append(float(loss.detach().item()))
        grad_norms.append(float(grad_norm.detach().item()))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        training_seconds += time.perf_counter() - update_started
        if (update + 1) % evaluation_interval == 0 or update + 1 == total_updates:
            latest = evaluate(update + 1)
    if latest is None or best_state is None:
        raise RuntimeError(f"candidate {name} did not produce an evaluation")

    torch.save(
        {
            "candidate": name,
            "step": total_updates,
            "best_step": best_step,
            "model_state": model.state_dict(),
            "config": config,
        },
        output_dir / f"final-{name}.pt",
    )
    final_train_logits = latest["train_logits"]
    final_validation_logits = latest["validation_logits"]
    best_model = CausalGatedTemporalDecision(
        video_size=int(readout["video_size"]),
        audio_size=int(readout["audio_size"]),
        text_size=int(readout["text_size"]),
        hidden_size=int(readout["hidden_size"]),
        modality_mode=mode,
    ).to(device)
    checkpoint = torch.load(output_dir / f"best-{name}.pt", map_location="cpu", weights_only=False)
    best_model.load_state_dict(checkpoint["model_state"], strict=True)
    best_model.eval()
    with torch.inference_mode():
        best_video = values["video"][train_count:]
        best_audio = values["audio"][train_count:]
        if mean_pooled:
            best_video = best_video.mean(dim=1, keepdim=True).expand_as(best_video)
            best_audio = best_audio.mean(dim=1, keepdim=True).expand_as(best_audio)
        best_output = best_model(
            best_video,
            best_audio,
            values["text"][train_count:],
        )
        best_validation_logits = best_output.logits[:, -1, :].float()
    best_metrics = _metrics(best_validation_logits, values["targets"][train_count:])
    if abs(float(best_metrics["nll"]) - best_score) > 1e-6:
        raise RuntimeError(f"best checkpoint reload changed selector metrics for {name}")
    best_predictions = _prediction_rows(
        validation_rows, best_validation_logits, values["targets"][train_count:]
    )
    final_predictions = _prediction_rows(
        validation_rows, final_validation_logits, values["targets"][train_count:]
    )
    (output_dir / f"predictions-best-{name}.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in best_predictions),
        encoding="utf-8",
    )
    (output_dir / f"predictions-final-{name}.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in final_predictions),
        encoding="utf-8",
    )
    result = {
        "candidate": name,
        "mode": mode,
        "mean_pooled_control": mean_pooled,
        "trainable_parameter_count": parameter_count,
        "optimizer_updates": total_updates,
        "consumed_training_examples": consumed_examples,
        "unique_training_examples_consumed": len(seen_train_indexes),
        "unique_training_scenes_consumed": len(
            {str(train_rows[index]["scene_group_id"]) for index in seen_train_indexes}
        ),
        "unique_training_media_assets_consumed": len(
            {str(train_rows[index]["media_sha256"]) for index in seen_train_indexes}
        ),
        "best_step": best_step,
        "best_validation": best_metrics,
        "best_validation_pairs": _scene_pair_metrics(
            validation_rows, best_validation_logits, values["targets"][train_count:]
        ),
        "best_validation_modality_gate_mean": {
            "video": float(best_output.modality_weights[..., 0].mean().item()),
            "audio": float(best_output.modality_weights[..., 1].mean().item()),
        },
        "final_train": _metrics(final_train_logits, values["targets"][:train_count]),
        "final_validation": _metrics(final_validation_logits, values["targets"][train_count:]),
        "training_seconds": training_seconds,
        "evaluation_seconds": evaluation_seconds,
        "peak_vram_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
        "start_time_utc": start_time,
        "end_time_utc": datetime.now(UTC).isoformat(),
        "best_checkpoint_sha256": _sha256(output_dir / f"best-{name}.pt"),
        "final_checkpoint_sha256": _sha256(output_dir / f"final-{name}.pt"),
    }
    if name == "joint_lstm":
        donor_indexes = _pair_donor_indices(validation_rows).to(device)
        with torch.inference_mode():
            swapped_output = best_model(
                values["video"][train_count:],
                values["audio"][train_count:].index_select(0, donor_indexes),
                values["text"][train_count:],
            )
            swapped_logits = swapped_output.logits[:, -1, :].float()
        normal_prediction = best_validation_logits.argmax(dim=-1)
        swapped_prediction = swapped_logits.argmax(dim=-1)
        donor_targets = values["targets"][train_count:].index_select(0, donor_indexes)
        result["paired_audio_swap_intervention"] = {
            "prediction_flip_rate": float(
                swapped_prediction.ne(normal_prediction).float().mean().item()
            ),
            "donor_target_accuracy": float(
                swapped_prediction.eq(donor_targets).float().mean().item()
            ),
            "original_target_accuracy_after_swap": float(
                swapped_prediction.eq(values["targets"][train_count:]).float().mean().item()
            ),
            "interpretation": (
                "audio is swapped only within each paired same-question/same-video event; "
                "donor target is the opposite label"
            ),
        }
        swapped_predictions = _prediction_rows(
            validation_rows, swapped_logits, values["targets"][train_count:]
        )
        for row, donor_index in zip(swapped_predictions, donor_indexes.cpu().tolist(), strict=True):
            row["counterfactual_audio_donor_id"] = validation_rows[donor_index]["id"]
            row["counterfactual_audio_donor_target"] = validation_rows[donor_index]["target"]
        (output_dir / "predictions-best-joint_lstm-audio-paired-swap.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in swapped_predictions),
            encoding="utf-8",
        )
    return result


def main() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    args = _args()
    config_path = args.config.resolve(strict=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (
        config.get("protocol_sha256")
        != "82b76e813455061769883b6a46517dbfb3a7a54f7593c1093ab92d8cc261fab7"
    ):
        raise ValueError("experiment config does not match the frozen AV timing protocol")
    train_rows, validation_rows, tensors, input_report = _load_inputs(
        config,
        args.feature_dir.resolve(strict=True),
        args.feature_report.resolve(strict=True),
        args.sample_dir.resolve(strict=True),
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; training will not start")
    gate = _gpu_gate(device)
    if args.smoke_only:
        model = CausalGatedTemporalDecision(
            video_size=int(config["readout"]["video_size"]),
            audio_size=int(config["readout"]["audio_size"]),
            text_size=int(config["readout"]["text_size"]),
            hidden_size=int(config["readout"]["hidden_size"]),
        ).to(device)
        with torch.inference_mode():
            output = model(
                tensors["video"][:2].to(device),
                tensors["audio"][:2].to(device),
                tensors["text"][:2].to(device),
            )
        print(
            json.dumps(
                {
                    "status": "smoke_pass",
                    "device": str(device),
                    "gpu_gate": gate,
                    "trainable_parameter_count": model.trainable_parameter_count,
                    "output_shape": list(output.logits.shape),
                    "sample_count": len(train_rows) + len(validation_rows),
                    "scene_overlap": 0,
                    "sealed_audit_opened": False,
                },
                indent=2,
            )
        )
        return

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite training attempt: {output_dir}")
    if args.device != "cuda":
        raise ValueError("the registered matched training run requires HotUGPU CUDA")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.synchronize(device)
    output_dir.mkdir(parents=True, exist_ok=False)
    effective = {
        **config,
        "resolved": {
            "config_path": str(config_path),
            "config_sha256": _sha256(config_path),
            "feature_dir": str(args.feature_dir.resolve()),
            "feature_cache_sha256": input_report["feature_report"]["feature_cache_sha256"],
            "feature_report_sha256": _sha256(args.feature_report.resolve(strict=True)),
            "sample_dir": str(args.sample_dir.resolve(strict=True)),
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "repository_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                check=True,
                text=True,
            ).stdout.strip(),
            "training_script_sha256": _sha256(Path(__file__).resolve()),
            "readout_implementation_sha256": _sha256(
                Path(__file__).resolve().parents[1]
                / "src"
                / "tiny_omni_decision"
                / "temporal_decision.py"
            ),
            "feature_builder_sha256": input_report["split"]["feature_builder_sha256"],
            "command": [sys.executable, *sys.argv],
            "hotgpu_reservation": gate,
        },
    }
    effective_path = output_dir / "effective-config.json"
    effective_path.write_text(
        json.dumps(effective, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    metadata = {
        "status": "running",
        "experiment_id": config["experiment_id"],
        "effective_config_sha256": _sha256(effective_path),
        "input_integrity": input_report["split"],
        "sealed_audit_opened": False,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "ended_at_utc": None,
        "completion_reason": None,
    }
    metadata_path = output_dir / "run-metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    train_count = len(train_rows)
    device_tensors = {key: value.to(device) for key, value in tensors.items()}
    batch_size = int(config["training"]["batch_size"])
    batches_per_epoch = math.ceil(train_count / batch_size)
    rng = torch.Generator(device="cpu").manual_seed(int(config["seed"]))
    epoch_orders = [
        torch.randperm(train_count, generator=rng)
        for _ in range(
            math.ceil(config["training"]["max_optimizer_updates_per_candidate"] / batches_per_epoch)
        )
    ]
    sampler_indexes = []
    for update in range(int(config["training"]["max_optimizer_updates_per_candidate"])):
        order = epoch_orders[update // batches_per_epoch]
        batch_index = update % batches_per_epoch
        sampler_indexes.extend(
            order[batch_index * batch_size : (batch_index + 1) * batch_size].tolist()
        )
    sampler_order_sha256 = hashlib.sha256(
        np.asarray(sampler_indexes, dtype="<i8").tobytes()
    ).hexdigest()
    effective["resolved"]["sampler_order_sha256"] = sampler_order_sha256
    effective["resolved"]["scheduled_examples_per_candidate"] = len(sampler_indexes)
    effective_path.write_text(
        json.dumps(effective, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    metadata["effective_config_sha256"] = _sha256(effective_path)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    run_results = []
    for name in config["training"]["candidate_order"]:
        if name == "joint_lstm":
            mode, mean_pooled = "joint", False
        elif name == "audio_only_lstm":
            mode, mean_pooled = "audio_only", False
        elif name == "video_only_lstm":
            mode, mean_pooled = "video_only", False
        elif name == "joint_mean_pooled":
            mode, mean_pooled = "joint", True
        else:
            raise ValueError(f"unknown matched candidate in config: {name}")
        result = _run_candidate(
            name=name,
            mode=mode,
            mean_pooled=mean_pooled,
            config=config,
            train_rows=train_rows,
            validation_rows=validation_rows,
            tensors=device_tensors,
            train_count=train_count,
            device=device,
            output_dir=output_dir,
            epoch_orders=epoch_orders,
        )
        run_results.append(result)
    best_validation_logits = {
        name: torch.tensor(
            [
                [item["probability"] for item in prediction["probabilities"]]
                for prediction in _read_prediction_jsonl(
                    output_dir / f"predictions-best-{name}.jsonl"
                )
            ],
            dtype=torch.float32,
        )
        .clamp_min(1e-8)
        .log()
        for name in ("joint_lstm", "audio_only_lstm")
    }
    bootstrap = _paired_scene_bootstrap(
        validation_rows,
        best_validation_logits["joint_lstm"],
        best_validation_logits["audio_only_lstm"],
        device_tensors["targets"][train_count:].cpu(),
        repetitions=int(config["validation"]["bootstrap_repetitions"]),
        seed=int(config["seed"]),
    )
    result = {
        "status": "complete_matched_clevrer_synthetic_av_temporal_decision_run",
        "experiment_id": config["experiment_id"],
        "source_commit": effective["resolved"]["repository_commit"],
        "effective_config_sha256": _sha256(effective_path),
        "input_integrity": input_report["split"],
        "candidate_results": run_results,
        "paired_scene_bootstrap_joint_vs_audio_only": bootstrap,
        "text_only_pair_control": {
            "accuracy": 0.5,
            "nll": math.log(2),
            "brier": 0.5,
            "ece_15_bins": 0.0,
            "reason": (
                "each paired variant has identical text and opposite labels; constant "
                "0.5 probabilities"
            ),
        },
        "all_encoders_frozen": True,
        "sealed_audit_opened": False,
        "compute_cost_usd": 0.0,
        "gpu": torch.cuda.get_device_name(device),
        "total_updates": sum(value["optimizer_updates"] for value in run_results),
        "started_at_utc": metadata["started_at_utc"],
        "ended_at_utc": datetime.now(UTC).isoformat(),
    }
    result_path = output_dir / "results.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metadata.update(
        {
            "status": "complete",
            "ended_at_utc": result["ended_at_utc"],
            "completion_reason": "all four matched candidates completed 2,048 optimizer updates",
            "results_sha256": _sha256(result_path),
        }
    )
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


def _read_prediction_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


if __name__ == "__main__":
    main()
