"""Fit text-only and ordered-video candidate scorers on a frozen CLEVRER mix."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer

from scripts.train_frozen_text_probe import CandidateScorer, _metrics
from scripts.train_frozen_video_probe import _text_features
from tiny_omni_decision.dataset import sha256_file
from tiny_omni_decision.temporal_candidate import (
    TemporalCandidateScorer,
    temporal_candidate_parameter_count,
)

QUESTION_TYPES = ("temporal_descriptive", "explanatory", "predictive", "counterfactual")
GPU_GATE_ENV = {
    "reserved_mib": "TINY_OMNI_GPU_GATE_RESERVED_MIB",
    "free_mib": "TINY_OMNI_GPU_GATE_FREE_MIB",
    "total_mib": "TINY_OMNI_GPU_GATE_TOTAL_MIB",
    "utilization_pct": "TINY_OMNI_GPU_GATE_UTILIZATION_PCT",
    "block_if_free_mib": "TINY_OMNI_GPU_GATE_BLOCK_IF_FREE_MIB",
    "min_child_free_mib": "TINY_OMNI_GPU_GATE_MIN_CHILD_FREE_MIB",
}
TEXT_REVISION = "4ca70771034acceecb2e72475f72050fcdde4ddc"
TEXT_PARAMETER_COUNT = 17_389_824
VIDEO_REVISION = "be719a406d563b66f0ac969e7c94bab8e997c81a"
VIDEO_PARAMETER_COUNT = 114_365_184
VIDEO_COMMIT = "c6eba0ece3dd49a24ffc44582966997a4a59f1d8"
VIDEO_SHA256 = "61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29"
TEXT_SHA256 = "cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433"


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _ordered_ids_hash(rows: list[dict[str, Any]]) -> str:
    return _sha_bytes("\n".join(str(row["id"]) for row in rows).encode())


def _counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {kind: sum(row["question_type"] == kind for row in rows) for kind in QUESTION_TYPES}


def _require_gpu_gate(config: dict[str, Any]) -> dict[str, int]:
    policy = config.get("gpu_gate", {})
    if policy.get("required") is not True:
        raise ValueError("GPU gate must remain required for this experiment")
    gate = {}
    for key, variable in GPU_GATE_ENV.items():
        value = os.environ.get(variable)
        if value is None:
            raise RuntimeError(f"GPU run must be launched through the HotUGPU gate ({variable})")
        try:
            gate[key] = int(value)
        except ValueError as exc:
            raise ValueError(f"invalid HotUGPU telemetry in {variable}") from exc
    if gate["reserved_mib"] < int(policy["min_reserved_mib"]):
        raise RuntimeError("HotUGPU VRAM reservation is below the configured minimum")
    if gate["free_mib"] >= int(policy["block_if_free_mib"]):
        raise RuntimeError("HotUGPU reservation did not close Orbitune's free-memory gate")
    if gate["free_mib"] < int(policy["min_child_free_mib"]):
        raise RuntimeError("HotUGPU reservation leaves too little VRAM for the child")
    if gate["utilization_pct"] >= int(policy["max_start_utilization_pct"]):
        raise RuntimeError("GPU utilization was too high at HotUGPU handoff")
    if (
        gate["block_if_free_mib"] != int(policy["block_if_free_mib"])
        or gate["min_child_free_mib"] != int(policy["min_child_free_mib"])
    ):
        raise RuntimeError("HotUGPU handoff thresholds differ from the frozen experiment policy")
    return gate


def _validate(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    source = config["source"]
    split_dir = Path(source["split_dir"]).resolve(strict=True)
    sample_dir = Path(source["sample_dir"]).resolve(strict=True)
    feature_path = Path(config["features"]["cache"]).resolve(strict=True)
    feature_report_path = Path(config["features"]["cache_report"]).resolve(strict=True)
    text_model = Path(config["text_encoder"]["model_path"]).resolve(strict=True)
    expected_split_report = json.loads((split_dir / "preflight.json").read_text(encoding="utf-8"))
    for name, expected in expected_split_report["counts"]["split_file_sha256"].items():
        if sha256_file(split_dir / name) != expected:
            raise ValueError(f"frozen source split hash mismatch: {name}")
    if source.get("sealed_audit_loaded") is not False:
        raise ValueError("sealed audit must not be loaded")
    if source["revision"] != "98b842082ba4f7c18b6b9e3f39145871782a65ef":
        raise ValueError("unexpected pinned CLEVRER revision")
    if (
        config["features"]["revision"] != VIDEO_REVISION
        or config["features"]["source_commit"] != VIDEO_COMMIT
    ):
        raise ValueError("unexpected VideoPrism source revision")
    if config["features"]["checkpoint_sha256"] != VIDEO_SHA256:
        raise ValueError("unexpected VideoPrism checkpoint hash")
    if (
        config["text_encoder"]["revision"] != TEXT_REVISION
        or config["text_encoder"]["weights_sha256"] != TEXT_SHA256
    ):
        raise ValueError("unexpected MiniLM revision or weight hash")
    if sha256_file(text_model / "model.safetensors") != TEXT_SHA256:
        raise ValueError("MiniLM weight file hash mismatch")

    train = _read_rows(sample_dir / "train.jsonl")
    validation = _read_rows(sample_dir / "validation.jsonl")
    source_train = _read_rows(split_dir / "train.jsonl")
    source_validation = _read_rows(split_dir / "validation.jsonl")
    if len(train) != len(source_train) or len(validation) != len(source_validation):
        raise ValueError("materialized sample counts differ from frozen source split")
    for rows, originals, split in (
        (train, source_train, "train"),
        (validation, source_validation, "validation"),
    ):
        if any(row.get("split") != split for row in rows):
            raise ValueError(f"materialized {split} manifest has a wrong split label")
        for row, original in zip(rows, originals, strict=True):
            if any(row.get(key) != value for key, value in original.items()):
                raise ValueError(f"materialization changed frozen source row {original['id']}")
            if row.get("question_type") not in QUESTION_TYPES:
                raise ValueError(f"unknown question type in {row['id']}")
            if not row["question"].strip() or row["target"] not in row["options"]:
                raise ValueError(f"invalid candidate decision row {row['id']}")
            if (
                row["question_type"] != "temporal_descriptive"
                and "Candidate statement:" not in row["question"]
            ):
                raise ValueError(f"reasoning candidate text missing from model input: {row['id']}")
            if row["question_type"] != "temporal_descriptive":
                if row["options"] != ["wrong", "correct"]:
                    raise ValueError(
                        f"reasoning choices are not the fixed binary options: {row['id']}"
                    )
                if row["candidate_statement"] not in row["question"]:
                    raise ValueError(f"candidate statement was not preserved in input: {row['id']}")
        reasoning_groups: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            if row["question_type"] == "temporal_descriptive":
                continue
            reasoning_groups.setdefault(str(row["parent_question_id"]), []).append(row)
        for parent_id, choices in reasoning_groups.items():
            if sorted(row["target"] for row in choices) != ["correct", "wrong"]:
                raise ValueError(
                    f"reasoning parent lacks one correct and one wrong row: {parent_id}"
                )
            if len({row["candidate_statement"] for row in choices}) != 2:
                raise ValueError(
                    f"reasoning parent has duplicate candidate statements: {parent_id}"
                )
    if {row["scene_index"] for row in train} & {row["scene_index"] for row in validation}:
        raise ValueError("train/validation scene overlap")
    train_media_hashes = {str(row["media_sha256"]) for row in train}
    validation_media_hashes = {str(row["media_sha256"]) for row in validation}
    if train_media_hashes & validation_media_hashes:
        raise ValueError("train/validation media identity overlap")
    validation_scene_counts = {
        kind: len({row["scene_index"] for row in validation if row["question_type"] == kind})
        for kind in QUESTION_TYPES
    }
    if any(
        count != int(source["validation_scene_count_per_question_type"])
        for count in validation_scene_counts.values()
    ):
        raise ValueError("validation scene counts do not match the frozen per-task quota")
    train_content = {
        (
            " ".join(row["question"].casefold().split()),
            tuple(sorted(" ".join(x.casefold().split()) for x in row["options"])),
        )
        for row in train
    }
    val_content = {
        (
            " ".join(row["question"].casefold().split()),
            tuple(sorted(" ".join(x.casefold().split()) for x in row["options"])),
        )
        for row in validation
    }
    if train_content & val_content:
        raise ValueError("normalized question/options overlap across train and validation")

    fetch_report = json.loads((sample_dir / "fetch-report.json").read_text(encoding="utf-8"))
    if (
        fetch_report.get("status") != "complete_temporal_media_sample"
        or fetch_report.get("revision") != source["revision"]
    ):
        raise ValueError("media materialization report is incomplete or from another revision")
    if fetch_report.get("scene_overlap") != 0 or fetch_report.get("media_hash_overlap") != 0:
        raise ValueError("media materialization report has split overlap")
    media_by_hash = {}
    for row in train + validation:
        media = sample_dir / row["media_path"]
        if not media.is_file() or sha256_file(media) != row["media_sha256"]:
            raise ValueError(f"missing or hash-mismatched media for {row['id']}")
        media_by_hash[row["media_sha256"]] = row["scene_index"]

    feature_report = json.loads(feature_report_path.read_text(encoding="utf-8"))
    if feature_report.get("status") != "complete_videoprism_framewise_feature_cache":
        raise ValueError("framewise feature cache is not complete")
    if feature_report.get("dataset_revision") != f"MIT-IBM/CLEVRER@{source['revision']}":
        raise ValueError("feature cache dataset revision mismatch")
    if (
        feature_report.get("model_revision") != VIDEO_REVISION
        or feature_report.get("upstream_source_commit") != VIDEO_COMMIT
    ):
        raise ValueError("feature cache VideoPrism provenance mismatch")
    if feature_report.get("checkpoint_sha256") != VIDEO_SHA256 or feature_report.get(
        "feature_cache_sha256"
    ) != sha256_file(feature_path):
        raise ValueError("feature cache checkpoint or file hash mismatch")
    expected_sample_manifests = {
        "train": {
            name: sha256_file(sample_dir / name)
            for name in ("train.jsonl", "validation.jsonl", "fetch-report.json")
        },
        "validation": {
            name: sha256_file(sample_dir / name)
            for name in ("train.jsonl", "validation.jsonl", "fetch-report.json")
        },
    }
    if (
        feature_report.get("train_manifest_sha256") != expected_sample_manifests["train"]
        or feature_report.get("evaluation_manifest_sha256")
        != expected_sample_manifests["validation"]
    ):
        raise ValueError("feature cache sample manifest hashes do not match the materialized split")
    if feature_report.get("unique_video_count") != len(media_by_hash):
        raise ValueError("feature cache video count differs from selected media assets")
    with np.load(feature_path, allow_pickle=False) as cache:
        hashes = cache["media_sha256"].astype(str).tolist()
        video = np.asarray(cache["features"], dtype=np.float32)
    if len(hashes) != len(set(hashes)) or set(hashes) != set(media_by_hash):
        raise ValueError("feature cache media identities do not match the frozen split")
    if video.shape != (len(hashes), 8, 768) or not np.isfinite(video).all():
        raise ValueError("VideoPrism features must be finite [unique_media,8,768]")
    return {
        "config_sha256": sha256_file(config_path),
        "source_split_hashes": expected_split_report["counts"]["split_file_sha256"],
        "source_split_preflight_sha256": sha256_file(split_dir / "preflight.json"),
        "excluded_scenes_sha256": sha256_file(split_dir / "excluded-scenes.json"),
        "source_questions_sha256": expected_split_report["source_questions_sha256"],
        "sample_manifest_hashes": {
            name: sha256_file(sample_dir / name)
            for name in ("train.jsonl", "validation.jsonl", "fetch-report.json")
        },
        "feature_cache_sha256": sha256_file(feature_path),
        "feature_report_sha256": sha256_file(feature_report_path),
        "text_weights_sha256": TEXT_SHA256,
        "train_rows": train,
        "validation_rows": validation,
        "feature_by_hash": dict(zip(hashes, video, strict=True)),
        "train_id_order_sha256": _ordered_ids_hash(train),
        "validation_id_order_sha256": _ordered_ids_hash(validation),
        "train_scene_count": len({row["scene_index"] for row in train}),
        "validation_scene_count": len({row["scene_index"] for row in validation}),
        "unique_media_count": len(media_by_hash),
        "unique_train_parent_questions": len(
            {row.get("parent_question_id", row["id"]) for row in train}
        ),
        "unique_validation_parent_questions": len(
            {row.get("parent_question_id", row["id"]) for row in validation}
        ),
        "source_counts": {
            "train": {
                str(source): sum(row.get("source") == source for row in train)
                for source in sorted({row.get("source") for row in train})
            },
            "validation": {
                str(source): sum(row.get("source") == source for row in validation)
                for source in sorted({row.get("source") for row in validation})
            },
        },
        "question_type_counts": {"train": _counts(train), "validation": _counts(validation)},
        "validation_scene_counts_by_task": validation_scene_counts,
        "fetch_report": fetch_report,
    }


def _batch_text(rows_features: list[Tensor], indexes: list[int]) -> tuple[Tensor, Tensor]:
    max_options = max(rows_features[index].shape[0] for index in indexes)
    feature_size = rows_features[indexes[0]].shape[-1]
    batch = torch.zeros((len(indexes), max_options, feature_size), dtype=torch.float32)
    mask = torch.zeros((len(indexes), max_options), dtype=torch.bool)
    for offset, index in enumerate(indexes):
        values = rows_features[index]
        batch[offset, : values.shape[0]] = values
        mask[offset, : values.shape[0]] = True
    return batch, mask


def _predict(
    model: nn.Module,
    text_features: list[Tensor],
    device: torch.device,
    videos: Tensor | None = None,
    batch_size: int = 64,
) -> list[Tensor]:
    model.eval()
    output: list[Tensor | None] = [None] * len(text_features)
    with torch.inference_mode():
        for start in range(0, len(text_features), batch_size):
            indexes = list(range(start, min(start + batch_size, len(text_features))))
            if videos is None:
                flat = torch.cat([text_features[index] for index in indexes]).to(device)
                scores = model(flat).squeeze(-1).float().cpu()
                offset = 0
                for index in indexes:
                    width = text_features[index].shape[0]
                    output[index] = scores[offset : offset + width]
                    offset += width
            else:
                padded, mask = _batch_text(text_features, indexes)
                logits = (
                    model(videos[indexes].to(device), padded.to(device), mask.to(device))
                    .float()
                    .cpu()
                )
                for row_offset, index in enumerate(indexes):
                    output[index] = logits[row_offset, : text_features[index].shape[0]]
    return [value for value in output if value is not None]


def _metrics_by_type(
    logits: list[Tensor], targets: list[int], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    by_type = {}
    for kind in QUESTION_TYPES:
        indexes = [index for index, row in enumerate(rows) if row["question_type"] == kind]
        if not indexes:
            raise ValueError(f"split missing task type: {kind}")
        by_type[kind] = {
            **_metrics([logits[index] for index in indexes], [targets[index] for index in indexes]),
            "unique_scenes": len({rows[index]["scene_index"] for index in indexes}),
        }
    macro = {
        metric: sum(float(by_type[kind][metric]) for kind in QUESTION_TYPES) / len(QUESTION_TYPES)
        for metric in ("accuracy", "nll", "brier", "ece_15_bins")
    }
    return {"macro_question_type": macro, "by_question_type": by_type}


def _fit_variant(
    *,
    name: str,
    model: nn.Module,
    train_text: list[Tensor],
    train_targets: list[int],
    train_rows: list[dict[str, Any]],
    validation_text: list[Tensor],
    validation_targets: list[int],
    validation_rows: list[dict[str, Any]],
    train_videos: Tensor | None,
    validation_videos: Tensor | None,
    config: dict[str, Any],
    seed: int,
    device: torch.device,
    run_dir: Path,
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    batch_size = int(config["training"]["batch_questions"])
    epochs = int(config["training"]["epochs"])
    generator = torch.Generator().manual_seed(seed)
    history = []
    best_state = None
    best_epoch = 0
    best_nll = float("inf")
    started = time.perf_counter()
    evaluation_seconds = 0.0
    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        order = torch.randperm(len(train_rows), generator=generator).tolist()
        losses = []
        gradient_norms = []
        for start in range(0, len(order), batch_size):
            indexes = order[start : start + batch_size]
            targets = torch.tensor([train_targets[index] for index in indexes], device=device)
            if train_videos is None:
                flat = torch.cat([train_text[index] for index in indexes]).to(device)
                flat_logits = model(flat).squeeze(-1)
                losses_by_row = []
                offset = 0
                for row_index in indexes:
                    width = train_text[row_index].shape[0]
                    losses_by_row.append(
                        F.cross_entropy(
                            flat_logits[offset : offset + width][None, :],
                            targets[len(losses_by_row) : len(losses_by_row) + 1],
                        )
                    )
                    offset += width
                loss = torch.stack(losses_by_row).mean()
            else:
                padded, mask = _batch_text(train_text, indexes)
                logits = model(train_videos[indexes].to(device), padded.to(device), mask.to(device))
                loss = F.cross_entropy(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss in {name} epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(f"non-finite gradient norm in {name} epoch {epoch}")
            gradient_norms.append(float(gradient_norm.detach().cpu()))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        evaluation_started = time.perf_counter()
        train_logits = _predict(model, train_text, device, train_videos)
        validation_logits = _predict(model, validation_text, device, validation_videos)
        evaluation_seconds += time.perf_counter() - evaluation_started
        train_metrics = _metrics_by_type(train_logits, train_targets, train_rows)
        validation_metrics = _metrics_by_type(
            validation_logits, validation_targets, validation_rows
        )
        history.append(
            {
                "epoch": epoch,
                "optimizer_updates": epoch * ((len(train_rows) + batch_size - 1) // batch_size),
                "train_cross_entropy_mean_batch": sum(losses) / len(losses),
                "gradient_norm_mean": sum(gradient_norms) / len(gradient_norms),
                "epoch_seconds": time.perf_counter() - epoch_started,
                "train_metrics": train_metrics,
                "validation_metrics": validation_metrics,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        current_nll = validation_metrics["macro_question_type"]["nll"]
        if current_nll < best_nll:
            best_nll = current_nll
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
        del train_logits, validation_logits

    if best_state is None:
        raise RuntimeError(f"no validation checkpoint selected for {name}")
    final_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state, strict=True)
    model.eval()
    evaluation_started = time.perf_counter()
    selected_train_logits = _predict(model, train_text, device, train_videos)
    selected_logits = _predict(model, validation_text, device, validation_videos)
    evaluation_seconds += time.perf_counter() - evaluation_started
    selected_train_metrics = _metrics_by_type(selected_train_logits, train_targets, train_rows)
    metrics = _metrics_by_type(selected_logits, validation_targets, validation_rows)
    train_validation_gap = {
        kind: {
            metric: float(selected_train_metrics["by_question_type"][kind][metric])
            - float(metrics["by_question_type"][kind][metric])
            for metric in ("accuracy", "nll", "brier", "ece_15_bins")
        }
        for kind in QUESTION_TYPES
    }
    safe_name = name.replace("-", "_")
    best_path = run_dir / f"{safe_name}-best.pt"
    final_path = run_dir / f"{safe_name}-final.pt"
    torch.save(best_state, best_path)
    # The final epoch state is separately retained from the selected state.
    torch.save(final_state, final_path)
    if name == "text_only":
        reload = CandidateScorer(
            int(config["model"]["text_feature_size"]),
            hidden_size=int(config["model"]["scorer_hidden_size"]),
        )
    else:
        reload = TemporalCandidateScorer(
            text_feature_size=int(config["model"]["text_feature_size"]),
            video_feature_size=int(config["model"]["video_feature_size"]),
            temporal_hidden_size=int(config["model"]["temporal_hidden_size"]),
            scorer_hidden_size=int(config["model"]["scorer_hidden_size"]),
        )
    reload.load_state_dict(
        torch.load(best_path, map_location="cpu", weights_only=True), strict=True
    )
    reload = reload.to(device).eval()
    reloaded_logits = _predict(reload, validation_text, device, validation_videos)
    max_delta = max(
        float((a.float() - b.float()).abs().max())
        for a, b in zip(selected_logits, reloaded_logits, strict=True)
    )
    if max_delta != 0.0:
        raise ValueError(f"best checkpoint reload changed logits for {name}: {max_delta}")
    predictions = []
    for row, logits, target in zip(
        validation_rows, reloaded_logits, validation_targets, strict=True
    ):
        probability = torch.softmax(logits.float(), dim=-1)
        prediction = int(probability.argmax())
        predictions.append(
            {
                "id": row["id"],
                "scene_index": row["scene_index"],
                "question_type": row["question_type"],
                "question": row["question"],
                "options": row["options"],
                "target": row["target"],
                "target_index": target,
                "prediction_index": prediction,
                "probabilities": probability.tolist(),
            }
        )
    (run_dir / f"{safe_name}-history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / f"{safe_name}-validation-predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in predictions),
        encoding="utf-8",
    )
    return {
        "best_epoch": best_epoch,
        "best_validation_macro_nll": best_nll,
        "selected_train_metrics": selected_train_metrics,
        "selected_validation_metrics": metrics,
        "train_validation_gap_by_question_type": train_validation_gap,
        "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "best_checkpoint_sha256": sha256_file(best_path),
        "final_checkpoint_sha256": sha256_file(final_path),
        "best_validation_predictions_sha256": sha256_file(
            run_dir / f"{safe_name}-validation-predictions.jsonl"
        ),
        "training_history_sha256": sha256_file(run_dir / f"{safe_name}-history.json"),
        "training_seconds": time.perf_counter() - started,
        "evaluation_seconds": evaluation_seconds,
        "validation_id_order_sha256": _ordered_ids_hash(validation_rows),
        "optimizer_updates": best_epoch * ((len(train_rows) + batch_size - 1) // batch_size),
        "max_optimizer_updates": epochs * ((len(train_rows) + batch_size - 1) // batch_size),
        "unique_train_examples": len(train_rows),
        "selected_checkpoint_example_presentations": len(train_rows) * best_epoch,
        "full_run_example_presentations": len(train_rows) * epochs,
    }


def _paired_scene_bootstrap(
    baseline_predictions: list[dict[str, Any]],
    temporal_predictions: list[dict[str, Any]],
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    if [row["id"] for row in baseline_predictions] != [row["id"] for row in temporal_predictions]:
        raise ValueError("paired bootstrap requires identical prediction ID order")
    metric_names = ("accuracy", "nll", "brier")
    baseline_by_type_scene: dict[str, dict[int, list[list[float]]]] = {
        kind: {} for kind in QUESTION_TYPES
    }
    temporal_by_type_scene: dict[str, dict[int, list[list[float]]]] = {
        kind: {} for kind in QUESTION_TYPES
    }
    for baseline, temporal in zip(baseline_predictions, temporal_predictions, strict=True):
        if any(
            baseline[key] != temporal[key]
            for key in ("id", "scene_index", "question_type", "target_index")
        ):
            raise ValueError("paired prediction target metadata mismatch")
        target = int(baseline["target_index"])
        metric_rows = []
        for prediction in (baseline, temporal):
            probabilities = np.asarray(prediction["probabilities"], dtype=np.float64)
            if probabilities.ndim != 1 or not np.isfinite(probabilities).all():
                raise ValueError("prediction probabilities must be finite vectors")
            if target >= len(probabilities) or not np.isclose(probabilities.sum(), 1.0, atol=1e-6):
                raise ValueError("prediction probability shape or normalization mismatch")
            onehot = np.zeros_like(probabilities)
            onehot[target] = 1.0
            metric_rows.append(
                [
                    float(int(np.argmax(probabilities) == target)),
                    float(-np.log(max(probabilities[target], 1e-12))),
                    float(np.square(probabilities - onehot).sum()),
                ]
            )
        kind = str(baseline["question_type"])
        scene = int(baseline["scene_index"])
        baseline_by_type_scene[kind].setdefault(scene, []).append(metric_rows[0])
        temporal_by_type_scene[kind].setdefault(scene, []).append(metric_rows[1])

    scene_differences: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for kind in QUESTION_TYPES:
        baseline_scenes = baseline_by_type_scene[kind]
        temporal_scenes = temporal_by_type_scene[kind]
        if set(baseline_scenes) != set(temporal_scenes) or len(baseline_scenes) < 2:
            raise ValueError(f"paired bootstrap scene sets differ or are too small for {kind}")
        scenes = sorted(baseline_scenes)
        baseline_means, temporal_means = [], []
        for scene in scenes:
            baseline_means.append(np.asarray(baseline_scenes[scene]).mean(axis=0))
            temporal_means.append(np.asarray(temporal_scenes[scene]).mean(axis=0))
        scene_differences[kind] = (
            np.stack(baseline_means),
            np.stack(temporal_means),
        )

    rng = np.random.default_rng(seed)
    values = {kind: np.empty((replicates, len(metric_names))) for kind in QUESTION_TYPES}
    macro_values = np.empty_like(values[QUESTION_TYPES[0]])
    for replicate in range(replicates):
        by_type = []
        for kind in QUESTION_TYPES:
            baseline_means, temporal_means = scene_differences[kind]
            indexes = rng.integers(0, len(baseline_means), size=len(baseline_means))
            delta = temporal_means[indexes].mean(axis=0) - baseline_means[indexes].mean(axis=0)
            values[kind][replicate] = delta
            by_type.append(delta)
        macro_values[replicate] = np.stack(by_type).mean(axis=0)

    def interval(array: np.ndarray) -> dict[str, list[float]]:
        return {
            metric: [float(value) for value in np.quantile(array[:, index], [0.025, 0.975])]
            for index, metric in enumerate(metric_names)
        }

    return {
        "method": "paired scene-cluster percentile bootstrap, stratified by question type",
        "replicates": replicates,
        "seed": seed,
        "interval": "95% percentile",
        "difference_direction": "VideoPrism+BiLSTM minus text-only",
        "macro_question_type": interval(macro_values),
        "by_question_type": {kind: interval(values[kind]) for kind in QUESTION_TYPES},
        "selection_note": (
            "Both checkpoints were selected on this same development validation; intervals are "
            "descriptive, not independent-test guarantees."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve(strict=True)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["training"].get("deterministic_algorithms") is not True:
        raise ValueError("this run requires deterministic PyTorch algorithms")
    if (
        config["training"]["epochs"]
        * (
            (
                len(
                    _read_rows(
                        Path(config["source"]["sample_dir"]).resolve(strict=True) / "train.jsonl"
                    )
                )
                + config["training"]["batch_questions"]
                - 1
            )
            // config["training"]["batch_questions"]
        )
        != config["training"]["max_optimizer_updates"]
    ):
        raise ValueError("configured epoch schedule does not match max_optimizer_updates")
    checked = _validate(config_path, config)
    report = {
        key: value
        for key, value in checked.items()
        if key not in {"train_rows", "validation_rows", "feature_by_hash", "fetch_report"}
    }
    report.update(
        {
            "status": "preflight_pass",
            "experiment_id": config["experiment_id"],
            "seed": config["seed"],
            "train_examples": len(checked["train_rows"]),
            "validation_examples": len(checked["validation_rows"]),
            "sealed_audit_loaded": False,
        }
    )
    preflight_path = Path(config["outputs"]["preflight"]).resolve()
    run_dir = Path(config["outputs"]["run_dir"]).resolve()
    if args.preflight_only:
        if preflight_path.exists():
            raise FileExistsError(f"refusing to overwrite {preflight_path}")
        preflight_path.parent.mkdir(parents=True, exist_ok=True)
        preflight_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        return
    gpu_gate = _require_gpu_gate(config)
    if not preflight_path.is_file() or run_dir.exists():
        raise FileNotFoundError("a completed preflight is required and run_dir must be new")
    saved = json.loads(preflight_path.read_text(encoding="utf-8"))
    for key in (
        "config_sha256",
        "source_split_hashes",
        "source_split_preflight_sha256",
        "excluded_scenes_sha256",
        "source_questions_sha256",
        "sample_manifest_hashes",
        "feature_cache_sha256",
        "feature_report_sha256",
        "train_id_order_sha256",
        "validation_id_order_sha256",
    ):
        if saved.get(key) != report.get(key):
            raise ValueError(f"preflight no longer matches current inputs: {key}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("configured temporal-mix fit requires CUDA")
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "effective-config.yaml").write_bytes(config_path.read_bytes())
    (run_dir / "preflight-used.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.reset_peak_memory_stats(device)
    model_path = Path(config["text_encoder"]["model_path"]).resolve(strict=True)
    text_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    encoder = AutoModel.from_pretrained(model_path, local_files_only=True).eval().to(device)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    train_rows = checked["train_rows"]
    validation_rows = checked["validation_rows"]
    train_text, train_targets = _text_features(train_rows, tokenizer, encoder, device)
    validation_text, validation_targets = _text_features(
        validation_rows, tokenizer, encoder, device
    )
    text_encoding_seconds = time.perf_counter() - text_started
    del encoder, tokenizer
    torch.cuda.empty_cache()
    feature_by_hash = checked["feature_by_hash"]
    train_video = torch.from_numpy(
        np.stack([feature_by_hash[row["media_sha256"]] for row in train_rows]).copy()
    )
    validation_video = torch.from_numpy(
        np.stack([feature_by_hash[row["media_sha256"]] for row in validation_rows]).copy()
    )
    del feature_by_hash
    torch.cuda.empty_cache()

    model_kwargs = {
        "text_feature_size": int(config["model"]["text_feature_size"]),
        "video_feature_size": int(config["model"]["video_feature_size"]),
        "temporal_hidden_size": int(config["model"]["temporal_hidden_size"]),
        "scorer_hidden_size": int(config["model"]["scorer_hidden_size"]),
    }
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    text_only = CandidateScorer(
        model_kwargs["text_feature_size"], hidden_size=model_kwargs["scorer_hidden_size"]
    )
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    temporal = TemporalCandidateScorer(**model_kwargs)
    run_started = datetime.now(UTC).isoformat()
    text_result = _fit_variant(
        name="text_only",
        model=text_only,
        train_text=train_text,
        train_targets=train_targets,
        train_rows=train_rows,
        validation_text=validation_text,
        validation_targets=validation_targets,
        validation_rows=validation_rows,
        train_videos=None,
        validation_videos=None,
        config=config,
        seed=seed,
        device=device,
        run_dir=run_dir,
    )
    temporal_result = _fit_variant(
        name="videoprism_bilstm_text",
        model=temporal,
        train_text=train_text,
        train_targets=train_targets,
        train_rows=train_rows,
        validation_text=validation_text,
        validation_targets=validation_targets,
        validation_rows=validation_rows,
        train_videos=train_video,
        validation_videos=validation_video,
        config=config,
        seed=seed,
        device=device,
        run_dir=run_dir,
    )
    baseline_predictions = _read_rows(run_dir / "text_only-validation-predictions.jsonl")
    temporal_predictions = _read_rows(
        run_dir / "videoprism_bilstm_text-validation-predictions.jsonl"
    )
    paired_bootstrap = _paired_scene_bootstrap(
        baseline_predictions,
        temporal_predictions,
        seed=int(config["training"]["bootstrap_seed"]),
        replicates=int(config["training"]["bootstrap_replicates"]),
    )
    torch.cuda.synchronize(device)
    final = {
        "status": "complete_clevrer_temporal_mix_readout",
        "experiment_id": config["experiment_id"],
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "config_sha256": checked["config_sha256"],
        "preflight_sha256": sha256_file(preflight_path),
        "source_split_hashes": checked["source_split_hashes"],
        "source_split_preflight_sha256": checked["source_split_preflight_sha256"],
        "excluded_scenes_sha256": checked["excluded_scenes_sha256"],
        "source_questions_sha256": checked["source_questions_sha256"],
        "sample_manifest_hashes": checked["sample_manifest_hashes"],
        "feature_cache_sha256": checked["feature_cache_sha256"],
        "source": {
            "dataset_id": config["source"]["dataset_id"],
            "revision": config["source"]["revision"],
            "license": config["source"]["license"],
        },
        "encoders": {
            "video_id": config["features"]["model_id"],
            "video_revision": VIDEO_REVISION,
            "video_checkpoint_sha256": VIDEO_SHA256,
            "video_frames": 8,
            "text_id": config["text_encoder"]["model_id"],
            "text_revision": TEXT_REVISION,
            "text_weights_sha256": TEXT_SHA256,
        },
        "train_id_order_sha256": checked["train_id_order_sha256"],
        "validation_id_order_sha256": checked["validation_id_order_sha256"],
        "seed": seed,
        "initialization": "fresh readout and BiLSTM; frozen MiniLM and VideoPrism",
        "started_at_utc": run_started,
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "sealed_audit_loaded": False,
        "training_data": {
            "train_examples": len(train_rows),
            "train_scenes": checked["train_scene_count"],
            "validation_examples": len(validation_rows),
            "validation_scenes": checked["validation_scene_count"],
            "unique_video_assets": checked["unique_media_count"],
            "unique_train_parent_questions": checked["unique_train_parent_questions"],
            "unique_validation_parent_questions": checked["unique_validation_parent_questions"],
            "source_counts": checked["source_counts"],
            "question_type_counts": checked["question_type_counts"],
            "realized_train_task_mix_percent": {
                kind: 100.0 * checked["question_type_counts"]["train"][kind] / len(train_rows)
                for kind in QUESTION_TYPES
            },
            "text_encoding_seconds": text_encoding_seconds,
        },
        "hardware": {
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
            "peak_vram_bytes": torch.cuda.max_memory_allocated(device),
            "peak_vram_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "python_version": platform.python_version(),
            "transformers_version": importlib.metadata.version("transformers"),
            "deterministic_algorithms": True,
        },
        "gpu_gate": gpu_gate,
        "variants": {"text_only": text_result, "videoprism_bilstm_text": temporal_result},
        "paired_validation_scene_bootstrap": paired_bootstrap,
        "note": (
            "Development validation selected both checkpoints; this is not a blind or "
            "general-world result."
        ),
        "trainable_parameter_counts": {
            "text_only": text_result["trainable_parameter_count"],
            "videoprism_bilstm_text": temporal_candidate_parameter_count(temporal),
        },
        "prequantization_component_parameter_counts": {
            "frozen_minilm_text_encoder": TEXT_PARAMETER_COUNT,
            "frozen_videoprism_video_encoder": VIDEO_PARAMETER_COUNT,
            "text_only_with_minilm": (
                TEXT_PARAMETER_COUNT + text_result["trainable_parameter_count"]
            ),
            "videoprism_bilstm_text": (
                TEXT_PARAMETER_COUNT
                + VIDEO_PARAMETER_COUNT
                + temporal_candidate_parameter_count(temporal)
            ),
        },
    }
    (run_dir / "run-metadata.json").write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
