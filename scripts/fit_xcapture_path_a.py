"""Fit a small trainable decision lens on fixed, frozen X-Capture features."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from tiny_omni_decision.xcapture_decision import (
    MODALITY_DIMS,
    FrozenFeatureDecision,
    decision_json,
    multiclass_metrics,
    paired_cross_modal_loss,
    projection_parameter_count,
)

SEED = 17
MAX_STEPS = 1000
EVALUATION_INTERVAL = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DEFAULT_FEATURES = Path(
    "C:/CodexArtifacts/pretrained-reuse-r2-local/"
    "xcapture-cross-point-features-seed17-20261128-attempt-01"
)
DEFAULT_OUTPUT = Path(
    "C:/CodexArtifacts/pretrained-reuse-r2-local/"
    "xcapture-path-a-seed17-20261128-attempt-01"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_feature_split(
    features_dir: Path, partition: str
) -> tuple[dict[str, torch.Tensor], list[str], list[list[str]]]:
    report_path = features_dir / "feature-smoke.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "PASS_FROZEN_FEATURE_RUNTIME_SMOKE_ONLY":
        raise ValueError("feature runtime receipt is not PASS")
    if report.get("evaluation_partition_accessed") is not False:
        raise ValueError("feature receipt records evaluation partition access")
    records = report["records"]
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(records):
        if row["partition"] == partition:
            grouped[row["object_id"]].append(index)
    point_counts = {len(indices) for indices in grouped.values()}
    if not grouped or len(point_counts) != 1 or next(iter(point_counts)) < 2:
        raise ValueError(f"{partition} requires an equal count of at least two points per object")
    for object_id, indices in grouped.items():
        grouped[object_id] = sorted(indices, key=lambda index: records[index]["point_id"])
    object_ids = sorted(grouped)
    arrays = {}
    for modality, expected_dim in MODALITY_DIMS.items():
        path = features_dir / f"{modality}-features.npy"
        if sha256_file(path) != report["feature_array_sha256"][path.name]:
            raise ValueError(f"feature array hash mismatch: {path}")
        values = np.load(path, allow_pickle=False)
        if values.ndim != 2 or values.shape[0] != len(records) or values.shape[1] != expected_dim:
            raise ValueError(f"unexpected {modality} feature shape: {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError(f"{modality} features contain NaN or Inf")
        arrays[modality] = torch.from_numpy(
            np.stack([values[grouped[object_id]] for object_id in object_ids]).astype(
                np.float32, copy=False
            )
        )
    ordered_point_ids = [
        [records[index]["point_id"] for index in grouped[object_id]]
        for object_id in object_ids
    ]
    return arrays, object_ids, ordered_point_ids


def _pair_logits(
    model: FrozenFeatureDecision,
    query_modality: str,
    gallery_modality: str,
    query: torch.Tensor,
    gallery: torch.Tensor,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    logits_by_direction = []
    predictions = []
    n_objects = query.shape[0]
    option_ids = list(range(n_objects))
    point_count = query.shape[1]
    for query_point in range(point_count):
        for gallery_point in range(point_count):
            if query_point == gallery_point:
                continue
            logits = model.pair_logits(
                query_modality,
                query[:, query_point],
                gallery_modality,
                gallery[:, gallery_point],
            )
            logits_by_direction.append(logits)
            probabilities = logits.softmax(dim=-1)
            preds = probabilities.argmax(dim=-1)
            for index in range(n_objects):
                predictions.append(
                    {
                        "direction": f"point{query_point}_to_point{gallery_point}",
                        "target_index": index,
                        "prediction_index": int(preds[index]),
                        "confidence": float(probabilities[index, preds[index]]),
                        "probabilities": [float(value) for value in probabilities[index]],
                        "option_indices": option_ids,
                    }
                )
    combined = torch.cat(logits_by_direction, dim=0)
    return combined, predictions


def _fused_query_logits(
    model: FrozenFeatureDecision,
    query_modalities: tuple[str, str],
    query: dict[str, torch.Tensor],
    candidate_modality: str,
    candidates: torch.Tensor,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    encoded = {
        modality: model.encode(modality, query[modality]) for modality in query_modalities
    }
    fused = F.normalize(sum(encoded.values()), dim=-1)
    candidate_vectors = model.encode(candidate_modality, candidates)
    scale = model.logit_scale.clamp(min=0.0, max=math.log(100.0)).exp()
    predictions = []
    count = candidates.shape[0]
    logits_by_direction = []
    point_count = candidates.shape[1]
    for query_point in range(point_count):
        for gallery_point in range(point_count):
            if query_point == gallery_point:
                continue
            logits = fused[:, query_point] @ candidate_vectors[:, gallery_point].T * scale
            logits_by_direction.append(logits)
            probabilities = logits.softmax(dim=-1)
            for index in range(count):
                pred = int(probabilities[index].argmax())
                predictions.append(
                    {
                        "direction": f"point{query_point}_to_point{gallery_point}",
                        "target_index": index,
                        "prediction_index": pred,
                        "confidence": float(probabilities[index, pred]),
                        "probabilities": [float(value) for value in probabilities[index]],
                        "option_indices": list(range(count)),
                    }
                )
    return torch.cat(logits_by_direction, dim=0), predictions


def evaluate_suite(
    model: FrozenFeatureDecision,
    features: dict[str, torch.Tensor],
    object_ids: list[str],
    *,
    include_predictions: bool = False,
) -> tuple[dict[str, Any], float]:
    arms: dict[str, dict[str, Any]] = {}
    primary_nlls = []
    for query_modality in MODALITY_DIMS:
        for gallery_modality in MODALITY_DIMS:
            if query_modality == gallery_modality:
                continue
            logits, rows = _pair_logits(
                model,
                query_modality,
                gallery_modality,
                features[query_modality],
                features[gallery_modality],
            )
            direction_count = features[query_modality].shape[1] * (
                features[query_modality].shape[1] - 1
            )
            targets = torch.arange(len(object_ids)).repeat(direction_count)
            metrics = multiclass_metrics(logits, targets=targets)
            arm = {
                **metrics,
                "query_count": int(logits.shape[0]),
                "independent_object_groups": len(object_ids),
                "queries_per_object_group": direction_count,
                "gallery_size": len(object_ids),
            }
            if include_predictions:
                arm["predictions"] = [
                    {
                        **row,
                        "query_object_id": object_ids[row_index % len(object_ids)],
                        "target_gallery_object_id": object_ids[row["target_index"]],
                        "predicted_gallery_object_id": object_ids[row["prediction_index"]],
                    }
                    for row_index, row in enumerate(rows)
                ]
            arms[f"{query_modality}_to_{gallery_modality}"] = arm
            primary_nlls.append(metrics["nll"])

    fused_arms = {}
    for query_modalities, candidate_modality in (
        (("image", "tactile"), "audio"),
        (("image", "audio"), "tactile"),
        (("audio", "tactile"), "image"),
    ):
        logits, rows = _fused_query_logits(
            model,
            query_modalities,
            features,
            candidate_modality,
            features[candidate_modality],
        )
        direction_count = features[query_modalities[0]].shape[1] * (
            features[query_modalities[0]].shape[1] - 1
        )
        targets = torch.arange(len(object_ids)).repeat(direction_count)
        metrics = multiclass_metrics(logits, targets=targets)
        arm_name = f"{'_plus_'.join(query_modalities)}_to_{candidate_modality}"
        arm = {
            **metrics,
            "query_count": int(logits.shape[0]),
            "independent_object_groups": len(object_ids),
            "queries_per_object_group": direction_count,
            "gallery_size": len(object_ids),
        }
        if include_predictions:
            arm["predictions"] = [
                {
                    **row,
                    "query_object_id": object_ids[row_index % len(object_ids)],
                    "target_gallery_object_id": object_ids[row["target_index"]],
                    "predicted_gallery_object_id": object_ids[row["prediction_index"]],
                }
                for row_index, row in enumerate(rows)
            ]
        fused_arms[arm_name] = arm

    macro = {
        metric: float(np.mean([row[metric] for row in arms.values()]))
        for metric in ("accuracy", "nll", "brier", "ece")
    }
    minimum_modality_accuracy = min(row["accuracy"] for row in arms.values())
    return (
        {
            "gallery_size": len(object_ids),
            "query_count_per_direction": len(object_ids),
            "cross_modal": arms,
            "fused_query": fused_arms,
            "macro_cross_modal": macro,
            "minimum_cross_modal_accuracy": minimum_modality_accuracy,
        },
        float(np.mean(primary_nlls)),
    )


def run_training(features_dir: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite Path A output: {output_dir}")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_initialized():
        raise RuntimeError("Path A run must remain CPU-only")

    train_features, train_ids, train_points = load_feature_split(features_dir, "train")
    validation_features, validation_ids, validation_points = load_feature_split(
        features_dir, "validation"
    )
    if set(train_ids) & set(validation_ids):
        raise ValueError("train/validation object identity overlap")
    model = FrozenFeatureDecision()
    parameter_count = projection_parameter_count(model)
    points_per_object = train_features["image"].shape[1]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    output_dir.mkdir(parents=True)
    started_at = datetime.now(UTC).isoformat()
    start = time.perf_counter()
    best = {"step": 0, "selector_nll": float("inf"), "state": None}
    history = []

    def evaluate_step(step: int, train_loss: float) -> None:
        nonlocal best
        model.eval()
        with torch.inference_mode():
            validation_result, selector_nll = evaluate_suite(
                model, validation_features, validation_ids
            )
            train_result, train_selector_nll = evaluate_suite(model, train_features, train_ids)
        if selector_nll < best["selector_nll"]:
            best = {
                "step": step,
                "selector_nll": selector_nll,
                "state": {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                },
            }
        history.append(
            {
                "step": step,
                "train_contrastive_ce": train_loss,
                "train_macro_cross_modal": train_result["macro_cross_modal"],
                "train_selector_macro_nll": train_selector_nll,
                "validation_macro_cross_modal": validation_result["macro_cross_modal"],
                "validation_selector_macro_nll": selector_nll,
                "validation_minimum_cross_modal_accuracy": validation_result[
                    "minimum_cross_modal_accuracy"
                ],
                "validation_fused_query": {
                    name: {key: value for key, value in metrics.items() if key != "predictions"}
                    for name, metrics in validation_result["fused_query"].items()
                },
                "learning_rate": optimizer.param_groups[0]["lr"],
                "cumulative_unique_train_points": (
                    len(train_ids) * points_per_object if step > 0 else 0
                ),
                "cumulative_unique_train_feature_rows_per_modality": (
                    len(train_ids) * points_per_object if step > 0 else 0
                ),
                "cumulative_train_feature_row_presentations_per_modality": (
                    step * len(train_ids) * points_per_object
                ),
                "ordered_cross_point_pairs_per_object_per_update": (
                    points_per_object * (points_per_object - 1)
                ),
                "elapsed_seconds": time.perf_counter() - start,
            }
        )

    model.eval()
    with torch.inference_mode():
        initial_loss = float(paired_cross_modal_loss(model, train_features).item())
    evaluate_step(0, initial_loss)

    for step in range(1, MAX_STEPS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = paired_cross_modal_loss(model, train_features)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite train loss at step {step}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"non-finite gradient norm at step {step}")
        optimizer.step()
        model.logit_scale.data.clamp_(min=0.0, max=math.log(100.0))
        if step % EVALUATION_INTERVAL == 0 or step == MAX_STEPS:
            evaluate_step(step, float(loss.detach().item()))

    if best["state"] is None:
        raise RuntimeError("validation checkpoint selection did not produce a checkpoint")
    last_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    torch.save(last_state, output_dir / "last.pt")
    model.load_state_dict(best["state"], strict=True)
    model.eval()
    with torch.inference_mode():
        reference_result, reference_selector_nll = evaluate_suite(
            model, validation_features, validation_ids, include_predictions=True
        )
    torch.save(best["state"], output_dir / "best-validation.pt")
    saved_best_state = torch.load(
        output_dir / "best-validation.pt", map_location="cpu", weights_only=True
    )
    model.load_state_dict(saved_best_state, strict=True)
    model.eval()
    with torch.inference_mode():
        best_result, best_selector_nll = evaluate_suite(
            model, validation_features, validation_ids, include_predictions=True
        )
        reload_probe = model.option_probabilities(
            "image",
            validation_features["image"][:2, 0],
            "audio",
            validation_features["audio"][:, 1],
        )
    if abs(best_selector_nll - best["selector_nll"]) > 1e-7:
        raise ValueError("best checkpoint reload changed selector metric")
    if best_selector_nll != reference_selector_nll or best_result != reference_result:
        raise ValueError("best checkpoint reload changed saved predictions")

    # Fixed-seed controls use the selected validation checkpoint and the same gallery.
    controls = {}
    validation_direction_count = validation_features["image"].shape[1] * (
        validation_features["image"].shape[1] - 1
    )
    for modality in MODALITY_DIMS:
        zero_logits = torch.zeros(
            (
                validation_direction_count * len(validation_ids),
                len(validation_ids),
            )
        )
        uniform_targets = torch.arange(len(validation_ids)).repeat(
            validation_direction_count
        )
        controls[f"uniform_{modality}"] = multiclass_metrics(
            zero_logits, targets=uniform_targets
        )
    permutation = torch.randperm(
        len(validation_ids), generator=torch.Generator().manual_seed(SEED)
    )
    if torch.equal(permutation, torch.arange(len(validation_ids))):
        permutation = torch.roll(permutation, shifts=1)
    shuffled_features = {
        modality: tensor[permutation] for modality, tensor in validation_features.items()
    }
    # Re-evaluate cross-modal arms with candidate representations shuffled by object.
    shuffled_pair_results = {}
    for query_modality in MODALITY_DIMS:
        for gallery_modality in MODALITY_DIMS:
            if query_modality == gallery_modality:
                continue
            logits, _ = _pair_logits(
                model,
                query_modality,
                gallery_modality,
                validation_features[query_modality],
                shuffled_features[gallery_modality],
            )
            shuffled_pair_results[f"{query_modality}_to_{gallery_modality}"] = (
                multiclass_metrics(logits, targets=uniform_targets)
            )

    elapsed = time.perf_counter() - start
    train_losses = [row["train_contrastive_ce"] for row in history]
    adapter_path = output_dir / "best-validation.pt"
    config = {
        "seed": SEED,
        "max_steps": MAX_STEPS,
        "evaluation_interval": EVALUATION_INTERVAL,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "gradient_clip_norm": 1.0,
        "selection_rule": (
            "minimum validation macro NLL over six ordered cross-modal retrieval arms; "
            "earliest tie wins"
        ),
        "embedding_dim": model.projections["image"].out_features,
        "points_per_object": points_per_object,
        "trainable_parameter_count_includes_frozen_encoders": False,
        "lora": False,
        "frozen_encoders": True,
        "trainable_parameter_count": parameter_count,
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    report = {
        "status": "PASS_PATH_A_FROZEN_ENCODER_DECISION_RUN",
        "experiment_id": "xcapture-path-a-crosspoint-v1-seed17",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": elapsed,
        "training_device": "cpu",
        "seed": SEED,
        "features_dir": str(features_dir),
        "feature_receipt_sha256": sha256_file(features_dir / "feature-smoke.json"),
        "sample_counts": {
            "train_objects": len(train_ids),
            "train_points": len(train_ids) * points_per_object,
            "validation_objects": len(validation_ids),
            "validation_points": len(validation_ids) * points_per_object,
            "train_unique_capture_points": len(train_ids) * points_per_object,
            "validation_unique_capture_points": len(validation_ids) * points_per_object,
            "train_unique_points_reused_each_update": len(train_ids) * points_per_object,
            "validation_feature_rows_per_modality": len(validation_ids) * points_per_object,
            "evaluation_media_accessed": False,
            "evaluation_objects_loaded": 0,
        },
        "unique_train_object_ids": train_ids,
        "unique_validation_object_ids": validation_ids,
        "train_validation_object_overlap": 0,
        "data_point_ids": {
            "train": train_points,
            "validation": validation_points,
        },
        "architecture": config,
        "selected_step": best["step"],
        "selected_validation_macro_nll": best["selector_nll"],
        "final_step": MAX_STEPS,
        "best_checkpoint_sha256": sha256_file(adapter_path),
        "best_checkpoint_bytes": adapter_path.stat().st_size,
        "last_checkpoint_sha256": sha256_file(output_dir / "last.pt"),
        "parameter_count": parameter_count,
        "learning_curve": history,
        "train_loss_first": train_losses[0],
        "train_loss_last": train_losses[-1],
        "selected_validation_metrics": best_result,
        "controls": {
            "uniform_probability_baseline": controls,
            "candidate_feature_permutation_seed17": shuffled_pair_results,
            "permutation": permutation.tolist(),
        },
        "best_reload_probability_probe_shape": list(reload_probe.shape),
        "training": True,
        "evaluation_partition_accessed": False,
        "cost_usd": 0,
    }
    # The model and structured report stay outside the repository.
    (output_dir / "training-history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    report_path = output_dir / "result.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["report_sha256"] = sha256_file(report_path)
    (output_dir / "result.sha256").write_text(
        report["report_sha256"] + "  result.json\n", encoding="ascii"
    )
    # Exercise the public JSON contract using the selected validation prediction.
    sample_probs = best_result["cross_modal"]["image_to_audio"]["predictions"][0][
        "probabilities"
    ]
    report["sample_decision_json"] = decision_json(validation_ids, sample_probs)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["report_sha256"] = sha256_file(report_path)
    (output_dir / "result.sha256").write_text(
        report["report_sha256"] + "  result.json\n", encoding="ascii"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = run_training(args.features_dir, args.output_dir)
    compact = {
        "status": report["status"],
        "selected_step": report["selected_step"],
        "validation": report["selected_validation_metrics"]["macro_cross_modal"],
        "fused_query": {
            name: {
                key: value
                for key, value in metrics.items()
                if key not in {"predictions", "query_count", "gallery_size"}
            }
            for name, metrics in report["selected_validation_metrics"]["fused_query"].items()
        },
        "trainable_parameters": report["parameter_count"],
        "elapsed_seconds": report["elapsed_seconds"],
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
