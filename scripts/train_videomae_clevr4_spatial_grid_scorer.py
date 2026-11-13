"""Fit and selector-evaluate one fixed spatial-grid Path A candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from scripts.train_frozen_text_probe import _metrics
from tiny_omni_decision.dataset import CLEVR4_TAXONOMIES, sha256_file
from tiny_omni_decision.videomae_spatial import (
    FEATURE_WIDTH,
    HIDDEN_WIDTH,
    TASK_EMBED_WIDTH,
    TASK_NAMES,
    SpatialGridClevr4Scorer,
)

SEED = 17
EPOCHS = 8
BATCH_TASKS = 512
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
OPTION_TO_INDEX = {
    task: {str(option): index for index, option in enumerate(options)}
    for task, options in CLEVR4_TAXONOMIES.items()
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _ordered_id_hash(ids: list[str]) -> str:
    return hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()


def _label_matrix(rows: list[dict[str, Any]]) -> tuple[list[str], Tensor]:
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate label IDs")
    targets = torch.tensor(
        [[OPTION_TO_INDEX[task][str(row[task])] for task in TASK_NAMES] for row in rows],
        dtype=torch.long,
    )
    return ids, targets


def _evaluate(
    model: SpatialGridClevr4Scorer,
    features: Tensor,
    targets: Tensor,
    ids: list[str],
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    logits = torch.empty((len(ids), len(TASK_NAMES), 10), dtype=torch.float32)
    with torch.inference_mode():
        for start in range(0, len(ids), 1024):
            stop = min(start + 1024, len(ids))
            feature_batch = features[start:stop].to(device)
            for task_id in range(len(TASK_NAMES)):
                task_batch = torch.full((stop - start,), task_id, dtype=torch.long, device=device)
                logits[start:stop, task_id] = model(feature_batch, task_batch).cpu()
    per_task: dict[str, Any] = {}
    for task_id, task in enumerate(TASK_NAMES):
        option_count = len(CLEVR4_TAXONOMIES[task])
        per_task[task] = _metrics(
            [row[:option_count] for row in logits[:, task_id]], targets[:, task_id].tolist()
        )
    macro = {
        metric: float(np.mean([per_task[task][metric] for task in TASK_NAMES]))
        for metric in ("accuracy", "nll", "brier", "ece_15_bins")
    }
    macro["count_per_taxonomy"] = len(ids)
    macro["total_decisions"] = len(ids) * len(TASK_NAMES)
    predictions = []
    for row_index, sample_id in enumerate(ids):
        for task_id, task in enumerate(TASK_NAMES):
            options = [str(option) for option in CLEVR4_TAXONOMIES[task]]
            row_logits = logits[row_index, task_id, : len(options)]
            probabilities = torch.softmax(row_logits, dim=-1)
            target = int(targets[row_index, task_id])
            predictions.append(
                {
                    "image_id": sample_id,
                    "task": task,
                    "target": options[target],
                    "prediction": options[int(probabilities.argmax())],
                    "options": options,
                    "probabilities": probabilities.tolist(),
                }
            )
    return macro, per_task, predictions


def _save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )


def train(
    *,
    inputs: Path,
    spatial_inputs: Path,
    features_path: Path,
    config_path: Path,
    baseline_history_path: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output {output}")
    receipt_path = inputs / "receipt.json"
    receipt = _load_json(receipt_path)
    input_config = _load_json(inputs / "config.json")
    config = _load_json(config_path)
    if receipt.get("status") != "FROZEN_VIDEOMAE_CLEVR4_PATH_A_INPUTS":
        raise ValueError("frozen input receipt is missing")
    if receipt.get("config_sha256") != sha256_file(inputs / "config.json"):
        raise ValueError("frozen input config hash mismatch")
    if (
        receipt.get("cross_split_id_overlap") != 0
        or receipt.get("cross_split_media_sha256_overlap") != 0
    ):
        raise ValueError("frozen inputs contain cross-split overlap")
    if input_config.get("experiment_id") != "path-a-image-videomae-clevr4-100k-seed17":
        raise ValueError("unexpected frozen input experiment")
    if config.get("experiment_id") != "path-a-image-videomae-clevr4-spatial-grid-4x4-seed17":
        raise ValueError("unexpected spatial-grid candidate")
    if config.get("input_receipt_sha256") != sha256_file(receipt_path):
        raise ValueError("candidate config references a different frozen input receipt")
    representation = config.get("representation", {})
    if (
        representation.get("encoder_source_sha256"),
        representation.get("encoder_checkpoint_sha256"),
        representation.get("encoder_parameters"),
        representation.get("spatial_reduction"),
        representation.get("normalization"),
    ) != (
        "d132c97369f25207e380e78c59294fd00b93d2f6aa0fb5c4ded19eff26a5a3cb",
        "24fb71687fa3671b8387cadfbcbab0f72af695692e93cf1ecc82caa888626172",
        22_153_286,
        "adaptive average pooling of final patch tokens to 4x4",
        "apply pretrained fc_norm independently to each pooled grid cell",
    ):
        raise ValueError("candidate config differs from the pinned spatial representation")
    if config.get("representation", {}).get("feature_width") != FEATURE_WIDTH:
        raise ValueError("candidate feature width differs from spatial-grid implementation")
    if config.get("selection_policy") != {
        "reporting_partition_loaded": False,
        "sealed_audit_loaded": False,
        "candidate_count": 1,
        "no_seed_or_hyperparameter_sweep": True,
    }:
        raise ValueError("candidate selection policy differs from the predeclared single run")
    recipe = config.get("readout", {})
    if (
        config.get("seed"),
        recipe.get("hidden_width"),
        recipe.get("task_embedding_width"),
        recipe.get("epochs"),
        recipe.get("batch_tasks"),
        recipe.get("learning_rate"),
        recipe.get("weight_decay"),
        recipe.get("scheduler"),
    ) != (
        SEED,
        HIDDEN_WIDTH,
        TASK_EMBED_WIDTH,
        EPOCHS,
        BATCH_TASKS,
        LEARNING_RATE,
        WEIGHT_DECAY,
        "constant",
    ):
        raise ValueError("candidate config differs from the fixed matched recipe")
    for filename in (
        "train-labels.jsonl",
        "selector-labels.jsonl",
        "train-media-manifest.jsonl",
        "selector-media-manifest.jsonl",
    ):
        expected_hash = receipt.get("file_hashes", {}).get(filename)
        if sha256_file(inputs / filename) != expected_hash:
            raise ValueError(f"frozen input file hash mismatch: {filename}")
    spatial_receipt_path = spatial_inputs / "receipt.json"
    spatial_receipt = _load_json(spatial_receipt_path)
    if spatial_receipt.get("status") != "FROZEN_SPATIAL_GRID_TRAIN_SELECTOR_INPUTS":
        raise ValueError("spatial train/selector input receipt is missing")
    if spatial_receipt.get("parent_input_receipt_sha256") != sha256_file(receipt_path):
        raise ValueError("spatial input receipt references different frozen source inputs")
    if spatial_receipt.get("reporting_partition_loaded") is not False:
        raise ValueError("reporting partition must remain isolated")
    if spatial_receipt.get("feature_row_count") != 73_010:
        raise ValueError("spatial inputs must contain train and selector rows only")
    if config.get("data_policy", {}).get("reporting_partition_extracted") is not False:
        raise ValueError("spatial candidate must exclude reporting media from extraction")
    feature_result = _load_json(features_path.parent / "result.json")
    if feature_result.get("status") != "COMPLETE_FROZEN_SPATIAL_GRID_FEATURE_EXTRACTION":
        raise ValueError("complete spatial feature extraction result is missing")
    if feature_result.get("input_row_count") != 73_010:
        raise ValueError("spatial feature result must contain train and selector rows only")
    if feature_result.get("reporting_partition_extracted") is not False:
        raise ValueError("reporting media must remain outside the spatial feature cache")
    if (
        feature_result.get("encoder_source_sha256") != representation["encoder_source_sha256"]
        or feature_result.get("encoder_checkpoint_sha256")
        != representation["encoder_checkpoint_sha256"]
        or feature_result.get("spatial_grid") != [4, 4]
        or feature_result.get("frozen") is not True
    ):
        raise ValueError("spatial features differ from the pinned frozen representation")
    if feature_result.get("feature_sha256") != sha256_file(features_path):
        raise ValueError("spatial feature array hash differs from extraction result")
    feature_manifest = spatial_inputs / "train-selector-feature-manifest.jsonl"
    if feature_result.get("feature_manifest_sha256") != sha256_file(feature_manifest):
        raise ValueError("spatial feature manifest hash mismatch")
    if feature_result.get("input_manifest_sha256") != spatial_receipt.get(
        "feature_manifest_sha256"
    ):
        raise ValueError("feature extraction used a different frozen train/selector manifest")
    if sha256_file(feature_manifest) != spatial_receipt.get("feature_manifest_sha256"):
        raise ValueError("spatial feature manifest differs from frozen train/selector inputs")

    feature_rows = _load_jsonl(feature_manifest)
    feature_ids = [str(row["id"]) for row in feature_rows]
    media_hashes = [str(row["media_sha256"]) for row in feature_rows]
    if len(feature_ids) != 73_010 or len(set(feature_ids)) != len(feature_ids):
        raise ValueError("spatial feature IDs are incomplete or duplicated")
    if len(set(media_hashes)) != len(media_hashes):
        raise ValueError("spatial feature media identities are duplicated")
    feature_index = {sample_id: index for index, sample_id in enumerate(feature_ids)}
    features = np.load(features_path, mmap_mode="r", allow_pickle=False)
    if features.shape != (len(feature_ids), FEATURE_WIDTH) or features.dtype != np.float32:
        raise ValueError("spatial feature array shape or dtype mismatch")
    if not np.isfinite(features).all():
        raise ValueError("spatial feature array contains NaN or Inf")

    train_ids, train_targets = _label_matrix(_load_jsonl(inputs / "train-labels.jsonl"))
    selector_ids, selector_targets = _label_matrix(_load_jsonl(inputs / "selector-labels.jsonl"))
    if _ordered_id_hash(train_ids) != receipt.get("train_id_order_sha256"):
        raise ValueError("training ID order differs from frozen receipt")
    if _ordered_id_hash(selector_ids) != receipt.get("selector_id_order_sha256"):
        raise ValueError("selector ID order differs from frozen receipt")
    if len(train_ids) != 72_010 or len(selector_ids) != 1_000:
        raise ValueError("frozen train/selector counts differ from the candidate protocol")
    if spatial_receipt.get("train_count") != len(train_ids) or spatial_receipt.get(
        "selector_count"
    ) != len(selector_ids):
        raise ValueError("spatial input receipt counts differ from frozen labels")
    train_media = _load_jsonl(inputs / "train-media-manifest.jsonl")
    selector_media = _load_jsonl(inputs / "selector-media-manifest.jsonl")
    for rows in (train_media, selector_media):
        if any(
            feature_rows[feature_index[str(row["id"])]]["media_sha256"] != row["media_sha256"]
            for row in rows
        ):
            raise ValueError("frozen media identity differs from spatial feature row")
    if set(train_ids) & set(selector_ids):
        raise ValueError("training and selector IDs overlap")

    baseline_rows = _load_jsonl(baseline_history_path)
    if len(baseline_rows) != EPOCHS or baseline_rows[-1].get("epoch") != EPOCHS:
        raise ValueError("matched baseline history is incomplete")
    baseline_selector = baseline_rows[-1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {
        ":4096:8",
        ":16:8",
    }:
        raise RuntimeError("set CUBLAS_WORKSPACE_CONFIG before deterministic CUDA training")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None

    output.mkdir(parents=True, exist_ok=False)
    (output / "effective-config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    train_indices = np.asarray([feature_index[sample_id] for sample_id in train_ids])
    selector_indices = np.asarray([feature_index[sample_id] for sample_id in selector_ids])
    train_features = torch.from_numpy(
        np.array(features[train_indices], dtype=np.float32, copy=True)
    )
    selector_features = torch.from_numpy(
        np.array(features[selector_indices], dtype=np.float32, copy=True)
    )
    model = SpatialGridClevr4Scorer().to(device)
    readout_parameters = sum(parameter.numel() for parameter in model.parameters())
    if readout_parameters >= 1_000_000:
        raise ValueError(f"spatial readout exceeds 1M parameters: {readout_parameters}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    order_rng = random.Random(SEED)
    history: list[dict[str, Any]] = []
    best_nll = float("inf")
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    started_at = datetime.now(UTC).isoformat()
    fit_started = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        order = list(range(len(train_ids) * len(TASK_NAMES)))
        order_rng.shuffle(order)
        model.train()
        loss_sum = 0.0
        correct = 0
        consumed = 0
        for start in range(0, len(order), BATCH_TASKS):
            batch_rows = order[start : start + BATCH_TASKS]
            image_indices = torch.tensor([index // len(TASK_NAMES) for index in batch_rows])
            task_ids = torch.tensor([index % len(TASK_NAMES) for index in batch_rows])
            x = train_features.index_select(0, image_indices).to(device)
            y = train_targets[image_indices, task_ids].to(device)
            logits = model(x, task_ids.to(device))
            loss = F.cross_entropy(logits, y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite gradient norm at epoch {epoch}")
            optimizer.step()
            batch_count = len(batch_rows)
            loss_sum += float(loss.detach()) * batch_count
            correct += int((logits.argmax(-1) == y).sum())
            consumed += batch_count
        selector_macro, selector_by_task, selector_predictions = _evaluate(
            model, selector_features, selector_targets, selector_ids, device
        )
        row = {
            "epoch": epoch,
            "train_ce_mean_over_batches": loss_sum / consumed,
            "train_accuracy_rolling": correct / consumed,
            "train_task_examples_consumed": consumed,
            "selector_macro": selector_macro,
            "selector_by_taxonomy": selector_by_task,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        (output / "history.jsonl").write_text(
            "".join(json.dumps(item, sort_keys=True) + "\n" for item in history), encoding="utf-8"
        )
        if selector_macro["nll"] < best_nll:
            best_nll = selector_macro["nll"]
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_accuracy": row["train_accuracy_rolling"],
                    "selector": selector_macro,
                }
            ),
            flush=True,
        )
    fit_seconds = time.perf_counter() - fit_started
    if best_state is None:
        raise RuntimeError("no selector-selected checkpoint was produced")
    final_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    torch.save(final_state, output / "final-readout.pt")
    torch.save(best_state, output / "best-readout.pt")
    model.load_state_dict(
        torch.load(output / "best-readout.pt", map_location=device, weights_only=True)
    )
    reloaded_macro, reloaded_by_task, selector_predictions = _evaluate(
        model, selector_features, selector_targets, selector_ids, device
    )
    if reloaded_macro["nll"] != best_nll:
        raise ValueError("best selector checkpoint reload changed selector NLL")
    _save_jsonl(output / "selector-predictions.jsonl", selector_predictions)
    baseline_by_task = baseline_selector["selector_by_taxonomy"]
    report = {
        "status": "COMPLETE_SPATIAL_GRID_CANDIDATE_SELECTOR_ONLY",
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "config_sha256": sha256_file(config_path),
        "input_receipt_sha256": sha256_file(receipt_path),
        "spatial_input_receipt_sha256": sha256_file(spatial_receipt_path),
        "features_sha256": sha256_file(features_path),
        "features_manifest_sha256": sha256_file(feature_manifest),
        "baseline_history_sha256": sha256_file(baseline_history_path),
        "train_count": len(train_ids),
        "unique_train_assets": len(train_media),
        "selector_count": len(selector_ids),
        "unique_selector_assets": len(selector_media),
        "encoder": "OpenGVLab VideoMAE V2 ViT-Small K710",
        "encoder_parameters": 22_153_286,
        "encoder_frozen": True,
        "spatial_grid": [4, 4],
        "feature_width": FEATURE_WIDTH,
        "readout_parameters": readout_parameters,
        "total_parameters": 22_153_286 + readout_parameters,
        "seed": SEED,
        "epochs_completed": EPOCHS,
        "batch_tasks": BATCH_TASKS,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "scheduler": "constant",
        "training_seconds": fit_seconds,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else 0,
        "best_epoch": best_epoch,
        "selector_macro": reloaded_macro,
        "selector_by_taxonomy": reloaded_by_task,
        "matched_global_vector_baseline_selector_macro": baseline_selector["selector_macro"],
        "matched_global_vector_baseline_selector_by_taxonomy": baseline_by_task,
        "selector_only_delta": {
            "macro_accuracy": reloaded_macro["accuracy"]
            - baseline_selector["selector_macro"]["accuracy"],
            "macro_nll": reloaded_macro["nll"] - baseline_selector["selector_macro"]["nll"],
            "count_accuracy": reloaded_by_task["count"]["accuracy"]
            - baseline_by_task["count"]["accuracy"],
        },
        "history_sha256": sha256_file(output / "history.jsonl"),
        "best_checkpoint_sha256": sha256_file(output / "best-readout.pt"),
        "final_checkpoint_sha256": sha256_file(output / "final-readout.pt"),
        "selector_predictions_sha256": sha256_file(output / "selector-predictions.jsonl"),
        "reporting_partition_loaded": False,
        "reporting_partition_extracted": False,
        "sealed_audit_loaded": False,
        "source_files_modified": False,
        "interpretation_scope": (
            "single predeclared architecture candidate on the existing development selector; "
            "not an independent release evaluation"
        ),
    }
    (output / "run-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report["run_report_sha256"] = sha256_file(output / "run-report.json")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--spatial-inputs", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--baseline-history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    train(
        inputs=args.inputs,
        spatial_inputs=args.spatial_inputs,
        features_path=args.features,
        config_path=args.config,
        baseline_history_path=args.baseline_history,
        output=args.output,
    )


if __name__ == "__main__":
    main()
