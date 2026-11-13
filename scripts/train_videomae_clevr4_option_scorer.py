"""Fit a tiny taxonomy-conditioned option scorer on frozen VideoMAE features."""

from __future__ import annotations

import argparse
import json
import platform
import random
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from scripts.train_frozen_text_probe import _metrics
from tiny_omni_decision.dataset import CLEVR4_TAXONOMIES, sha256_file
from tiny_omni_decision.decision import DecisionOutput
from tiny_omni_decision.videomae_runtime import (
    FEATURE_WIDTH,
    HIDDEN_WIDTH,
    TASK_EMBED_WIDTH,
    TASK_NAMES,
    TASK_TO_INDEX,
    TaxonomyConditionedScorer,
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


OPTION_TO_INDEX = {
    task: {str(option): index for index, option in enumerate(options)}
    for task, options in CLEVR4_TAXONOMIES.items()
}
EPOCHS = 8
BATCH_TASKS = 512
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
SEED = 17


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def score_options(
    model: TaxonomyConditionedScorer,
    image_feature: Tensor,
    task: str,
    options: list[str],
    device: torch.device,
) -> tuple[list[str], list[float]]:
    if task not in OPTION_TO_INDEX:
        raise ValueError(f"unknown taxonomy {task!r}")
    if len(options) < 2 or len(set(options)) != len(options):
        raise ValueError("at least two unique supplied options are required")
    try:
        class_indices = [OPTION_TO_INDEX[task][str(option)] for option in options]
    except KeyError as error:
        raise ValueError(f"option {error.args[0]!r} is not valid for taxonomy {task}") from error
    model.eval()
    with torch.inference_mode():
        logits = model(
            image_feature.reshape(1, -1).to(device),
            torch.tensor([TASK_TO_INDEX[task]], dtype=torch.long, device=device),
        )[0]
        option_logits = logits[class_indices].float()
        probabilities = torch.softmax(option_logits, dim=-1).cpu().tolist()
    return [str(option) for option in options], [float(value) for value in probabilities]


def decision_json(
    model: TaxonomyConditionedScorer,
    image_feature: Tensor,
    task: str,
    options: list[str],
    request_id: str,
    device: torch.device,
) -> str:
    option_ids, probabilities = score_options(model, image_feature, task, options, device)
    probability_tensor = torch.tensor(probabilities, dtype=torch.float32)
    output = DecisionOutput(
        option_logits=probability_tensor.clamp_min(1e-12).log(),
        probabilities=probability_tensor,
        predicted_index=int(probability_tensor.argmax()),
        confidence=float(probability_tensor.max()),
    )
    return output.to_decision_json(
        request_id=f"clevr4.{task}:{request_id}",
        decision_type="choice",
        option_ids=option_ids,
        option_texts=option_ids,
    )


def _read_features(path: Path, manifest_path: Path) -> tuple[dict[str, Tensor], dict[str, str]]:
    rows = _load_jsonl(manifest_path)
    with np.load(path, allow_pickle=False) as archive:
        ids = archive["ids"].astype(str).tolist()
        media_hashes = archive["media_sha256"].astype(str).tolist()
        values = archive["features"]
    if values.shape != (len(ids), FEATURE_WIDTH) or len(ids) != len(rows):
        raise ValueError("feature shape/manifest count mismatch")
    if ids != [str(row["id"]) for row in rows]:
        raise ValueError("feature ID order differs from frozen feature manifest")
    if media_hashes != [str(row["media_sha256"]) for row in rows]:
        raise ValueError("feature media hashes differ from frozen feature manifest")
    if len(ids) != len(set(ids)) or len(media_hashes) != len(set(media_hashes)):
        raise ValueError("feature cache contains duplicate IDs or media identities")
    if not np.isfinite(values).all():
        raise ValueError("feature cache contains NaN or Inf")
    return (
        {sample_id: torch.from_numpy(values[index].copy()) for index, sample_id in enumerate(ids)},
        dict(zip(ids, media_hashes, strict=True)),
    )


def _label_matrix(rows: list[dict[str, Any]]) -> tuple[list[str], Tensor]:
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate label IDs")
    targets = []
    for row in rows:
        targets.append([OPTION_TO_INDEX[task][str(row[task])] for task in TASK_NAMES])
    return ids, torch.tensor(targets, dtype=torch.long)


def _evaluate(
    model: TaxonomyConditionedScorer,
    features: Tensor,
    labels: Tensor,
    device: torch.device,
    batch_size: int = 1024,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    all_logits = torch.empty((labels.shape[0], len(TASK_NAMES), 10), dtype=torch.float32)
    with torch.inference_mode():
        for start in range(0, labels.shape[0], batch_size):
            stop = min(start + batch_size, labels.shape[0])
            image_batch = features[start:stop].to(device)
            for task_id in range(len(TASK_NAMES)):
                task_batch = torch.full((stop - start,), task_id, dtype=torch.long, device=device)
                all_logits[start:stop, task_id] = model(image_batch, task_batch).cpu()
    by_task: dict[str, Any] = {}
    task_logits = []
    task_targets = []
    for task_id, task in enumerate(TASK_NAMES):
        option_count = len(CLEVR4_TAXONOMIES[task])
        logits = [row[:option_count] for row in all_logits[:, task_id]]
        targets = labels[:, task_id].tolist()
        by_task[task] = _metrics(logits, targets)
        task_logits.append(logits)
        task_targets.append(targets)
    macro = {
        metric: float(np.mean([by_task[name][metric] for name in TASK_NAMES]))
        for metric in ("accuracy", "nll", "brier", "ece_15_bins")
    }
    macro["count_per_taxonomy"] = int(labels.shape[0])
    macro["total_decisions"] = int(labels.shape[0] * len(TASK_NAMES))
    predictions = []
    for row_index in range(labels.shape[0]):
        for task_id, task in enumerate(TASK_NAMES):
            options = [str(option) for option in CLEVR4_TAXONOMIES[task]]
            option_logits = all_logits[row_index, task_id, : len(options)]
            probabilities = torch.softmax(option_logits, dim=-1)
            target_index = int(labels[row_index, task_id])
            predictions.append(
                {
                    "image_id": row_index,
                    "task": task,
                    "target": options[target_index],
                    "prediction": options[int(probabilities.argmax())],
                    "options": options,
                    "probabilities": probabilities.tolist(),
                }
            )
    return macro, by_task, predictions


def _save_predictions(path: Path, rows: list[dict[str, Any]], ids: list[str]) -> None:
    for row in rows:
        row["image_id"] = ids[row["image_id"]]
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )


def train(*, inputs: Path, features_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output {output}")
    input_receipt = json.loads((inputs / "receipt.json").read_text(encoding="utf-8"))
    effective = json.loads((inputs / "config.json").read_text(encoding="utf-8"))
    if input_receipt.get("status") != "FROZEN_VIDEOMAE_CLEVR4_PATH_A_INPUTS":
        raise ValueError("frozen input receipt is missing")
    if input_receipt.get("config_sha256") != sha256_file(inputs / "config.json"):
        raise ValueError("frozen config hash differs from input receipt")
    if effective.get("experiment_id") != "path-a-image-videomae-clevr4-100k-seed17":
        raise ValueError("unexpected experiment configuration")
    if (
        effective.get("task", {}).get("training_images"),
        effective.get("task", {}).get("selector_images"),
        effective.get("task", {}).get("one_time_reporting_images"),
    ) != (
        72_010,
        1_000,
        5_296,
    ):
        raise ValueError("effective config counts differ from frozen inputs")
    readout_config = effective.get("readout", {})
    configured_recipe = (
        effective.get("seed"),
        readout_config.get("hidden_width"),
        readout_config.get("task_embedding_width"),
        readout_config.get("epochs"),
        readout_config.get("batch_tasks"),
        readout_config.get("learning_rate"),
        readout_config.get("weight_decay"),
        readout_config.get("scheduler"),
    )
    if configured_recipe != (
        SEED,
        HIDDEN_WIDTH,
        TASK_EMBED_WIDTH,
        EPOCHS,
        BATCH_TASKS,
        LEARNING_RATE,
        WEIGHT_DECAY,
        "constant",
    ):
        raise ValueError("effective config differs from fixed training recipe")
    if input_receipt.get("features_manifest_sha256") != sha256_file(
        inputs / "features-manifest.jsonl"
    ):
        raise ValueError("feature manifest differs from frozen input receipt")
    feature_manifest = inputs / "features-manifest.jsonl"
    features_by_id, media_by_id = _read_features(features_path, feature_manifest)
    if len(features_by_id) != 78_306:
        raise ValueError(f"expected 78,306 frozen feature rows, got {len(features_by_id)}")
    train_rows = _load_jsonl(inputs / "train-labels.jsonl")
    selector_rows = _load_jsonl(inputs / "selector-labels.jsonl")
    train_ids, train_targets = _label_matrix(train_rows)
    selector_ids, selector_targets = _label_matrix(selector_rows)
    all_ids = train_ids + selector_ids
    missing = [sample_id for sample_id in all_ids if sample_id not in features_by_id]
    if missing:
        raise ValueError(f"frozen feature cache misses {len(missing)} train/selector IDs")
    train_manifest = _load_jsonl(inputs / "train-media-manifest.jsonl")
    selector_manifest = _load_jsonl(inputs / "selector-media-manifest.jsonl")
    for rows in (train_manifest, selector_manifest):
        if any(media_by_id[row["id"]] != row["media_sha256"] for row in rows):
            raise ValueError("feature media identity does not match frozen split manifest")
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    train_features = torch.stack([features_by_id[sample_id] for sample_id in train_ids])
    selector_features = torch.stack([features_by_id[sample_id] for sample_id in selector_ids])
    model = TaxonomyConditionedScorer().to(device)
    readout_parameters = parameter_count(model)
    if readout_parameters != 52_682:
        raise ValueError(f"unexpected readout parameter count {readout_parameters}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    order_rng = random.Random(SEED)
    history: list[dict[str, Any]] = []
    best_score = float("inf")
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    started_at = datetime.now(UTC).isoformat()
    fit_started = time.perf_counter()
    train_n = len(train_ids)
    for epoch in range(1, EPOCHS + 1):
        order = list(range(train_n * len(TASK_NAMES)))
        order_rng.shuffle(order)
        model.train()
        loss_sum = 0.0
        correct_count = 0
        consumed = 0
        for start in range(0, len(order), BATCH_TASKS):
            batch_rows = order[start : start + BATCH_TASKS]
            image_indices = torch.tensor([index // len(TASK_NAMES) for index in batch_rows])
            task_ids = torch.tensor(
                [index % len(TASK_NAMES) for index in batch_rows], dtype=torch.long
            )
            inputs_batch = train_features.index_select(0, image_indices).to(device)
            targets_batch = train_targets[image_indices, task_ids].to(device)
            logits = model(inputs_batch, task_ids.to(device))
            loss = F.cross_entropy(logits, targets_batch)
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
            correct_count += int((logits.argmax(-1) == targets_batch).sum())
            consumed += batch_count
        selector_macro, selector_by_task, selector_predictions = _evaluate(
            model, selector_features, selector_targets, device
        )
        row = {
            "epoch": epoch,
            "train_ce_mean_over_batches": loss_sum / consumed,
            "train_accuracy_rolling": correct_count / consumed,
            "train_task_examples_consumed": consumed,
            "selector_macro": selector_macro,
            "selector_by_taxonomy": selector_by_task,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        (output / "history.jsonl").write_text(
            "".join(json.dumps(item, sort_keys=True) + "\n" for item in history),
            encoding="utf-8",
        )
        if selector_macro["nll"] < best_score:
            best_score = selector_macro["nll"]
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
        raise RuntimeError("no validation-selected checkpoint was produced")
    final_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    torch.save(final_state, output / "final-readout.pt")
    torch.save(best_state, output / "best-readout.pt")
    model.load_state_dict(
        torch.load(output / "best-readout.pt", map_location=device, weights_only=True)
    )
    reloaded_macro, reloaded_by_task, selector_predictions = _evaluate(
        model, selector_features, selector_targets, device
    )
    if reloaded_macro["nll"] != best_score:
        raise ValueError("best checkpoint reload changed the fixed selector NLL")
    _save_predictions(
        output / "selector-best-predictions.jsonl", selector_predictions, selector_ids
    )

    # Load reporting labels only after checkpoint selection and reload are complete.
    report_rows = _load_jsonl(inputs / "report-labels.jsonl")
    report_ids, report_targets = _label_matrix(report_rows)
    report_manifest = _load_jsonl(inputs / "report-media-manifest.jsonl")
    if [row["id"] for row in report_manifest] != report_ids:
        raise ValueError("report labels/media order mismatch")
    if set(report_ids) & (set(train_ids) | set(selector_ids)):
        raise ValueError("reporting IDs overlap train or selector IDs")
    if any(sample_id not in features_by_id for sample_id in report_ids):
        raise ValueError("frozen feature cache misses reporting IDs")
    if any(media_by_id[row["id"]] != row["media_sha256"] for row in report_manifest):
        raise ValueError("report feature media identities differ from frozen manifest")
    report_features = torch.stack([features_by_id[sample_id] for sample_id in report_ids])
    report_macro, report_by_task, report_predictions = _evaluate(
        model, report_features, report_targets, device
    )
    _save_predictions(
        output / "reporting-one-time-predictions.jsonl", report_predictions, report_ids
    )
    report = {
        "status": "COMPLETE_FROZEN_FEATURE_PATH_A_CLEVR4_INTERNAL_VALIDATION",
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "inputs_receipt_sha256": sha256_file(inputs / "receipt.json"),
        "config_sha256": sha256_file(inputs / "config.json"),
        "features_sha256": sha256_file(features_path),
        "feature_manifest_sha256": sha256_file(feature_manifest),
        "train_count": len(train_ids),
        "unique_train_assets": len({row["media_sha256"] for row in train_manifest}),
        "train_question_count": len(train_ids) * len(TASK_NAMES),
        "selector_count": len(selector_ids),
        "reporting_count": len(report_ids),
        "unique_reporting_assets": len({row["media_sha256"] for row in report_manifest}),
        "encoder": "OpenGVLab VideoMAE V2 ViT-Small K710",
        "encoder_parameters": 22_153_286,
        "encoder_frozen": True,
        "encoder_checkpoint_sha256": (
            "24fb71687fa3671b8387cadfbcbab0f72af695692e93cf1ecc82caa888626172"
        ),
        "readout_parameters": readout_parameters,
        "total_parameters": 22_153_286 + readout_parameters,
        "readout_weight_bytes": (output / "best-readout.pt").stat().st_size,
        "encoder_weight_bytes": 44_334_609,
        "runtime_weight_bytes": (output / "best-readout.pt").stat().st_size + 44_334_609,
        "task_scope": (
            "closed-set synthetic image taxonomy choices; no natural-language "
            "question comprehension or temporal evidence"
        ),
        "task_names": TASK_NAMES,
        "option_vocabularies": CLEVR4_TAXONOMIES,
        "seed": SEED,
        "epochs_completed": EPOCHS,
        "batch_tasks": BATCH_TASKS,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "scheduler": "constant",
        "early_stopping": False,
        "training_task_examples_per_epoch": len(train_ids) * len(TASK_NAMES),
        "training_seconds": fit_seconds,
        "total_fit_and_evaluation_seconds": time.perf_counter() - fit_started,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else 0,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "best_epoch": best_epoch,
        "checkpoint_selection": (
            "minimum equal-taxonomy macro NLL on fixed 1000-image selector; earliest epoch on ties"
        ),
        "selector_metrics": reloaded_macro,
        "selector_by_taxonomy": reloaded_by_task,
        "one_time_reporting_metrics": report_macro,
        "reporting_by_taxonomy": report_by_task,
        "history_sha256": sha256_file(output / "history.jsonl"),
        "best_checkpoint_sha256": sha256_file(output / "best-readout.pt"),
        "final_checkpoint_sha256": sha256_file(output / "final-readout.pt"),
        "selector_predictions_sha256": sha256_file(output / "selector-best-predictions.jsonl"),
        "reporting_predictions_sha256": sha256_file(
            output / "reporting-one-time-predictions.jsonl"
        ),
        "sealed_audit_loaded": False,
        "source_files_modified": False,
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
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    train(inputs=args.inputs, features_path=args.features, output=args.output)


if __name__ == "__main__":
    main()
