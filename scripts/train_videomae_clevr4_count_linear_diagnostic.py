"""Fixed-budget count-only probe on frozen VideoMAE Clevr-4 features."""

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

from scripts.audit_clevr4_content_overlap import _load_jsonl, sha256_file
from scripts.train_frozen_text_probe import _metrics
from tiny_omni_decision.dataset import CLEVR4_TAXONOMIES

FEATURE_WIDTH = 384
COUNT_OPTIONS = CLEVR4_TAXONOMIES["count"]
COUNT_TO_INDEX = {str(option): index for index, option in enumerate(COUNT_OPTIONS)}
SEED = 17
EPOCHS = 8
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
TRAIN_COUNT = 72_010
SELECTOR_COUNT = 1_000
FEATURE_COUNT = 78_306
ENCODER_PARAMETERS = 22_153_286
BASELINE_MLP_PARAMETERS = 52_682


class CountLinearReadout(nn.Module):
    """Linear count classifier over the frozen 384-wide image feature."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(FEATURE_WIDTH, len(COUNT_OPTIONS))

    def forward(self, features: Tensor) -> Tensor:
        return self.linear(features)


def count_targets(rows: list[dict[str, Any]]) -> tuple[list[str], Tensor]:
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate image IDs in count labels")
    try:
        targets = [COUNT_TO_INDEX[str(row["count"])] for row in rows]
    except KeyError as error:
        raise ValueError(f"unknown count label {error.args[0]!r}") from error
    return ids, torch.tensor(targets, dtype=torch.long)


def _metrics_for_logits(logits: Tensor, targets: Tensor) -> dict[str, Any]:
    return _metrics([row for row in logits], targets.tolist())


def _load_split_features(
    *,
    features_path: Path,
    feature_manifest_path: Path,
    train_ids: list[str],
    selector_ids: list[str],
    train_manifest: list[dict[str, Any]],
    selector_manifest: list[dict[str, Any]],
) -> tuple[Tensor, Tensor, dict[str, str]]:
    if len(train_ids) != TRAIN_COUNT or len(selector_ids) != SELECTOR_COUNT:
        raise ValueError("frozen train/selector image counts do not match the preregistered split")
    if set(train_ids) & set(selector_ids):
        raise ValueError("train and selector IDs overlap")
    train_hashes = [str(row["media_sha256"]) for row in train_manifest]
    selector_hashes = [str(row["media_sha256"]) for row in selector_manifest]
    if (
        len(train_hashes) != len(set(train_hashes))
        or len(selector_hashes) != len(set(selector_hashes))
    ):
        raise ValueError("duplicate media identity within train or selector")
    if set(train_hashes) & set(selector_hashes):
        raise ValueError("train and selector media identities overlap")
    if [str(row["id"]) for row in train_manifest] != train_ids:
        raise ValueError("train label/media manifest order mismatch")
    if [str(row["id"]) for row in selector_manifest] != selector_ids:
        raise ValueError("selector label/media manifest order mismatch")

    manifest_rows = _load_jsonl(feature_manifest_path)
    with np.load(features_path, allow_pickle=False) as archive:
        cached_ids = archive["ids"].astype(str).tolist()
        cached_hashes = archive["media_sha256"].astype(str).tolist()
        cached_features = archive["features"]
    if cached_features.shape != (FEATURE_COUNT, FEATURE_WIDTH):
        raise ValueError(f"unexpected feature matrix shape {cached_features.shape}")
    if len(cached_ids) != len(set(cached_ids)) or len(cached_hashes) != len(set(cached_hashes)):
        raise ValueError("frozen feature cache contains duplicate IDs or media identities")
    if cached_ids != [str(row["id"]) for row in manifest_rows]:
        raise ValueError("feature ID order differs from frozen manifest")
    if cached_hashes != [str(row["media_sha256"]) for row in manifest_rows]:
        raise ValueError("feature media hashes differ from frozen manifest")
    if not np.isfinite(cached_features).all():
        raise ValueError("frozen feature cache contains NaN or Inf")

    row_by_id = {sample_id: index for index, sample_id in enumerate(cached_ids)}
    required_ids = train_ids + selector_ids
    missing = [sample_id for sample_id in required_ids if sample_id not in row_by_id]
    if missing:
        raise ValueError(f"feature cache misses {len(missing)} train/selector IDs")
    hash_by_id = dict(zip(cached_ids, cached_hashes, strict=True))
    for split_rows in (train_manifest, selector_manifest):
        for row in split_rows:
            sample_id = str(row["id"])
            if hash_by_id[sample_id] != str(row["media_sha256"]):
                raise ValueError("feature media identity differs from frozen split manifest")

    # Only the fixed train and selector IDs are materialized for this analysis.
    train = torch.from_numpy(
        np.stack([cached_features[row_by_id[sample_id]] for sample_id in train_ids]).copy()
    )
    selector = torch.from_numpy(
        np.stack([cached_features[row_by_id[sample_id]] for sample_id in selector_ids]).copy()
    )
    return train, selector, {
        "feature_cache_rows": len(cached_ids),
        "feature_cache_sha256": sha256_file(features_path),
        "feature_manifest_sha256": sha256_file(feature_manifest_path),
        "train_unique_media": len(set(train_hashes)),
        "selector_unique_media": len(set(selector_hashes)),
    }


def _save_predictions(path: Path, ids: list[str], targets: Tensor, logits: Tensor) -> None:
    probabilities = logits.softmax(dim=-1)
    lines = []
    for index, sample_id in enumerate(ids):
        target_index = int(targets[index])
        predicted_index = int(probabilities[index].argmax())
        lines.append(
            json.dumps(
                {
                    "image_id": sample_id,
                    "target": COUNT_OPTIONS[target_index],
                    "prediction": COUNT_OPTIONS[predicted_index],
                    "options": COUNT_OPTIONS,
                    "probabilities": probabilities[index].tolist(),
                },
                sort_keys=True,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(*, inputs: Path, features_path: Path, mlp_report: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output directory {output}")
    receipt_path = inputs / "receipt.json"
    config_path = inputs / "config.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    effective = json.loads(config_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "FROZEN_VIDEOMAE_CLEVR4_PATH_A_INPUTS":
        raise ValueError("frozen input receipt is missing")
    if receipt.get("config_sha256") != sha256_file(config_path):
        raise ValueError("frozen config hash differs from input receipt")
    if effective.get("experiment_id") != "path-a-image-videomae-clevr4-100k-seed17":
        raise ValueError("unexpected frozen experiment")
    if receipt.get("features_manifest_sha256") != sha256_file(inputs / "features-manifest.jsonl"):
        raise ValueError("frozen feature manifest differs from input receipt")
    train_rows = _load_jsonl(inputs / "train-labels.jsonl")
    selector_rows = _load_jsonl(inputs / "selector-labels.jsonl")
    train_ids, train_y = count_targets(train_rows)
    selector_ids, selector_y = count_targets(selector_rows)
    train_manifest = _load_jsonl(inputs / "train-media-manifest.jsonl")
    selector_manifest = _load_jsonl(inputs / "selector-media-manifest.jsonl")
    train_x, selector_x, integrity = _load_split_features(
        features_path=features_path,
        feature_manifest_path=inputs / "features-manifest.jsonl",
        train_ids=train_ids,
        selector_ids=selector_ids,
        train_manifest=train_manifest,
        selector_manifest=selector_manifest,
    )
    mlp = json.loads(mlp_report.read_text(encoding="utf-8"))
    if mlp.get("features_sha256") != integrity["feature_cache_sha256"]:
        raise ValueError("MLP baseline used a different frozen feature cache")
    if mlp.get("feature_manifest_sha256") != integrity["feature_manifest_sha256"]:
        raise ValueError("MLP baseline used a different feature manifest")
    baseline_count = mlp.get("selector_by_taxonomy", {}).get("count")
    if not isinstance(baseline_count, dict) or baseline_count.get("count") != SELECTOR_COUNT:
        raise ValueError("same-selector MLP count metrics are unavailable")

    output.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    model = CountLinearReadout().to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != 3_850:
        raise ValueError(f"unexpected count readout parameter count {parameter_count}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    shuffle_rng = random.Random(SEED)
    history: list[dict[str, Any]] = []
    started_at = datetime.now(UTC).isoformat()
    fit_started = time.perf_counter()
    model.train()
    for epoch in range(1, EPOCHS + 1):
        order = list(range(TRAIN_COUNT))
        shuffle_rng.shuffle(order)
        loss_sum = 0.0
        correct = 0
        consumed = 0
        for start in range(0, TRAIN_COUNT, BATCH_SIZE):
            indices = order[start : start + BATCH_SIZE]
            batch_x = train_x[indices].to(device)
            batch_y = train_y[indices].to(device)
            logits = model(batch_x)
            loss = F.cross_entropy(logits, batch_y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite gradient norm at epoch {epoch}")
            optimizer.step()
            count = len(indices)
            loss_sum += float(loss.detach()) * count
            correct += int((logits.argmax(dim=-1) == batch_y).sum())
            consumed += count

        model.eval()
        with torch.inference_mode():
            selector_logits = torch.cat(
                [model(selector_x[start : start + 1024].to(device)).cpu()
                 for start in range(0, SELECTOR_COUNT, 1024)],
                dim=0,
            )
        selector_metrics = _metrics_for_logits(selector_logits, selector_y)
        row = {
            "epoch": epoch,
            "train_ce": loss_sum / consumed,
            "train_accuracy_rolling": correct / consumed,
            "train_unique_examples_consumed_this_epoch": consumed,
            "train_unique_examples_total": TRAIN_COUNT,
            "selector_metrics": selector_metrics,
            "selector_role": "diagnostic_curve_only; no checkpoint or hyperparameter selection",
        }
        history.append(row)
        (output / "history.jsonl").write_text(
            "".join(json.dumps(item, sort_keys=True) + "\n" for item in history),
            encoding="utf-8",
        )
        model.train()
        print(json.dumps({"epoch": epoch, **row}, sort_keys=True), flush=True)

    fit_seconds = time.perf_counter() - fit_started
    final_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    final_checkpoint = output / "final-count-readout.pt"
    torch.save(final_state, final_checkpoint)
    reload_model = CountLinearReadout().to(device)
    reloaded_state = torch.load(final_checkpoint, map_location=device, weights_only=True)
    reload_model.load_state_dict(reloaded_state)
    reload_model.eval()
    with torch.inference_mode():
        final_logits = torch.cat(
            [reload_model(selector_x[start : start + 1024].to(device)).cpu()
             for start in range(0, SELECTOR_COUNT, 1024)],
            dim=0,
        )
    if not torch.equal(final_logits, selector_logits):
        raise ValueError("save/reload changed selector logits")
    final_metrics = _metrics_for_logits(final_logits, selector_y)
    _save_predictions(output / "selector-predictions.jsonl", selector_ids, selector_y, final_logits)

    config = {
        "experiment_id": "clevr4-videomae-count-linear-diagnostic-seed17-epoch8",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "seed": SEED,
        "device": str(device),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "encoder": "OpenGVLab VideoMAE V2 ViT-Small K710, frozen features",
        "encoder_parameters": ENCODER_PARAMETERS,
        "readout": "single affine linear layer, raw 384-wide frozen features to 10 count options",
        "readout_parameters": parameter_count,
        "total_parameters": ENCODER_PARAMETERS + parameter_count,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "scheduler": "constant",
        "selection": "fixed final epoch 8; selector curve diagnostic only",
        "train_examples_per_epoch": TRAIN_COUNT,
        "train_examples_total": TRAIN_COUNT * EPOCHS,
        "selector_examples": SELECTOR_COUNT,
        "reporting_labels_loaded": False,
        "reporting_features_scored": False,
        "sealed_audit_loaded": False,
        "frozen_input_receipt_sha256": sha256_file(receipt_path),
        "frozen_config_sha256": sha256_file(config_path),
        **integrity,
    }
    config_path_out = output / "effective-config.json"
    config_text = json.dumps(config, indent=2, sort_keys=True) + "\n"
    config_path_out.write_text(config_text, encoding="utf-8")
    report = {
        "status": "COMPLETE_FIXED_COUNT_LINEAR_DIAGNOSTIC",
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "fit_and_selector_seconds": fit_seconds,
        "peak_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
        "final_selector_count_metrics": final_metrics,
        "same_selector_existing_mlp_epoch8_count_metrics": baseline_count,
        "delta_linear_minus_mlp": {
            metric: float(final_metrics[metric] - baseline_count[metric])
            for metric in ("accuracy", "nll", "brier", "ece_15_bins")
        },
        "selector_metric_history": [
            {"epoch": row["epoch"], **row["selector_metrics"]} for row in history
        ],
        "training_history_sha256": sha256_file(output / "history.jsonl"),
        "selector_predictions_sha256": sha256_file(output / "selector-predictions.jsonl"),
        "checkpoint_sha256": sha256_file(final_checkpoint),
        "effective_config_sha256": sha256_file(config_path_out),
        "reload_logits_exact": True,
        "all_train_and_selector_images_unique": True,
        "reporting_labels_loaded": False,
        "reporting_features_scored": False,
        "sealed_audit_loaded": False,
    }
    result_text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    (output / "result.json").write_text(result_text, encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--mlp-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(
        inputs=args.inputs,
        features_path=args.features,
        mlp_report=args.mlp_report,
        output=args.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
