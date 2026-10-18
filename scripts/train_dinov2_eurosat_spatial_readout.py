"""Fit one seed-17 linear Decision readout on frozen EuroSAT spatial features."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tiny_omni_decision.clevr4_readout import classification_quality
from tiny_omni_decision.eurosat import eurosat_class_from_filename

SEED = 17
EPOCHS = 50
BATCH_SIZE = 256
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
CLASS_ORDER = (
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict:
    probs = torch.softmax(logits.float(), dim=-1)
    summary = classification_quality(probs, targets)
    supports = torch.bincount(targets, minlength=len(CLASS_ORDER)).tolist()
    recalls = []
    per_class = {}
    predictions = logits.argmax(-1)
    for index, name in enumerate(CLASS_ORDER):
        mask = targets.eq(index)
        recall = float(predictions[mask].eq(targets[mask]).float().mean()) if mask.any() else None
        per_class[name] = {"support": int(supports[index]), "accuracy": recall}
        if recall is not None:
            recalls.append(recall)
    summary["macro_accuracy"] = float(np.mean(recalls))
    summary["balanced_accuracy"] = summary["macro_accuracy"]
    summary["per_class"] = per_class
    return summary


def predict(
    model: nn.Module,
    features: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    batch_size: int = BATCH_SIZE,
) -> torch.Tensor:
    outputs = []
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(features), batch_size):
            batch = (features[offset : offset + batch_size] - mean) / std
            outputs.append(model(batch).cpu())
    return torch.cat(outputs)


def write_predictions(path: Path, records: list[dict], logits: torch.Tensor) -> None:
    probs = torch.softmax(logits.float(), dim=-1).tolist()
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record, row in zip(records, probs, strict=True):
            target_index = CLASS_ORDER.index(record["label"])
            prediction_index = int(np.argmax(row))
            stream.write(
                json.dumps(
                    {
                        "id": record["id"],
                        "target": CLASS_ORDER[target_index],
                        "prediction": CLASS_ORDER[prediction_index],
                        "options": CLASS_ORDER,
                        "probabilities": row,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    feature_result = json.loads((args.features / "result.json").read_text(encoding="utf-8"))
    if feature_result.get("spatial_test_media_decoded") is not False:
        raise ValueError("feature artifact indicates the spatial test media was decoded")
    train_records = load_records(args.features / "train-records.jsonl")
    val_records = load_records(args.features / "validation-records.jsonl")
    train_ids = [row["id"] for row in train_records]
    val_ids = [row["id"] for row in val_records]
    if (
        not train_ids
        or not val_ids
        or len(set(train_ids)) != len(train_ids)
        or len(set(val_ids)) != len(val_ids)
    ):
        raise ValueError("train/validation IDs must be nonempty and unique")
    if set(train_ids) & set(val_ids):
        raise ValueError("train and validation IDs overlap")
    for records in (train_records, val_records):
        for row in records:
            if row["label"] != eurosat_class_from_filename(row["filename"]):
                raise ValueError(f"label/filename mismatch: {row['id']}")
    train = np.load(args.features / "train-features.npy", mmap_mode="r")
    validation = np.load(args.features / "validation-features.npy", mmap_mode="r")
    if train.shape != (len(train_records), 768) or validation.shape != (len(val_records), 768):
        raise ValueError("feature matrix shape does not match frozen record manifests")
    if not np.isfinite(train).all() or not np.isfinite(validation).all():
        raise ValueError("feature matrices contain non-finite values")
    for split in ("train", "validation"):
        expected_hash = feature_result["feature_file_sha256"][split]
        if sha256_file(args.features / f"{split}-features.npy") != expected_hash:
            raise ValueError(f"{split} feature hash mismatch")

    args.output.mkdir(parents=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(2)
    device = torch.device("cuda")
    train_tensor = torch.from_numpy(np.asarray(train).copy()).to(device)
    validation_tensor = torch.from_numpy(np.asarray(validation).copy()).to(device)
    train_targets = torch.tensor(
        [CLASS_ORDER.index(row["label"]) for row in train_records], device=device
    )
    validation_targets = torch.tensor([CLASS_ORDER.index(row["label"]) for row in val_records])
    mean = train_tensor.mean(0, keepdim=True)
    std = train_tensor.std(0, unbiased=False, keepdim=True).clamp_min(1e-6)
    model = nn.Linear(768, len(CLASS_ORDER)).to(device)
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters())
    if trainable_parameters != 7_690:
        raise ValueError(f"unexpected readout parameter count: {trainable_parameters}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    rng = np.random.default_rng(SEED)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    best_nll = float("inf")
    best_epoch = None
    history = []
    class_counts = torch.bincount(train_targets, minlength=len(CLASS_ORDER)).float()
    prior = class_counts / class_counts.sum()
    prior_logits = prior.clamp_min(1e-12).log().expand(len(validation_targets), -1)
    config = {
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "started_utc": datetime.now(UTC).isoformat(),
        "feature_result_sha256": sha256_file(args.features / "result.json"),
        "train_records_sha256": sha256_file(args.features / "train-records.jsonl"),
        "validation_records_sha256": sha256_file(args.features / "validation-records.jsonl"),
        "seed": SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "checkpoint_selector": "earliest minimum validation NLL",
        "train_examples": len(train_records),
        "validation_examples": len(val_records),
        "class_order": CLASS_ORDER,
        "trainable_readout_parameters": trainable_parameters,
        "frozen_dinov2_parameters": 22_056_576,
        "prequantization_parameters": 22_064_266,
        "spatial_test_read": False,
        "option_policy": "softmax over the fixed ten EuroSAT class option IDs",
        "normalization": "per-feature mean/std fitted on train features only",
        "gpu": torch.cuda.get_device_name(),
    }
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = rng.permutation(len(train_records))
        losses = []
        for offset in range(0, len(order), BATCH_SIZE):
            indices = torch.as_tensor(order[offset : offset + BATCH_SIZE], device=device)
            batch = (train_tensor.index_select(0, indices) - mean) / std
            logits = model(batch)
            loss = F.cross_entropy(logits, train_targets.index_select(0, indices))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        train_logits = predict(model, train_tensor, mean, std)
        val_logits = predict(model, validation_tensor, mean, std)
        train_metric = metrics(train_logits, train_targets.cpu())
        val_metric = metrics(val_logits, validation_targets)
        history_row = {
            "epoch": epoch,
            "train_ce": float(np.mean(losses)),
            "train_metrics": train_metric,
            "validation_metrics": val_metric,
            "elapsed_seconds": time.perf_counter() - started,
        }
        history.append(history_row)
        with (args.output / "training-history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(history_row, sort_keys=True, separators=(",", ":")) + "\n")
        if val_metric["nll"] < best_nll:
            best_nll = val_metric["nll"]
            best_epoch = epoch
            torch.save(
                {
                    "model": {
                        key: value.detach().cpu() for key, value in model.state_dict().items()
                    },
                    "mean": mean.detach().cpu(),
                    "std": std.detach().cpu(),
                },
                args.output / "best-readout.pt",
            )
            write_predictions(
                args.output / "best-validation-predictions.jsonl", val_records, val_logits
            )
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_ce": history_row["train_ce"],
                    "validation_nll": val_metric["nll"],
                    "validation_accuracy": val_metric["accuracy"],
                }
            ),
            flush=True,
        )

    torch.save(
        {
            "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "mean": mean.detach().cpu(),
            "std": std.detach().cpu(),
        },
        args.output / "final-readout.pt",
    )
    reloaded = nn.Linear(768, len(CLASS_ORDER))
    checkpoint = torch.load(args.output / "best-readout.pt", weights_only=True, map_location="cpu")
    reloaded.load_state_dict(checkpoint["model"])
    best_mean, best_std = checkpoint["mean"], checkpoint["std"]
    reloaded_validation_logits = predict(
        reloaded, torch.from_numpy(np.asarray(validation).copy()), best_mean, best_std
    )
    saved_prediction_rows = [
        json.loads(line)
        for line in (args.output / "best-validation-predictions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    saved_probabilities = torch.tensor(
        [row["probabilities"] for row in saved_prediction_rows], dtype=torch.float32
    )
    reloaded_probabilities = torch.softmax(reloaded_validation_logits.float(), dim=-1)
    reload_delta = float((reloaded_probabilities - saved_probabilities).abs().max())
    if reload_delta > 1e-6 or len(saved_prediction_rows) != len(val_records):
        raise ValueError("best checkpoint reload does not reproduce validation predictions")
    best_row = min(history, key=lambda row: (row["validation_metrics"]["nll"], row["epoch"]))
    best_summary = best_row["validation_metrics"]
    result = {
        **config,
        "finished_utc": datetime.now(UTC).isoformat(),
        "selected_epoch": best_epoch,
        "final_epoch": EPOCHS,
        "selected_train_metrics": best_row["train_metrics"],
        "selected_validation_metrics": best_summary,
        "final_train_metrics": history[-1]["train_metrics"],
        "final_validation_metrics": history[-1]["validation_metrics"],
        "validation_train_prior_metrics": metrics(prior_logits, validation_targets),
        "best_reload_validation_logits_delta": reload_delta,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated(),
        "best_readout_sha256": sha256_file(args.output / "best-readout.pt"),
        "final_readout_sha256": sha256_file(args.output / "final-readout.pt"),
        "history_sha256": sha256_file(args.output / "training-history.jsonl"),
        "validation_predictions_sha256": sha256_file(
            args.output / "best-validation-predictions.jsonl"
        ),
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
