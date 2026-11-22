"""Fit a frozen MiniLM-L12 linear readout on the pinned BANKING77 split."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import AutoModel

from scripts.train_banking77_minilm_readout import (
    CLASS_COUNT,
    extract_features,
    load_and_validate_data,
    metric_summary,
    sha256_file,
    train_readout,
)

MODEL_ID = "sentence-transformers/all-MiniLM-L12-v2"
MODEL_REVISION = "a50ef00143b4d5391434df20ae11632588ac25be"
EXPERIMENT_ID = "banking77-minilm-l12-frozen-readout-v1"


def verify_model_snapshot(model_dir: Path, expected_hashes: dict[str, str]) -> dict[str, str]:
    actual: dict[str, str] = {}
    for relative, expected in expected_hashes.items():
        path = model_dir / relative
        if not path.is_file():
            raise FileNotFoundError(f"Pinned model file is missing: {relative}")
        actual[relative] = sha256_file(path)
        if actual[relative] != expected:
            raise ValueError(f"Pinned MiniLM-L12 file hash mismatch: {relative}")
    return actual


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output_dir}")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("Unexpected experiment config")
    if config["model"].get("revision") != MODEL_REVISION:
        raise ValueError("Unexpected pinned model revision")
    actual_hashes = verify_model_snapshot(args.model_dir, config["model"]["files_sha256"])

    # Count actual loaded weights and fail closed if architecture differs from the pin.
    encoder = AutoModel.from_pretrained(args.model_dir, local_files_only=True).cpu().eval()
    parameter_count = sum(parameter.numel() for parameter in encoder.parameters())
    if parameter_count != config["model"]["parameters"]:
        raise ValueError(f"Unexpected MiniLM-L12 parameter count: {parameter_count}")
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    del encoder

    train_rows, validation_rows, split_manifest = load_and_validate_data(args.data_dir)
    split_hash = sha256_file(args.data_dir / "split-manifest.json")
    if split_hash != config["source"]["split_manifest_sha256"]:
        raise ValueError("BANKING77 split differs from the fixed L3/L6 comparison")
    started = datetime.now(UTC).isoformat()
    timer = time.perf_counter()
    train_features = extract_features(train_rows, args.model_dir)
    validation_features = extract_features(validation_rows, args.model_dir)
    train_labels = torch.tensor([int(row["label"]) for row in train_rows], dtype=torch.long)
    validation_labels = torch.tensor(
        [int(row["label"]) for row in validation_rows], dtype=torch.long
    )
    fit_config = config["training"]
    model, fit = train_readout(
        train_features,
        train_labels,
        validation_features,
        validation_labels,
        seed=fit_config["seed"],
        learning_rate=fit_config["learning_rate"],
        weight_decay=fit_config["weight_decay"],
        batch_size=fit_config["batch_size"],
        max_epochs=fit_config["max_epochs"],
        patience=fit_config["early_stopping_patience"],
        min_delta=fit_config["early_stopping_min_delta"],
    )
    with torch.inference_mode():
        logits = model((validation_features - fit["train_mean"]) / fit["train_std"])
        probabilities = torch.softmax(logits, dim=-1)
        predictions = logits.argmax(dim=-1)

    args.output_dir.mkdir(parents=True)
    checkpoint_path = args.output_dir / "best-readout.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "train_mean": fit["train_mean"],
            "train_std": fit["train_std"],
            "class_count": CLASS_COUNT,
            "best_epoch": fit["best_epoch"],
        },
        checkpoint_path,
    )
    history_path = args.output_dir / "training-history.jsonl"
    history_path.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in fit["history"]),
        encoding="utf-8",
    )
    prediction_path = args.output_dir / "validation-predictions.jsonl"
    with prediction_path.open("w", encoding="utf-8", newline="\n") as stream:
        for index, row in enumerate(validation_rows):
            stream.write(
                json.dumps(
                    {
                        "sample_id": row["sample_id"],
                        "content_group_id": row["content_group_id"],
                        "target": int(validation_labels[index]),
                        "prediction": int(predictions[index]),
                        "probabilities": probabilities[index].tolist(),
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
    target_counts = Counter(int(row["label"]) for row in validation_rows)
    metrics: dict[str, Any] = {
        **metric_summary(logits, validation_labels),
        "validation_rows": len(validation_rows),
        "class_count": CLASS_COUNT,
        "per_class": {},
        "source_group_confidence_interval": None,
        "source_group_confidence_interval_note": (
            "Source-person/conversation identities are unavailable."
        ),
    }
    for label in range(CLASS_COUNT):
        mask = validation_labels.eq(label)
        metrics["per_class"][str(label)] = {
            "count": target_counts[label],
            "accuracy": float(predictions[mask].eq(validation_labels[mask]).float().mean()),
            "nll": nn.functional.cross_entropy(logits[mask], validation_labels[mask]).item(),
        }
    metrics_path = args.output_dir / "validation-metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    metadata = {
        "experiment_id": EXPERIMENT_ID,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_file_sha256": actual_hashes,
        "model_weights_sha256": actual_hashes["model.safetensors"],
        "base_encoder_parameters": parameter_count,
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "total_parameters": parameter_count
        + sum(parameter.numel() for parameter in model.parameters()),
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "train_unique_groups": len({row["content_group_id"] for row in train_rows}),
        "validation_unique_groups": len({row["content_group_id"] for row in validation_rows}),
        "official_test_loaded": False,
        "device": "cpu",
        "dtype": "float32",
        "seed": fit_config["seed"],
        "selection_metric": "minimum validation NLL",
        "best_epoch": fit["best_epoch"],
        "epochs_completed": fit["epochs_completed"],
        "started_at_utc": started,
        "ended_at_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": time.perf_counter() - timer,
        "best_checkpoint_sha256": sha256_file(checkpoint_path),
        "history_sha256": sha256_file(history_path),
        "validation_predictions_sha256": sha256_file(prediction_path),
        "validation_metrics_sha256": sha256_file(metrics_path),
        "validation_metrics": metric_summary(logits, validation_labels),
        "dataset_revision": split_manifest["source_revision"],
        "dataset_split_sha256": split_hash,
        "config_sha256": sha256_file(args.config),
        "official_test_payload_downloaded_or_opened": False,
    }
    metadata_path = args.output_dir / "run-metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
