"""Fit a frozen MiniLM-L6 linear readout on the pinned BANKING77 split."""

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

from scripts.train_banking77_minilm_readout import (
    CLASS_COUNT,
    extract_features,
    load_and_validate_data,
    metric_summary,
    sha256_file,
    train_readout,
)

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MODEL_FILES_SHA256 = {
    "config.json": "953f9c0d463486b10a6871cc2fd59f223b2c70184f49815e7efbcab5d8908b41",
    "model.safetensors": "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db",
    "modules.json": "84e40c8e006c9b1d6c122e02cba9b02458120b5fb0c87b746c41e0207cf642cf",
    "sentence_bert_config.json": "fc1993fde0a95c24ec6c022539d41cf6e2f7c9721e5415d6fb6897472a9cd4b7",
    "special_tokens_map.json": "303df45a03609e4ead04bc3dc1536d0ab19b5358db685b6f3da123d05ec200e3",
    "tokenizer_config.json": "acb92769e8195aabd29b7b2137a9e6d6e25c476a4f15aa4355c233426c61576b",
    "tokenizer.json": "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037",
    "vocab.txt": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
    "1_Pooling/config.json": "4be450dde3b0273bb9787637cfbd28fe04a7ba6ab9d36ac48e92b11e350ffc23",
}


def verify_model_snapshot(model_dir: Path) -> dict[str, str]:
    actual: dict[str, str] = {}
    for relative, expected in MODEL_FILES_SHA256.items():
        path = model_dir / relative
        if not path.is_file():
            raise FileNotFoundError(f"Pinned model file is missing: {relative}")
        actual[relative] = sha256_file(path)
        if actual[relative] != expected:
            raise ValueError(f"Pinned MiniLM-L6 file hash mismatch: {relative}")
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
    if config.get("experiment_id") != "banking77-minilm-l6-frozen-readout-v1":
        raise ValueError("Unexpected experiment config")
    model_file_hashes = verify_model_snapshot(args.model_dir)
    train_rows, validation_rows, split_manifest = load_and_validate_data(args.data_dir)
    started = datetime.now(UTC).isoformat()
    time_start = time.perf_counter()
    train_features = extract_features(train_rows, args.model_dir)
    validation_features = extract_features(validation_rows, args.model_dir)
    train_labels = torch.tensor([int(row["label"]) for row in train_rows], dtype=torch.long)
    validation_labels = torch.tensor(
        [int(row["label"]) for row in validation_rows], dtype=torch.long
    )
    model, fit = train_readout(
        train_features,
        train_labels,
        validation_features,
        validation_labels,
        seed=17,
        learning_rate=0.01,
        weight_decay=0.01,
        batch_size=256,
        max_epochs=60,
        patience=6,
        min_delta=1e-5,
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
            record = {
                "sample_id": row["sample_id"],
                "content_group_id": row["content_group_id"],
                "target": int(validation_labels[index]),
                "prediction": int(predictions[index]),
                "probabilities": probabilities[index].tolist(),
            }
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
    target_counts = Counter(int(row["label"]) for row in validation_rows)
    metrics: dict[str, Any] = {
        **metric_summary(logits, validation_labels),
        "validation_rows": len(validation_rows),
        "validation_unique_content_groups": len(
            {row["content_group_id"] for row in validation_rows}
        ),
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
        "experiment_id": config["experiment_id"],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_file_sha256": model_file_hashes,
        "model_weights_sha256": model_file_hashes["model.safetensors"],
        "split_manifest_sha256": sha256_file(args.data_dir / "split-manifest.json"),
        "base_encoder_parameters": 22_713_216,
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "total_parameters": 22_713_216 + sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "train_unique_groups": len({row["content_group_id"] for row in train_rows}),
        "validation_unique_groups": len(
            {row["content_group_id"] for row in validation_rows}
        ),
        "official_test_loaded": False,
        "device": "cpu",
        "dtype": "float32",
        "seed": 17,
        "selection_metric": "minimum validation NLL",
        "best_epoch": fit["best_epoch"],
        "epochs_completed": fit["epochs_completed"],
        "started_at_utc": started,
        "ended_at_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": time.perf_counter() - time_start,
        "best_checkpoint_sha256": sha256_file(checkpoint_path),
        "history_sha256": sha256_file(history_path),
        "validation_predictions_sha256": sha256_file(prediction_path),
        "validation_metrics_sha256": sha256_file(metrics_path),
        "validation_metrics": metric_summary(logits, validation_labels),
        "dataset_revision": split_manifest["source_revision"],
        "dataset_split_sha256": sha256_file(args.data_dir / "split-manifest.json"),
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
