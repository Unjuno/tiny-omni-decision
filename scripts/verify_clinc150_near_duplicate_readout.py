"""Independently reload a saved CLINC150 readout and verify predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from train_clinc150_near_duplicate_readout import (
    extract_features,
    file_hash,
    load_and_validate_data,
    metrics,
    read_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    verification_path = args.run_dir / "run-verification.json"
    if verification_path.exists():
        raise FileExistsError(f"Refusing to overwrite verification: {verification_path}")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    metadata_path = args.run_dir / "run-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if file_hash(args.config) != metadata["config_sha256"]:
        raise ValueError("Run config hash mismatch")
    train, validation, manifest_hash = load_and_validate_data(args.data_dir, config)
    if manifest_hash != metadata["data_manifest_sha256"]:
        raise ValueError("Run split manifest hash mismatch")

    features, token_audit = extract_features(
        validation, args.model_dir, config["training"]["max_length"]
    )
    checkpoint_path = args.run_dir / "best-readout.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    head = nn.Linear(features.shape[-1], config["readout"]["output_classes"])
    head.load_state_dict(checkpoint["state_dict"])
    head.eval()
    with torch.inference_mode():
        standardized = (features - checkpoint["train_mean"]) / checkpoint["train_std"]
        logits = head(standardized)
        probabilities = torch.softmax(logits, dim=-1)
        predictions = logits.argmax(dim=-1)
    saved_predictions = read_jsonl(args.run_dir / "validation-predictions.jsonl")
    if len(saved_predictions) != len(validation):
        raise ValueError("Saved prediction count mismatch")
    max_probability_error = 0.0
    id_order_match = True
    target_match = True
    prediction_match = True
    for index, (row, saved) in enumerate(zip(validation, saved_predictions, strict=True)):
        id_order_match &= row["sample_id"] == saved["sample_id"]
        target_match &= int(row["label"]) == int(saved["label"])
        prediction_match &= int(predictions[index]) == int(saved["prediction"])
        saved_probs = torch.tensor(saved["probabilities"], dtype=probabilities.dtype)
        max_probability_error = max(
            max_probability_error,
            torch.max(torch.abs(probabilities[index] - saved_probs)).item(),
        )
    if not id_order_match or not target_match or not prediction_match:
        raise AssertionError("Saved prediction rows do not match checkpoint reload")
    if max_probability_error > 1e-6:
        raise AssertionError(f"Reloaded probability error too large: {max_probability_error}")

    labels = torch.tensor([int(row["label"]) for row in validation])
    recomputed_metrics = metrics(logits, labels)
    expected_metrics = metadata["selected_validation_metrics"]
    metric_errors = {
        key: abs(recomputed_metrics[key] - expected_metrics[key])
        for key in recomputed_metrics
    }
    if max(metric_errors.values()) > 1e-6:
        raise AssertionError(f"Reloaded metrics do not match run metadata: {metric_errors}")
    verification: dict[str, Any] = {
        "experiment_id": metadata["experiment_id"],
        "checkpoint_sha256": file_hash(checkpoint_path),
        "metadata_sha256": file_hash(metadata_path),
        "validation_prediction_sha256": file_hash(
            args.run_dir / "validation-predictions.jsonl"
        ),
        "validation_rows": len(validation),
        "train_rows": len(train),
        "data_manifest_sha256": manifest_hash,
        "model_revision": config["model"]["revision"],
        "tokenization": token_audit,
        "reload_max_probability_abs_error": max_probability_error,
        "sample_id_order_match": id_order_match,
        "targets_match": target_match,
        "predictions_match": prediction_match,
        "metrics": recomputed_metrics,
        "metric_abs_errors": metric_errors,
        "best_epoch_matches_metadata": checkpoint["best_epoch"]
        == metadata["selected_epoch"],
        "official_test_loaded": False,
        "sealed_audit_loaded": False,
    }
    if not verification["best_epoch_matches_metadata"]:
        raise AssertionError("Selected checkpoint epoch mismatch")
    verification_path.write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    verification["run_verification_sha256"] = file_hash(verification_path)
    print(json.dumps(verification, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
