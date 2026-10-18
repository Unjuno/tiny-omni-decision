"""Independently verify and finalize an existing EuroSAT spatial readout run on CPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tiny_omni_decision.clevr4_readout import classification_quality
from tiny_omni_decision.eurosat import (
    eurosat_class_from_filename,
    require_probability_replay,
)

SEED = 17
EPOCHS = 50
BATCH_SIZE = 256
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
REPLAY_ATOL = 2e-6
EXECUTED_TRAIN_RUNNER_SHA256 = "e8119d32407ae98b6dfa0a18ecb19ae1594eec22628e28e1ea6649f2eaab6e65"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def logits_for(
    model, features: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    outputs = []
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(features), BATCH_SIZE):
            outputs.append(model((features[offset : offset + BATCH_SIZE] - mean) / std))
    return torch.cat(outputs)


def summarize(logits: torch.Tensor, targets: torch.Tensor) -> dict:
    probabilities = torch.softmax(logits.float(), dim=-1)
    result = classification_quality(probabilities, targets)
    predictions = logits.argmax(-1)
    recalls = []
    by_class = {}
    for index, name in enumerate(CLASS_ORDER):
        mask = targets.eq(index)
        support = int(mask.sum())
        accuracy = float(predictions[mask].eq(targets[mask]).float().mean()) if support else None
        by_class[name] = {"support": support, "accuracy": accuracy}
        if accuracy is not None:
            recalls.append(accuracy)
    result["macro_accuracy"] = float(np.mean(recalls))
    result["balanced_accuracy"] = result["macro_accuracy"]
    result["per_class"] = by_class
    return result


def _assert_metric_close(replayed: dict, saved: dict, *, atol: float = 2e-6) -> None:
    for key in ("accuracy", "nll", "brier", "ece", "mean_confidence"):
        if abs(float(replayed[key]) - float(saved[key])) > atol:
            raise ValueError(f"reloaded {key} differs from saved selected metric")
    if replayed["per_class"] != saved["per_class"]:
        raise ValueError("reloaded per-class metrics differ from saved selector metrics")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    result_path = args.run / "result.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    config = json.loads((args.run / "config.json").read_text(encoding="utf-8"))
    feature_result_path = args.features / "result.json"
    feature_result = json.loads(feature_result_path.read_text(encoding="utf-8"))
    if sha256_file(feature_result_path) != config["feature_result_sha256"]:
        raise ValueError("feature artifact result hash differs from run config")
    if (
        config["spatial_test_read"] is not False
        or feature_result["spatial_test_media_decoded"] is not False
    ):
        raise ValueError("run metadata says spatial test data was read")
    if config["seed"] != SEED or config["epochs"] != EPOCHS or config["batch_size"] != BATCH_SIZE:
        raise ValueError("unexpected training schedule")
    if tuple(config["class_order"]) != CLASS_ORDER:
        raise ValueError("unexpected class option order")

    train_records = load_jsonl(args.features / "train-records.jsonl")
    val_records = load_jsonl(args.features / "validation-records.jsonl")
    for split, rows in (("train", train_records), ("validation", val_records)):
        record_path = args.features / f"{split}-records.jsonl"
        if sha256_file(record_path) != config[f"{split}_records_sha256"]:
            raise ValueError(f"{split} records hash mismatch")
        for row in rows:
            if row["label"] != eurosat_class_from_filename(row["filename"]):
                raise ValueError(f"{split} label/filename mismatch")
    if set(row["id"] for row in train_records) & set(row["id"] for row in val_records):
        raise ValueError("train/validation IDs overlap")

    feature_matrices = {}
    for split, records in (("train", train_records), ("validation", val_records)):
        path = args.features / f"{split}-features.npy"
        if sha256_file(path) != feature_result["feature_file_sha256"][split]:
            raise ValueError(f"{split} feature file hash mismatch")
        matrix = np.load(path, mmap_mode="r")
        if matrix.shape != (len(records), 768) or not np.isfinite(matrix).all():
            raise ValueError(f"invalid {split} feature matrix")
        feature_matrices[split] = torch.from_numpy(np.asarray(matrix).copy())

    history_path = args.run / "training-history.jsonl"
    history = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines()]
    if len(history) != EPOCHS or [row["epoch"] for row in history] != list(range(1, EPOCHS + 1)):
        raise ValueError("training history is incomplete or out of order")
    selected = min(history, key=lambda row: (row["validation_metrics"]["nll"], row["epoch"]))
    if config["checkpoint_selector"] != "earliest minimum validation NLL":
        raise ValueError("unexpected checkpoint selector")

    checkpoint_path = args.run / "best-readout.pt"
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    model = nn.Linear(768, len(CLASS_ORDER))
    model.load_state_dict(checkpoint["model"])
    mean, std = checkpoint["mean"], checkpoint["std"]
    if mean.shape != (1, 768) or std.shape != (1, 768) or (std <= 0).any():
        raise ValueError("invalid train-only normalization state")
    val_targets = torch.tensor([CLASS_ORDER.index(row["label"]) for row in val_records])
    reloaded_logits = logits_for(model, feature_matrices["validation"], mean, std)
    reloaded_probs = torch.softmax(reloaded_logits.float(), dim=-1).numpy()
    prediction_path = args.run / "best-validation-predictions.jsonl"
    predictions = load_jsonl(prediction_path)
    if len(predictions) != len(val_records):
        raise ValueError("selected validation prediction count mismatch")
    for expected, observed in zip(val_records, predictions, strict=True):
        if observed["id"] != expected["id"] or observed["target"] != expected["label"]:
            raise ValueError("saved validation prediction ID/target mismatch")
        if tuple(observed["options"]) != CLASS_ORDER:
            raise ValueError("saved validation option order mismatch")
    saved_probs = np.asarray([row["probabilities"] for row in predictions], dtype=np.float64)
    if saved_probs.shape != (len(val_records), len(CLASS_ORDER)):
        raise ValueError("saved validation probability shape mismatch")
    if not np.isfinite(saved_probs).all() or not np.allclose(saved_probs.sum(-1), 1.0, atol=1e-5):
        raise ValueError("saved probabilities are non-finite or unnormalized")
    replay_delta = require_probability_replay(saved_probs, reloaded_probs, atol=REPLAY_ATOL)
    for observed, probs in zip(predictions, reloaded_probs, strict=True):
        chosen = CLASS_ORDER[int(np.argmax(probs))]
        if observed["prediction"] != chosen:
            raise ValueError("saved validation prediction differs from reloaded argmax")
    selected_replay_metrics = summarize(reloaded_logits, val_targets)
    _assert_metric_close(selected_replay_metrics, selected["validation_metrics"])

    train_targets = torch.tensor([CLASS_ORDER.index(row["label"]) for row in train_records])
    train_logits = logits_for(model, feature_matrices["train"], mean, std)
    selected_train_metrics = summarize(train_logits, train_targets)
    _assert_metric_close(selected_train_metrics, selected["train_metrics"])

    final_checkpoint_path = args.run / "final-readout.pt"
    if not final_checkpoint_path.is_file():
        raise ValueError("final checkpoint is missing")
    final_checkpoint = torch.load(final_checkpoint_path, weights_only=True, map_location="cpu")
    final_model = nn.Linear(768, len(CLASS_ORDER))
    final_model.load_state_dict(final_checkpoint["model"])
    final_logits = logits_for(
        final_model,
        feature_matrices["validation"],
        final_checkpoint["mean"],
        final_checkpoint["std"],
    )
    final_metrics = summarize(final_logits, val_targets)
    final_train_logits = logits_for(
        final_model, feature_matrices["train"], final_checkpoint["mean"], final_checkpoint["std"]
    )
    final_train_metrics = summarize(final_train_logits, train_targets)
    if final_checkpoint_path.stat().st_size == checkpoint_path.stat().st_size:
        if sha256_file(final_checkpoint_path) == sha256_file(checkpoint_path):
            raise ValueError("best and final checkpoint unexpectedly have identical hashes")

    counts = torch.bincount(train_targets, minlength=len(CLASS_ORDER)).float()
    prior_logits = counts.clamp_min(1).log().expand(len(val_targets), -1)
    prior_metrics = summarize(prior_logits, val_targets)
    duration = float(history[-1]["elapsed_seconds"])
    result = {
        **config,
        "result_status": "TRAINING_50_EPOCHS_COMPLETED_BEST_RELOAD_VERIFIED_POSTHOC",
        "run_process_exit_note": (
            "training completed all epochs; initial process stopped at its strict 1e-6 "
            "GPU-to-CPU replay assertion before writing result.json"
        ),
        "finalizer_completed_utc": datetime.now(UTC).isoformat(),
        "training_loop_elapsed_seconds_including_epoch_validation": duration,
        "full_process_wall_seconds": None,
        "full_process_wall_note": (
            "not persisted before the training process replay-tolerance assertion"
        ),
        "optimizer_updates": EPOCHS * ((len(train_records) + BATCH_SIZE - 1) // BATCH_SIZE),
        "train_example_encounters": EPOCHS * len(train_records),
        "unique_train_examples": len(train_records),
        "unique_validation_examples": len(val_records),
        "unique_underlying_train_assets": len({row["media_sha256"] for row in train_records}),
        "unique_underlying_validation_assets": len({row["media_sha256"] for row in val_records}),
        "selected_epoch": int(selected["epoch"]),
        "final_epoch": EPOCHS,
        "selected_train_metrics": selected_train_metrics,
        "selected_validation_metrics": selected_replay_metrics,
        "final_train_metrics": final_train_metrics,
        "final_validation_metrics": final_metrics,
        "validation_train_prior_metrics": prior_metrics,
        "best_reload_max_probability_delta": replay_delta,
        "best_reload_probability_atol": REPLAY_ATOL,
        "peak_vram_allocated_bytes": None,
        "peak_vram_note": ("not serialized before the training process replay-tolerance assertion"),
        "best_readout_sha256": sha256_file(checkpoint_path),
        "final_readout_sha256": sha256_file(final_checkpoint_path),
        "training_runner_sha256_at_execution": EXECUTED_TRAIN_RUNNER_SHA256,
        "history_sha256": sha256_file(history_path),
        "validation_predictions_sha256": sha256_file(prediction_path),
        "selector_epoch_from_full_history": int(selected["epoch"]),
        "spatial_test_data_used_for_fit_or_selection": False,
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
