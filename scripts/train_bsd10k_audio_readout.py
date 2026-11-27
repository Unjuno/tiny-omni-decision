"""Fit one frozen-feature BSD10k audio classifier using train/validation only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import platform
import random
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

SEED = 17
LABELS = (
    "fx-a",
    "fx-el",
    "fx-ex",
    "fx-h",
    "fx-m",
    "fx-n",
    "fx-o",
    "fx-v",
    "is-e",
    "is-k",
    "is-p",
    "is-s",
    "is-w",
    "m-m",
    "m-si",
    "m-sp",
    "sp-c",
    "sp-p",
    "sp-s",
    "ss-i",
    "ss-n",
    "ss-s",
    "ss-u",
)
ENCODER_PARAMETERS = 1_961_239
ENCODER_SHA256 = "49b0e63577021994257b3de2ac5f1686d71554b93c8149f9d2568ad5760b904e"
ENCODER_COMMIT = "a425fdce92572e602a1d5634799bd9f1f2efa806"
MAX_EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 12
MIN_DELTA = 1e-4


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_labels(metadata_zip: Path, split_manifest_path: Path, split: str) -> dict[str, dict]:
    from scripts.audit_bsd10k_metadata import ALLOWED_LICENSE_PATHS, canonical_license

    manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    expected = {str(row["sound_id"]): row for row in manifest["examples"] if row["split"] == split}
    with zipfile.ZipFile(metadata_zip) as archive:
        names = [
            name for name in archive.namelist() if name.casefold().endswith("bsd10k_metadata.csv")
        ]
        if len(names) != 1:
            raise ValueError("metadata archive must contain exactly one BSD10k metadata CSV")
        with archive.open(names[0]) as raw:
            rows = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
            output = {}
            for row in rows:
                sound_id = row["sound_id"].strip()
                if sound_id not in expected:
                    continue
                frozen = expected[sound_id]
                label = row["class"].strip()
                group_hash = hashlib.sha256(row["uploader"].strip().casefold().encode()).hexdigest()
                confidence = int(row["confidence"])
                license_name = canonical_license(row["license"])
                if (
                    label != frozen["class"]
                    or group_hash != frozen["uploader_group_sha256"]
                    or license_name != frozen["license"]
                    or license_name not in ALLOWED_LICENSE_PATHS.values()
                    or label not in LABELS
                    or confidence not in range(1, 6)
                ):
                    raise ValueError(f"eligible metadata differs from frozen split row: {sound_id}")
                output[sound_id] = {
                    "class_id": label,
                    "confidence": confidence,
                    "uploader_group_sha256": group_hash,
                    "class_top": row["class_top"].strip(),
                }
    if set(output) != set(expected):
        raise ValueError(f"metadata rows do not cover the complete {split} split")
    return output


def _metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float | int]:
    probabilities = torch.softmax(logits.float(), dim=-1)
    count = len(targets)
    confidence, predictions = probabilities.max(dim=-1)
    correct = predictions.eq(targets)
    one_hot = F.one_hot(targets, num_classes=len(LABELS)).float()
    ece = 0.0
    for bin_index in range(15):
        mask = (confidence >= bin_index / 15) & (
            confidence < (bin_index + 1) / 15 if bin_index < 14 else confidence <= 1
        )
        if mask.any():
            ece += mask.float().mean().item() * abs(
                correct[mask].float().mean().item() - confidence[mask].mean().item()
            )
    return {
        "count": count,
        "accuracy": correct.float().mean().item(),
        "nll": F.cross_entropy(logits.float(), targets).item(),
        "brier": (probabilities - one_hot).square().sum(dim=-1).mean().item(),
        "ece_15_bins": ece,
        "mean_confidence": confidence.mean().item(),
    }


def _per_class_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, dict]:
    result = {}
    for index, label in enumerate(LABELS):
        mask = targets == index
        if mask.any():
            result[label] = _metrics(logits[mask], targets[mask])
    return result


def _macro(metrics_by_class: dict[str, dict]) -> dict[str, float]:
    if set(metrics_by_class) != set(LABELS):
        raise ValueError("validation split must contain all 23 fixed classes")
    return {
        metric: sum(float(row[metric]) for row in metrics_by_class.values()) / len(LABELS)
        for metric in ("accuracy", "nll", "brier", "ece_15_bins")
    }


def _group_bootstrap_macro_accuracy(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    group_hashes: list[str],
    *,
    seed: int = SEED,
    replicates: int = 2_000,
) -> dict[str, float | int]:
    groups = sorted(set(group_hashes))
    if len(groups) < 2 or len(group_hashes) != len(targets):
        raise ValueError("invalid uploader groups for clustered bootstrap")
    group_to_index = {group: index for index, group in enumerate(groups)}
    counts = np.zeros((len(groups), len(LABELS)), dtype=np.int32)
    correct = np.zeros_like(counts)
    for target, prediction, group in zip(
        targets.tolist(), predictions.tolist(), group_hashes, strict=True
    ):
        group_index = group_to_index[group]
        counts[group_index, target] += 1
        correct[group_index, target] += int(target == prediction)
    rng = np.random.default_rng(seed)
    scores = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sampled = rng.integers(0, len(groups), size=len(groups))
        class_counts = counts[sampled].sum(axis=0)
        class_correct = correct[sampled].sum(axis=0)
        present = class_counts > 0
        scores[replicate] = np.mean(class_correct[present] / class_counts[present])
    low, high = np.quantile(scores, [0.025, 0.975]).tolist()
    return {
        "cluster": "normalized Freesound uploader",
        "independent_groups": len(groups),
        "replicates": replicates,
        "seed": seed,
        "interval_method": "percentile bootstrap resampling uploader groups with replacement",
        "macro_accuracy_ci95": [float(low), float(high)],
    }


def run(
    metadata_zip: Path,
    split_manifest_path: Path,
    train_feature_dir: Path,
    validation_feature_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    from tiny_omni_decision.textfree_audio import (
        TextFreeAudioReadout,
        fit_scalar_temperature,
        speaker_group_split,
    )

    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite readout output: {output_dir}")
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    metadata_sha = sha256_file(metadata_zip)
    if split_manifest.get("metadata_zip_sha256") != metadata_sha:
        raise ValueError("split manifest is not pinned to the supplied metadata archive")
    train_labels = _load_labels(metadata_zip, split_manifest_path, "train")
    validation_labels = _load_labels(metadata_zip, split_manifest_path, "validation")

    def load_features(directory: Path, split: str, labels: dict[str, dict]):
        report_path = directory / "run.json"
        index_path = directory / "sample-index.json"
        feature_path = directory / "features.npy"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        index = json.loads(index_path.read_text(encoding="utf-8"))
        if (
            report.get("status") != "PASS_FROZEN_FEATURES"
            or report.get("partial_smoke")
            or report.get("split") != split
            or report.get("model_parameters") != ENCODER_PARAMETERS
            or report.get("weights_sha256") != ENCODER_SHA256
            or report.get("source_commit") != ENCODER_COMMIT
            or report.get("split_manifest_sha256") != sha256_file(split_manifest_path)
            or report.get("feature_sha256") != sha256_file(feature_path)
            or index.get("split") != split
        ):
            raise ValueError(f"invalid frozen feature receipt for {split}")
        ids = [str(item) for item in index["sound_ids"]]
        if len(ids) != len(set(ids)) or set(ids) != set(labels):
            raise ValueError(f"feature IDs do not match the full {split} labels")
        values = np.load(feature_path, allow_pickle=False)
        if values.shape != (len(ids), 384) or not np.isfinite(values).all():
            raise ValueError(f"invalid {split} feature matrix")
        features = torch.from_numpy(values.copy()).float()
        targets = torch.tensor([LABELS.index(labels[sound_id]["class_id"]) for sound_id in ids])
        return ids, features, targets, report

    train_ids, x_train_all, y_train_all, train_feature_receipt = load_features(
        train_feature_dir, "train", train_labels
    )
    validation_ids, x_validation, y_validation, validation_feature_receipt = load_features(
        validation_feature_dir, "validation", validation_labels
    )
    train_groups = [train_labels[sound_id]["uploader_group_sha256"] for sound_id in train_ids]
    fit_indices, calibration_indices = speaker_group_split(
        train_groups, seed=SEED, calibration_fraction=0.2
    )
    fit_x = x_train_all[fit_indices]
    fit_y = y_train_all[fit_indices]
    calibration_x = x_train_all[calibration_indices]
    calibration_y = y_train_all[calibration_indices]
    fit_groups = {train_groups[index] for index in fit_indices}
    calibration_groups = {train_groups[index] for index in calibration_indices}
    validation_groups = {
        validation_labels[sound_id]["uploader_group_sha256"] for sound_id in validation_ids
    }
    if (
        fit_groups & calibration_groups
        or fit_groups & validation_groups
        or calibration_groups & validation_groups
    ):
        raise ValueError("fit/calibration/validation uploader groups overlap")
    if set(fit_y.tolist()) != set(range(len(LABELS))) or set(y_validation.tolist()) != set(
        range(len(LABELS))
    ):
        raise ValueError("fit and validation must each contain all 23 classes")

    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(4)
    model = TextFreeAudioReadout(384, list(LABELS))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    order_rng = random.Random(SEED)
    best_state = None
    best_macro_nll = math.inf
    best_epoch = 0
    stale_epochs = 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, MAX_EPOCHS + 1):
        order = list(range(len(fit_indices)))
        order_rng.shuffle(order)
        model.train()
        train_losses = []
        for start in range(0, len(order), BATCH_SIZE):
            indexes = torch.tensor(order[start : start + BATCH_SIZE])
            loss = F.cross_entropy(model(fit_x[indexes]), fit_y[indexes])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            validation_logits = model(x_validation)
        class_metrics = _per_class_metrics(validation_logits, y_validation)
        macro_metrics = _macro(class_metrics)
        history.append(
            {
                "epoch": epoch,
                "fit_train_ce": float(np.mean(train_losses)),
                "validation_macro_nll": macro_metrics["nll"],
                "validation_macro_accuracy": macro_metrics["accuracy"],
                "validation_macro_brier": macro_metrics["brier"],
                "validation_macro_ece_15_bins": macro_metrics["ece_15_bins"],
            }
        )
        if macro_metrics["nll"] < best_macro_nll - MIN_DELTA:
            best_macro_nll = macro_metrics["nll"]
            best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= EARLY_STOPPING_PATIENCE:
                break
    training_seconds = time.perf_counter() - started
    if best_state is None:
        raise RuntimeError("validation did not select a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        fit_logits = model(fit_x)
        calibration_logits = model(calibration_x)
        validation_logits = model(x_validation)
    temperature = fit_scalar_temperature(calibration_logits, calibration_y)
    calibrated_logits = validation_logits / temperature
    validation_class_metrics = _per_class_metrics(validation_logits, y_validation)
    calibrated_class_metrics = _per_class_metrics(calibrated_logits, y_validation)
    predictions = validation_logits.argmax(dim=-1)
    predictions_calibrated = calibrated_logits.argmax(dim=-1)

    output_dir.mkdir(parents=True)
    weights_path = output_dir / "best-readout.pt"
    torch.save(best_state, weights_path)
    reloaded = TextFreeAudioReadout(384, list(LABELS)).eval()
    reloaded.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    with torch.inference_mode():
        reloaded_logits = reloaded(x_validation)
    if not torch.equal(reloaded_logits, validation_logits):
        raise ValueError("save/reload changed the selected validation logits")

    prediction_path = output_dir / "validation-predictions.jsonl"
    with prediction_path.open("w", encoding="utf-8", newline="\n") as stream:
        all_options = [{"option_id": f"bsd10k:{label}", "class_id": label} for label in LABELS]
        for index, (sound_id, target, logits, calibrated) in enumerate(
            zip(
                validation_ids,
                y_validation.tolist(),
                validation_logits,
                calibrated_logits,
                strict=True,
            )
        ):
            raw_decision = reloaded.decide(x_validation[index], all_options)
            calibrated_probs = torch.softmax(calibrated.float(), dim=-1)
            stream.write(
                json.dumps(
                    {
                        "sound_id": sound_id,
                        "target_class_id": LABELS[target],
                        "prediction_class_id": LABELS[int(logits.argmax())],
                        "probabilities_uncalibrated": torch.softmax(
                            logits.float(), dim=-1
                        ).tolist(),
                        "probabilities_temperature_scaled": calibrated_probs.tolist(),
                        "decision_json_uncalibrated": raw_decision,
                        "decision_json_temperature_scaled": reloaded.decide(
                            x_validation[index], all_options, temperature=temperature
                        ),
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )

    confidence_metrics: dict[str, dict] = {}
    for score in range(1, 6):
        indices = [
            index
            for index, sound_id in enumerate(validation_ids)
            if validation_labels[sound_id]["confidence"] == score
        ]
        if indices:
            selected = torch.tensor(indices)
            confidence_metrics[str(score)] = {
                "count": len(indices),
                "uncalibrated": _metrics(validation_logits[selected], y_validation[selected]),
                "temperature_scaled": _metrics(calibrated_logits[selected], y_validation[selected]),
            }

    config = {
        "seed": SEED,
        "encoder": "EfficientAT DyMN-0.4 AudioSet",
        "encoder_commit": ENCODER_COMMIT,
        "encoder_sha256": ENCODER_SHA256,
        "encoder_parameters_frozen": ENCODER_PARAMETERS,
        "feature_aggregation": "unit-normalized mean of 2-second windows with 1-second hop",
        "readout": "linear 384-to-23 TextFreeAudioReadout",
        "calibration_fraction_of_train_uploader_groups": 0.2,
        "calibration": "single positive scalar temperature fitted on held-out train groups",
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "early_stopping": {
            "selector": "minimum validation macro NLL",
            "patience_epochs": EARLY_STOPPING_PATIENCE,
            "min_delta": MIN_DELTA,
        },
        "labels_and_confidence_used_for": "readout targets and stratified reporting only",
    }
    class_top = {
        label: next(
            validation_labels[sound_id]["class_top"]
            for sound_id in validation_ids
            if validation_labels[sound_id]["class_id"] == label
        )
        for label in LABELS
    }
    report = {
        "schema": "tiny-omni-decision/bsd10k-audio-path-a-readout-v1",
        "status": "complete_validation_selected_audio_development_probe",
        "claim_scope": (
            "23-class heterogeneous audio taxonomy; custom public development split only"
        ),
        "seed": SEED,
        "config": config,
        "config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "metadata_zip_sha256": sha256_file(metadata_zip),
        "split_manifest_sha256": sha256_file(split_manifest_path),
        "train_feature_report_sha256": sha256_file(train_feature_dir / "run.json"),
        "validation_feature_report_sha256": sha256_file(validation_feature_dir / "run.json"),
        "train_examples": len(train_ids),
        "train_unique_uploader_groups": len(
            {row["uploader_group_sha256"] for row in train_labels.values()}
        ),
        "fit_examples": len(fit_indices),
        "fit_unique_uploader_groups": len(fit_groups),
        "calibration_examples": len(calibration_indices),
        "calibration_unique_uploader_groups": len(calibration_groups),
        "validation_examples": len(validation_ids),
        "validation_unique_uploader_groups": len(validation_groups),
        "fit_calibration_group_overlap": 0,
        "train_validation_group_overlap": 0,
        "class_count": len(LABELS),
        "class_top_mapping": class_top,
        "selected_epoch": best_epoch,
        "epochs_run": len(history),
        "training_seconds_cpu": training_seconds,
        "training_history": history,
        "calibration_temperature": temperature,
        "calibration_examples_by_class": dict(
            Counter(train_labels[train_ids[index]]["class_id"] for index in calibration_indices)
        ),
        "trainable_parameters": sum(parameter.numel() for parameter in reloaded.parameters()),
        "readout_checkpoint_bytes": weights_path.stat().st_size,
        "all_persistent_parameters": ENCODER_PARAMETERS
        + sum(parameter.numel() for parameter in reloaded.parameters()),
        "validation_metrics_uncalibrated_by_class": validation_class_metrics,
        "validation_metrics_temperature_scaled_by_class": calibrated_class_metrics,
        "validation_macro_metrics_uncalibrated": _macro(validation_class_metrics),
        "validation_macro_metrics_temperature_scaled": _macro(calibrated_class_metrics),
        "validation_confidence_strata": confidence_metrics,
        "validation_uploader_cluster_bootstrap": _group_bootstrap_macro_accuracy(
            y_validation,
            predictions,
            [validation_labels[sound_id]["uploader_group_sha256"] for sound_id in validation_ids],
        ),
        "mean_validation_fit_accuracy": _metrics(fit_logits, fit_y)["accuracy"],
        "save_reload_exact_logits": True,
        "validation_probabilities_argmax_unchanged_by_temperature": bool(
            torch.equal(predictions, predictions_calibrated)
        ),
        "readout_sha256": sha256_file(weights_path),
        "validation_predictions_sha256": sha256_file(prediction_path),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "test_split_loaded": False,
        "dcase_evaluation_loaded": False,
        "sealed_audit_loaded": False,
        "validation_scope": "used for early stopping and checkpoint selection; not blind evidence",
        "limitations": [
            "Manual labels are single targets; the metadata does not establish multi-rater "
            "consensus.",
            "Uploader groups are highly skewed and some per-class validation support is small.",
            "EfficientAT AudioSet pretraining overlap with exact Freesound media is unknown.",
            "This audio-only taxonomy result says nothing about text, image, video, sensor, "
            "or fusion quality.",
        ],
    }
    (output_dir / "run-report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-zip", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--validation-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.metadata_zip.resolve(strict=True),
        args.split_manifest.resolve(strict=True),
        args.train_features.resolve(strict=True),
        args.validation_features.resolve(strict=True),
        args.output.resolve(),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
