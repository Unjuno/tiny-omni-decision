"""Fit the frozen-feature FSD50K four-question binary Decision readout."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from scripts.build_fsd50k_annotation_decision_manifest import (
    TARGET_LABELS,
    canonical_sha256,
)
from scripts.validate_fsd50k_annotation_acquisition import load_frozen_manifest
from tiny_omni_decision.native_feature_decision import (
    EnumOptionField,
    NativeFeatureDecisionScorer,
    TypedOptionSchema,
)
from tiny_omni_decision.observations import NativeOption
from tiny_omni_decision.textfree_audio import fit_scalar_temperature

SPLITS = ("train", "calibration", "validation")
FEATURE_WIDTH = 384
HIDDEN_WIDTH = 64
SEED = 17
EPOCHS = 20
BATCH_CLIPS = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EXPECTED_READOUT_PARAMETERS = 25_089
ENCODER_PARAMETERS = 1_961_239


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def load_feature_data(
    preflight_dir: Path,
    acquisition_verification: Path,
    feature_dir: Path,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    preflight, splits = load_frozen_manifest(preflight_dir)
    if preflight.get("official_eval_labels_read") is not False:
        raise ValueError("official eval must remain unopened")
    verification = json.loads(acquisition_verification.read_text(encoding="utf-8"))
    if (
        verification.get("status") != "PASS"
        or verification.get("preflight_manifest_sha256") != preflight.get("manifest_sha256")
        or verification.get("official_eval_labels_read") is not False
        or verification.get("official_eval_media_acquired") is not False
    ):
        raise ValueError("matching complete-media verification is required")

    expected_ids = [row["sample_id"] for split in SPLITS for row in splits[split]]
    expected_id_hash = canonical_sha256(expected_ids)

    feature_path = feature_dir / "features.npy"
    rows_path = feature_dir / "feature-rows.jsonl"
    feature_receipt_path = feature_dir / "feature-run.json"
    receipt = json.loads(feature_receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("status") != "PASS"
        or receipt.get("preflight_manifest_sha256") != preflight.get("manifest_sha256")
        or receipt.get("acquisition_verification_sha256") != sha256_file(acquisition_verification)
        or receipt.get("sample_order_sha256") != expected_id_hash
        or receipt.get("features_sha256") != sha256_file(feature_path)
        or receipt.get("feature_rows_sha256") != sha256_file(rows_path)
        or receipt.get("encoder_revision") != "a425fdce92572e602a1d5634799bd9f1f2efa806"
        or receipt.get("encoder_weights_sha256")
        != "49b0e63577021994257b3de2ac5f1686d71554b93c8149f9d2568ad5760b904e"
        or receipt.get("feature_width") != FEATURE_WIDTH
        or receipt.get("feature_dtype") != "float32"
        or receipt.get("temporal_policy") != "whole_clip_native_audio_resampled_to_32khz"
    ):
        raise ValueError("feature receipt does not bind to the frozen media and manifests")
    features = np.load(feature_path, allow_pickle=False)
    rows = _jsonl(rows_path)
    if (
        features.dtype != np.float32
        or features.shape != (len(expected_ids), FEATURE_WIDTH)
        or len(rows) != len(expected_ids)
        or not np.isfinite(features).all()
    ):
        raise ValueError("feature matrix is incomplete, nonfinite, or has the wrong shape")

    expected = {row["sample_id"]: row for split in SPLITS for row in splits[split]}
    for index, (sample_id, row) in enumerate(zip(expected_ids, rows, strict=True)):
        manifest_row = expected[sample_id]
        if (
            row.get("sample_id") != sample_id
            or row.get("feature_index") != index
            or row.get("split") != manifest_row["split"]
            or row.get("source_id") != manifest_row["source_id"]
            or row.get("group_id") != manifest_row["group_id"]
            or row.get("license") != manifest_row["license"]
            or row.get("targets") != manifest_row["targets"]
            or not isinstance(row.get("media_sha256"), str)
            or len(row["media_sha256"]) != 64
            or any(char not in "0123456789abcdef" for char in row["media_sha256"])
        ):
            raise ValueError(f"feature row does not match frozen manifest at index {index}")
    for field in ("sample_id", "source_id", "group_id"):
        owners: dict[str, set[str]] = {}
        for row in rows:
            owners.setdefault(str(row[field]), set()).add(row["split"])
        if any(len(value) != 1 for value in owners.values()):
            raise ValueError(f"feature {field} overlaps frozen splits")
    return features, rows, {
        "preflight_manifest_sha256": preflight["manifest_sha256"],
        "acquisition_verification_sha256": sha256_file(acquisition_verification),
        "feature_receipt_sha256": sha256_file(feature_receipt_path),
        "features_sha256": sha256_file(feature_path),
        "feature_rows_sha256": sha256_file(rows_path),
        "sample_order_sha256": expected_id_hash,
    }


def typed_binary_options() -> Tensor:
    schema = TypedOptionSchema(
        enums=(
            EnumOptionField("label", tuple(TARGET_LABELS)),
            EnumOptionField("response", ("no_annotation", "annotated")),
        )
    )
    values = []
    for label in TARGET_LABELS:
        options = [
            NativeOption(
                option_id=response,
                attributes={"label": label, "response": response},
            )
            for response in ("no_annotation", "annotated")
        ]
        values.append(schema.encode(options))
    return torch.stack(values)


def expand_clip_batch(features: Tensor, targets: Tensor, option_features: Tensor):
    """Expand each clip into four typed binary questions without text leakage."""
    if features.ndim != 2 or features.shape[1] != FEATURE_WIDTH:
        raise ValueError("clip features must have shape [batch, 384]")
    if targets.shape != (len(features), len(TARGET_LABELS)):
        raise ValueError("targets must contain one boolean per clip and broad label")
    if option_features.shape != (len(TARGET_LABELS), 2, 6):
        raise ValueError("typed option tensor must have shape [4, 2, 6]")
    batch = len(features)
    obs = features[:, None, :].expand(-1, len(TARGET_LABELS), -1).reshape(-1, FEATURE_WIDTH)
    opts = option_features[None].expand(batch, -1, -1, -1).reshape(-1, 2, 6)
    labels = targets.to(torch.long).reshape(-1)
    return obs, opts, torch.ones((len(obs), 2), dtype=torch.bool), labels


def metric_summary(logits: Tensor, targets: Tensor) -> dict[str, float]:
    logits = logits.detach().to(torch.float64)
    targets = targets.detach().to(torch.long)
    if logits.ndim != 2 or logits.shape[1] != 2 or len(logits) != len(targets):
        raise ValueError("binary metric inputs have invalid shapes")
    if not torch.isfinite(logits).all() or targets.min() < 0 or targets.max() > 1:
        raise ValueError("binary metric inputs must be finite and targets in {0,1}")
    probabilities = torch.softmax(logits, dim=-1)
    predictions = probabilities.argmax(dim=-1)
    confidence = probabilities.max(dim=-1).values
    correct = predictions.eq(targets)
    true_positive = (predictions.eq(1) & targets.eq(1)).sum().to(torch.float64)
    false_positive = (predictions.eq(1) & targets.eq(0)).sum().to(torch.float64)
    true_negative = (predictions.eq(0) & targets.eq(0)).sum().to(torch.float64)
    false_negative = (predictions.eq(0) & targets.eq(1)).sum().to(torch.float64)
    positive_precision = true_positive / (true_positive + false_positive).clamp_min(1.0)
    positive_recall = true_positive / (true_positive + false_negative).clamp_min(1.0)
    specificity = true_negative / (true_negative + false_positive).clamp_min(1.0)
    onehot = F.one_hot(targets, num_classes=2).to(torch.float64)
    edges = torch.linspace(0.0, 1.0, 16, dtype=torch.float64)
    ece = torch.zeros((), dtype=torch.float64)
    for index in range(15):
        mask = (confidence > edges[index]) & (confidence <= edges[index + 1])
        if mask.any():
            ece += mask.to(torch.float64).mean() * (
                correct[mask].to(torch.float64).mean() - confidence[mask].mean()
            ).abs()
    return {
        "count": int(len(targets)),
        "accuracy": float(correct.to(torch.float64).mean()),
        "nll": float(F.cross_entropy(logits, targets)),
        "brier": float((probabilities - onehot).square().sum(-1).mean()),
        "ece_15": float(ece),
        "mean_confidence": float(confidence.mean()),
        "positive_precision": float(positive_precision),
        "positive_recall": float(positive_recall),
        "balanced_accuracy": float((positive_recall + specificity) / 2.0),
        "positive_support": int(targets.eq(1).sum()),
    }


def macro_metrics(per_label: dict[str, dict[str, float]]) -> dict[str, float]:
    if set(per_label) != set(TARGET_LABELS):
        raise ValueError("macro metrics require all four frozen labels")
    return {
        key: float(np.mean([per_label[label][key] for label in TARGET_LABELS]))
        for key in (
            "accuracy",
            "nll",
            "brier",
            "ece_15",
            "mean_confidence",
            "positive_precision",
            "positive_recall",
            "balanced_accuracy",
        )
    }


def _label_targets(rows: list[dict[str, Any]], device: torch.device) -> Tensor:
    return torch.tensor(
        [[bool(row["targets"][label]) for label in TARGET_LABELS] for row in rows],
        dtype=torch.long,
        device=device,
    )


def _logits_for_split(model, features, targets, option_features, batch_clips):
    outputs = []
    for start in range(0, len(features), batch_clips):
        x = features[start : start + batch_clips]
        y = targets[start : start + batch_clips]
        obs, opts, mask, _ = expand_clip_batch(x, y, option_features)
        outputs.append(model(obs, opts, mask).reshape(len(x), len(TARGET_LABELS), 2))
    return torch.cat(outputs)


def run_training(
    features: np.ndarray,
    rows: list[dict[str, Any]],
    hashes: dict[str, str],
    output_dir: Path,
    *,
    device: str = "cuda",
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite FSD50K Decision output: {output_dir}")
    output_dir.mkdir(parents=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    target_device = torch.device(device)
    torch.cuda.manual_seed_all(SEED) if target_device.type == "cuda" else None
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    split_indices = {
        split: [i for i, row in enumerate(rows) if row["split"] == split]
        for split in SPLITS
    }
    if not all(split_indices.values()):
        raise ValueError("train, calibration, and validation must all contain rows")
    raw = torch.from_numpy(features.copy()).to(target_device)
    train_idx = torch.tensor(split_indices["train"], device=target_device)
    mean = raw[train_idx].mean(dim=0)
    scale = raw[train_idx].std(dim=0, unbiased=False).clamp_min(1e-6)
    normalized = (raw - mean) / scale
    targets_by_split = {
        split: _label_targets([rows[i] for i in indexes], target_device)
        for split, indexes in split_indices.items()
    }
    x_by_split = {
        split: normalized[torch.tensor(indexes, device=target_device)]
        for split, indexes in split_indices.items()
    }
    option_features = typed_binary_options().to(target_device)

    model = NativeFeatureDecisionScorer(FEATURE_WIDTH, 6, HIDDEN_WIDTH).to(target_device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != EXPECTED_READOUT_PARAMETERS:
        raise AssertionError(f"readout parameter count changed: {parameter_count}")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    config = {
        "format": "fsd50k_annotation_decision_training_v1",
        "seed": SEED,
        "encoder_frozen": True,
        "feature_width": FEATURE_WIDTH,
        "typed_questions": list(TARGET_LABELS),
        "options": ["no_annotation", "annotated"],
        "readout": (
            "NativeFeatureDecisionScorer(feature_width=384, option_width=6, "
            "hidden_width=64)"
        ),
        "readout_parameters": parameter_count,
        "total_encoder_plus_readout_parameters": ENCODER_PARAMETERS + parameter_count,
        "epochs": EPOCHS,
        "batch_clips": BATCH_CLIPS,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "selector": (
            "minimum validation macro NLL after scalar temperature fitted on calibration only"
        ),
        "temperature_fit": (
            "one shared positive scalar over all four binary questions on calibration only"
        ),
        "ece_bins": 15,
        "input_hashes": hashes,
        "device": str(target_device),
        "official_eval_used": False,
    }
    config_path = output_dir / "effective-config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    config_hash = sha256_file(config_path)
    history_path = output_dir / "training-history.jsonl"
    best_score = math.inf
    best_epoch = None
    started = time.perf_counter()
    best_validation_logits = None
    best_temperature = None
    best_validation_per_label = None
    best_validation_macro = None
    best_train_per_label = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = torch.randperm(len(x_by_split["train"]), device=target_device)
        batch_losses = []
        for start in range(0, len(order), BATCH_CLIPS):
            clip_index = order[start : start + BATCH_CLIPS]
            obs, opts, mask, target = expand_clip_batch(
                x_by_split["train"][clip_index],
                targets_by_split["train"][clip_index],
                option_features,
            )
            logits = model(obs, opts, mask)
            loss = F.cross_entropy(logits, target)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite FSD50K Decision loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(
                parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            ):
                raise ValueError("nonfinite FSD50K Decision gradient")
            optimizer.step()
            batch_losses.append(float(loss.detach()))

        model.eval()
        with torch.inference_mode():
            split_logits = {
                split: _logits_for_split(
                    model,
                    x_by_split[split],
                    targets_by_split[split],
                    option_features,
                    BATCH_CLIPS,
                )
                for split in SPLITS
            }
        calibration_logits = split_logits["calibration"].reshape(-1, 2)
        calibration_targets = targets_by_split["calibration"].reshape(-1)
        temperature = fit_scalar_temperature(calibration_logits, calibration_targets)
        train_logits_calibrated = split_logits["train"] / temperature
        validation_logits = split_logits["validation"] / temperature
        per_label = {
            label: metric_summary(validation_logits[:, i], targets_by_split["validation"][:, i])
            for i, label in enumerate(TARGET_LABELS)
        }
        validation_macro = macro_metrics(per_label)
        train_metrics = {
            label: metric_summary(split_logits["train"][:, i], targets_by_split["train"][:, i])
            for i, label in enumerate(TARGET_LABELS)
        }
        train_calibrated_metrics = {
            label: metric_summary(
                train_logits_calibrated[:, i], targets_by_split["train"][:, i]
            )
            for i, label in enumerate(TARGET_LABELS)
        }
        calibration_per_label = {
            label: metric_summary(
                split_logits["calibration"][:, i], targets_by_split["calibration"][:, i]
            )
            for i, label in enumerate(TARGET_LABELS)
        }
        calibration_calibrated_per_label = {
            label: metric_summary(
                (split_logits["calibration"] / temperature)[:, i],
                targets_by_split["calibration"][:, i],
            )
            for i, label in enumerate(TARGET_LABELS)
        }
        record = {
            "epoch": epoch,
            "optimizer_updates": epoch * math.ceil(len(x_by_split["train"]) / BATCH_CLIPS),
            "mean_batch_train_ce": float(np.mean(batch_losses)),
            "calibration_temperature": float(temperature),
            "train_per_label": train_metrics,
            "train_per_label_calibrated": train_calibrated_metrics,
            "calibration_per_label_raw": calibration_per_label,
            "calibration_per_label_calibrated": calibration_calibrated_per_label,
            "validation_per_label": per_label,
            "validation_macro": validation_macro,
            "selected": False,
        }
        with history_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        if validation_macro["nll"] < best_score:
            best_score = validation_macro["nll"]
            best_epoch = epoch
            best_temperature = float(temperature)
            best_validation_logits = validation_logits.detach().cpu().clone()
            best_validation_per_label = per_label
            best_validation_macro = validation_macro
            best_train_per_label = train_calibrated_metrics
            record["selected"] = True
            temporary = output_dir / "best-checkpoint.pt.tmp"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "feature_mean": mean.detach().cpu(),
                    "feature_scale": scale.detach().cpu(),
                    "temperature": best_temperature,
                    "selected_epoch": best_epoch,
                    "parameter_count": parameter_count,
                    "config_sha256": config_hash,
                },
                temporary,
            )
            temporary.replace(output_dir / "best-checkpoint.pt")
            # Rewrite the just-appended row to preserve the selection marker atomically.
            history = _jsonl(history_path)
            for history_row in history:
                history_row["selected"] = False
            history[-1]["selected"] = True
            tmp_history = history_path.with_suffix(".jsonl.tmp")
            tmp_history.write_text(
                "".join(json.dumps(item, sort_keys=True) + "\n" for item in history),
                encoding="utf-8",
            )
            tmp_history.replace(history_path)

    model.eval()
    with torch.inference_mode():
        final_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save(
        {
            "state_dict": final_state,
            "feature_mean": mean.detach().cpu(),
            "feature_scale": scale.detach().cpu(),
            "optimizer_updates": EPOCHS
            * math.ceil(len(x_by_split["train"]) / BATCH_CLIPS),
            "config_sha256": config_hash,
        },
        output_dir / "final-checkpoint.pt",
    )
    elapsed = time.perf_counter() - started
    if (
        best_epoch is None
        or best_validation_logits is None
        or best_validation_per_label is None
        or best_validation_macro is None
        or best_train_per_label is None
    ):
        raise AssertionError("validation selector did not select a checkpoint")

    best_checkpoint = torch.load(
        output_dir / "best-checkpoint.pt", map_location=target_device, weights_only=True
    )
    model.load_state_dict(best_checkpoint["state_dict"], strict=True)
    model.eval()
    with torch.inference_mode():
        reloaded_logits = _logits_for_split(
            model,
            x_by_split["validation"],
            targets_by_split["validation"],
            option_features,
            BATCH_CLIPS,
        ) / float(best_checkpoint["temperature"])
    reloaded_per_label = {
        label: metric_summary(reloaded_logits[:, i], targets_by_split["validation"][:, i])
        for i, label in enumerate(TARGET_LABELS)
    }
    if reloaded_per_label != best_validation_per_label:
        raise ValueError("best checkpoint reload does not reproduce selected validation metrics")

    best_predictions = []
    validation_rows = [rows[i] for i in split_indices["validation"]]
    probabilities = torch.softmax(reloaded_logits.cpu(), dim=-1).numpy()
    for sample_index, row in enumerate(validation_rows):
        for label_index, label in enumerate(TARGET_LABELS):
            best_predictions.append(
                {
                    "sample_id": row["sample_id"],
                    "group_id": row["group_id"],
                    "label": label,
                    "target": int(targets_by_split["validation"][sample_index, label_index].item()),
                    "option_probabilities": {
                        "no_annotation": float(probabilities[sample_index, label_index, 0]),
                        "annotated": float(probabilities[sample_index, label_index, 1]),
                    },
                }
            )
    prediction_path = output_dir / "selected-validation-predictions.jsonl"
    prediction_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in best_predictions),
        encoding="utf-8",
    )
    result = {
        "status": "PASS",
        "selected_epoch": best_epoch,
        "selected_validation_macro_nll": best_score,
        "selected_temperature": best_temperature,
        "readout_parameters": parameter_count,
        "total_encoder_plus_readout_parameters": ENCODER_PARAMETERS + parameter_count,
        "train_clips": len(split_indices["train"]),
        "calibration_clips": len(split_indices["calibration"]),
        "validation_clips": len(split_indices["validation"]),
        "unique_train_assets": len({rows[i]["source_id"] for i in split_indices["train"]}),
        "unique_train_uploaders": len({rows[i]["group_id"] for i in split_indices["train"]}),
        "unique_validation_assets": len(
            {rows[i]["source_id"] for i in split_indices["validation"]}
        ),
        "unique_validation_uploaders": len(
            {rows[i]["group_id"] for i in split_indices["validation"]}
        ),
        "validation_per_label": best_validation_per_label,
        "validation_macro": best_validation_macro,
        "train_per_label_at_selected_epoch_calibrated": best_train_per_label,
        "best_checkpoint_reload_verified": True,
        "training_wall_seconds": elapsed,
        "best_checkpoint_sha256": sha256_file(output_dir / "best-checkpoint.pt"),
        "final_checkpoint_sha256": sha256_file(output_dir / "final-checkpoint.pt"),
        "official_eval_used": False,
        "effective_config_sha256": config_hash,
        "training_history_sha256": sha256_file(history_path),
        "selected_predictions_sha256": sha256_file(prediction_path),
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--acquisition-verification", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    features, rows, hashes = load_feature_data(
        args.preflight, args.acquisition_verification, args.features
    )
    print(json.dumps(run_training(features, rows, hashes, args.output, device=args.device)))


if __name__ == "__main__":
    main()
