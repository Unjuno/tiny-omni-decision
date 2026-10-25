"""Fit a fixed CPU-only image-to-option readout over frozen VideoPrism features."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import subprocess
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_SRC = ROOT / "src"
if str(REPOSITORY_SRC) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_SRC))
SEED = 17
DEFAULT_EPOCHS = 512
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
IMAGE_WIDTH = 768
ENCODER_PARAMETERS = 114_365_184
ENCODER_CHECKPOINT_SHA256 = "61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29"
OPTION_ORDER_FLOAT32_TOLERANCE = 2e-7


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _metrics(logits: Tensor, targets: Tensor) -> dict[str, float | int]:
    probabilities = torch.softmax(logits.float(), dim=-1)
    confidence, predictions = probabilities.max(dim=-1)
    correct = predictions.eq(targets)
    one_hot = F.one_hot(targets, num_classes=logits.shape[-1]).float()
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
        "count": int(len(targets)),
        "accuracy": correct.float().mean().item(),
        "nll": F.cross_entropy(logits.float(), targets).item(),
        "brier": (probabilities - one_hot).square().sum(dim=-1).mean().item(),
        "ece_15_bins": ece,
        "mean_confidence": confidence.mean().item(),
    }


def _per_class_metrics(
    logits: Tensor, targets: Tensor, class_ids: list[int]
) -> dict[str, dict[str, float | int]]:
    return {
        str(class_id): _metrics(logits[targets == class_index], targets[targets == class_index])
        for class_index, class_id in enumerate(class_ids)
        if bool((targets == class_index).any())
    }


def _macro_metrics(per_class: dict[str, dict[str, float | int]]) -> dict[str, float]:
    if not per_class:
        raise ValueError("macro metrics require at least one represented class")
    return {
        metric: sum(float(values[metric]) for values in per_class.values()) / len(per_class)
        for metric in ("accuracy", "nll", "brier", "ece_15_bins", "mean_confidence")
    }


def _class_weights(targets: Tensor, class_count: int) -> Tensor:
    counts = torch.bincount(targets, minlength=class_count).float()
    if (counts == 0).any():
        raise ValueError("every class must appear in the training split")
    weights = len(targets) / (class_count * counts)
    return weights / weights.mean()


def _load_split(feature_dir: Path, split: str, expected_class_ids: list[int] | None = None):
    with np.load(feature_dir / f"{split}-features.npz", allow_pickle=False) as cache:
        ids = cache["sample_ids"].astype(str).tolist()
        feature_ids = cache["class_ids"].astype(np.int64).tolist()
        features = cache["features"].astype(np.float32, copy=True)
        targets = cache["targets"].astype(np.int64, copy=True)
        observations = cache["observation_ids"].astype(str).tolist()
        photos = cache["photo_ids"].astype(str).tolist()
    rows = _jsonl(feature_dir / f"{split}-rows.jsonl")
    if [str(row["sample_id"]) for row in rows] != ids:
        raise ValueError(f"{split} feature sample order differs from its row manifest")
    if (
        len(set(ids)) != len(ids)
        or len(set(observations)) != len(ids)
        or len(set(photos)) != len(ids)
    ):
        raise ValueError(f"{split} feature cache contains repeated sample or media identity")
    if expected_class_ids is not None and feature_ids != expected_class_ids:
        raise ValueError("train and validation class order differs")
    if features.shape != (len(rows), IMAGE_WIDTH) or not np.isfinite(features).all():
        raise ValueError(f"invalid {split} feature array")
    if targets.shape != (len(rows),) or (targets < 0).any() or (targets >= len(feature_ids)).any():
        raise ValueError(f"invalid {split} target index array")
    if any(
        int(row["taxon_id"]) != feature_ids[int(target)]
        for row, target in zip(rows, targets, strict=True)
    ):
        raise ValueError(f"{split} target indices do not match source taxon labels")
    return rows, torch.from_numpy(features), torch.from_numpy(targets), feature_ids


def _evaluate(model, taxonomy: str, features: Tensor, targets: Tensor, class_ids: list[int]):
    model.eval()
    with torch.inference_mode():
        logits = model(features, taxonomy)
    per_class = _per_class_metrics(logits, targets, class_ids)
    return {
        "overall": _metrics(logits, targets),
        "per_class": per_class,
        "macro": _macro_metrics(per_class),
        "logits": logits.detach().clone(),
    }


def _predictions(
    rows: list[dict[str, Any]], logits: Tensor, class_ids: list[int], class_names: list[str]
) -> list[dict[str, Any]]:
    probabilities = torch.softmax(logits.float(), dim=-1)
    indexes = probabilities.argmax(dim=-1)
    return [
        {
            "sample_id": row["sample_id"],
            "observation_id": row["observation_id"],
            "photo_id": row["photo_id"],
            "media_sha256": row["media_sha256"],
            "target_class_id": int(row["taxon_id"]),
            "target_label": str(row["label"]),
            "prediction_class_id": class_ids[int(index)],
            "prediction_label": class_names[int(index)],
            "probabilities": [
                {
                    "class_id": class_ids[class_index],
                    "class_label": class_names[class_index],
                    "probability": float(probabilities[row_index, class_index]),
                }
                for class_index in range(len(class_ids))
            ],
        }
        for row_index, (row, index) in enumerate(zip(rows, indexes, strict=True))
    ]


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _option_order_max_delta(model, taxonomy: str, features: Tensor, class_ids: list[int]) -> float:
    options = [{"option_id": str(value), "class_id": str(value)} for value in class_ids]
    maximum = 0.0
    with torch.inference_mode():
        for feature in features[:32]:
            forward = model.decide(feature, taxonomy, options)
            reverse = model.decide(feature, taxonomy, list(reversed(options)))
            forward_by_id = {
                str(option["option_id"]): float(option["probability"])
                for option in forward["options"]
            }
            reverse_by_id = {
                str(option["option_id"]): float(option["probability"])
                for option in reverse["options"]
            }
            maximum = max(
                maximum,
                max(abs(forward_by_id[key] - reverse_by_id[key]) for key in forward_by_id),
            )
    return maximum


def _option_order_is_stable(maximum_delta: float) -> bool:
    return math.isfinite(maximum_delta) and 0.0 <= maximum_delta <= OPTION_ORDER_FLOAT32_TOLERANCE


def run(feature_dir: Path, output_dir: Path, epochs: int = DEFAULT_EPOCHS) -> dict[str, Any]:
    started_counter = time.perf_counter()
    started_utc = datetime.now(UTC).isoformat()
    feature_dir = feature_dir.resolve(strict=True)
    output_dir = output_dir.resolve()
    if output_dir.is_relative_to(ROOT):
        raise ValueError("trained readout and source IDs must stay outside the Git repository")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite readout output: {output_dir}")
    if epochs < 1 or epochs > 2048:
        raise ValueError("epochs must be between 1 and 2048")
    feature_config = json.loads((feature_dir / "config.json").read_text(encoding="utf-8"))
    if (
        feature_config.get("status") != "completed"
        or feature_config.get("video_model_revision") != "be719a406d563b66f0ac969e7c94bab8e997c81a"
        or feature_config.get("checkpoint_sha256") != ENCODER_CHECKPOINT_SHA256
        or feature_config.get("sealed_audit_loaded") is not False
    ):
        raise ValueError("frozen VideoPrism feature run is incomplete or unpinned")
    train_rows, x_train, y_train, class_ids = _load_split(feature_dir, "train")
    val_rows, x_val, y_val, val_class_ids = _load_split(feature_dir, "validation", class_ids)
    if not train_rows or not val_rows:
        raise ValueError("train and validation must both be nonempty")
    for key in ("observation_id", "photo_id", "observer_group_sha256", "normalized_rgb_sha256"):
        if {str(row[key]) for row in train_rows} & {str(row[key]) for row in val_rows}:
            raise ValueError(f"train/validation {key} overlap")
    class_names_by_id = {}
    for row in train_rows + val_rows:
        class_id = int(row["taxon_id"])
        label = str(row["label"])
        if class_id in class_names_by_id and class_names_by_id[class_id] != label:
            raise ValueError("a source taxon ID has conflicting class labels")
        class_names_by_id[class_id] = label
    class_names = [class_names_by_id[class_id] for class_id in class_ids]
    taxonomy = "inaturalist_bird_species"
    output_dir.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(4)
    from tiny_omni_decision.textfree_image import TextFreeImageReadout

    model = TextFreeImageReadout(IMAGE_WIDTH, {taxonomy: tuple(str(value) for value in class_ids)})
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    weights = _class_weights(y_train, len(class_ids))
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters())
    if trainable_parameters != IMAGE_WIDTH * len(class_ids) + len(class_ids):
        raise AssertionError("unexpected linear readout parameter count")

    best_score = math.inf
    best_epoch = 0
    best_state = None
    history = []
    history_path = output_dir / "training-history.jsonl"
    final_state = None
    training_seconds = 0.0
    evaluation_seconds = 0.0
    model.train()
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        logits = model(x_train, taxonomy)
        loss = F.cross_entropy(logits, y_train, weight=weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.linalg.vector_norm(
            torch.stack(
                [
                    parameter.grad.detach().norm()
                    for parameter in model.parameters()
                    if parameter.grad is not None
                ]
            )
        ).item()
        optimizer.step()
        training_seconds += time.perf_counter() - started

        evaluation_started = time.perf_counter()
        train_result = _evaluate(model, taxonomy, x_train, y_train, class_ids)
        val_result = _evaluate(model, taxonomy, x_val, y_val, class_ids)
        evaluation_seconds += time.perf_counter() - evaluation_started
        score = float(val_result["macro"]["nll"])
        improved = score < best_score
        record = {
            "epoch": epoch,
            "optimizer_updates": epoch,
            "train_cross_entropy_weighted": float(loss.detach()),
            "gradient_norm": gradient_norm,
            "train_metrics": {
                key: value for key, value in train_result.items() if key != "logits"
            },
            "validation_metrics": {
                key: value for key, value in val_result.items() if key != "logits"
            },
            "best_so_far_epoch": epoch if improved else best_epoch,
            "best_so_far_macro_validation_nll": score if improved else best_score,
        }
        history.append(record)
        with history_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        if improved:
            best_score = score
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            torch.save(best_state, output_dir / "best-readout.pt")
            _write_json(
                output_dir / "best-checkpoint.json",
                {
                    "epoch": epoch,
                    "macro_validation_nll": score,
                    "selector": "earliest minimum macro validation NLL",
                },
            )
        final_state = {
            key: value.detach().cpu().clone() for key, value in model.state_dict().items()
        }
        if epoch == 1 or epoch % 16 == 0 or epoch == epochs:
            print(
                json.dumps(
                    {
                        "epoch": epoch,
                        "train_accuracy": train_result["overall"]["accuracy"],
                        "validation_accuracy": val_result["overall"]["accuracy"],
                        "validation_macro_accuracy": val_result["macro"]["accuracy"],
                        "validation_macro_nll": score,
                        "best_epoch": best_epoch,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    if best_state is None or final_state is None:
        raise RuntimeError("training produced no best or final checkpoint")
    torch.save(final_state, output_dir / "final-readout.pt")
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        selected_logits = model(x_val, taxonomy)
        zero_logits = model(torch.zeros_like(x_val), taxonomy)
        generator = torch.Generator().manual_seed(SEED + 1000)
        permutation = torch.randperm(len(x_val), generator=generator)
        shuffled_logits = model(x_val[permutation], taxonomy)
    selected_metrics = _evaluate(model, taxonomy, x_val, y_val, class_ids)
    baseline_logits = torch.zeros((len(y_val), len(class_ids)), dtype=torch.float32)
    majority_class_index = int(torch.bincount(y_train, minlength=len(class_ids)).argmax())
    majority_logits = torch.full_like(baseline_logits, -20.0)
    majority_logits[:, majority_class_index] = 20.0
    control_metrics = {
        "aligned_image": {
            "overall": _metrics(selected_logits, y_val),
            "macro": _macro_metrics(_per_class_metrics(selected_logits, y_val, class_ids)),
        },
        "zero_image": {
            "overall": _metrics(zero_logits, y_val),
            "macro": _macro_metrics(_per_class_metrics(zero_logits, y_val, class_ids)),
            "top1_agreement_with_aligned": float(
                (zero_logits.argmax(-1) == selected_logits.argmax(-1)).float().mean()
            ),
        },
        "validation_image_permutation": {
            "overall": _metrics(shuffled_logits, y_val),
            "macro": _macro_metrics(_per_class_metrics(shuffled_logits, y_val, class_ids)),
            "top1_agreement_with_aligned": float(
                (shuffled_logits.argmax(-1) == selected_logits.argmax(-1)).float().mean()
            ),
            "permutation_seed": SEED + 1000,
        },
        "uniform_option_distribution": {
            "overall": _metrics(baseline_logits, y_val),
            "macro": _macro_metrics(_per_class_metrics(baseline_logits, y_val, class_ids)),
        },
        "training_majority_class": {
            "overall": _metrics(majority_logits, y_val),
            "macro": _macro_metrics(_per_class_metrics(majority_logits, y_val, class_ids)),
            "predicted_class_id": class_ids[majority_class_index],
        },
    }
    max_option_order_delta = _option_order_max_delta(model, taxonomy, x_val, class_ids)
    predictions = _predictions(val_rows, selected_logits, class_ids, class_names)
    prediction_path = output_dir / "validation-predictions.jsonl"
    prediction_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in predictions),
        encoding="utf-8",
    )
    model.load_state_dict(
        torch.load(output_dir / "best-readout.pt", map_location="cpu", weights_only=True)
    )
    with torch.inference_mode():
        reloaded_logits = model(x_val, taxonomy)
    reload_delta = float((reloaded_logits - selected_logits).abs().max())
    if reload_delta != 0.0 or not _option_order_is_stable(max_option_order_delta):
        raise AssertionError("best readout reload or option permutation parity failed")

    metadata = {
        "schema_version": 1,
        "experiment_id": "inat-birds-videoprism-frozen-linear-readout-v1",
        "status": "completed",
        "source_commit": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "runner_sha256": _sha256(Path(__file__)),
        "feature_config_sha256": _sha256(feature_dir / "config.json"),
        "feature_hashes": {
            split: _sha256(feature_dir / f"{split}-features.npz")
            for split in ("train", "validation")
        },
        "started_utc": started_utc,
        "completed_utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "epochs_configured": epochs,
        "completed_epochs": len(history),
        "optimizer_updates": len(history),
        "checkpoint_selector": "earliest minimum macro validation NLL",
        "best_epoch": best_epoch,
        "train_samples": len(train_rows),
        "validation_samples": len(val_rows),
        "class_ids": class_ids,
        "class_names": class_names,
        "train_class_counts": dict(
            sorted(Counter(int(row["taxon_id"]) for row in train_rows).items())
        ),
        "validation_class_counts": dict(
            sorted(Counter(int(row["taxon_id"]) for row in val_rows).items())
        ),
        "class_loss_weights": {
            str(class_id): float(weights[index]) for index, class_id in enumerate(class_ids)
        },
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "loss": "inverse-frequency class-weighted train CE; unweighted validation metrics",
        "trainable_readout_parameters": trainable_parameters,
        "frozen_encoder_parameters": ENCODER_PARAMETERS,
        "total_unquantized_parameters": ENCODER_PARAMETERS + trainable_parameters,
        "unique_train_examples_consumed": len(train_rows),
        "total_train_example_exposures": len(train_rows) * len(history),
        "repeated_train_example_exposures": len(train_rows) * (len(history) - 1),
        "encoder_checkpoint_sha256": ENCODER_CHECKPOINT_SHA256,
        "encoder_checkpoint_bytes": 457_475_296,
        "readout_checkpoint_bytes": (output_dir / "best-readout.pt").stat().st_size,
        "estimated_unquantized_weight_bytes": (
            457_475_296 + (output_dir / "best-readout.pt").stat().st_size
        ),
        "best_validation_metrics": {
            key: value for key, value in selected_metrics.items() if key != "logits"
        },
        "input_controls": control_metrics,
        "option_order_max_probability_delta_first_32": max_option_order_delta,
        "best_reload_max_logit_delta": reload_delta,
        "training_seconds": training_seconds,
        "evaluation_seconds": evaluation_seconds,
        "total_wall_seconds": time.perf_counter() - started_counter,
        "device": "CPU",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "sealed_audit_loaded": False,
        "task_scope": (
            "closed-set eight-species image classification; not open-vocabulary decisions"
        ),
        "artifact_sha256": {
            name: _sha256(output_dir / name)
            for name in (
                "best-readout.pt",
                "final-readout.pt",
                "validation-predictions.jsonl",
                "training-history.jsonl",
            )
        },
        "versions": {key: importlib.metadata.version(key) for key in ("torch", "numpy")},
    }
    _write_json(output_dir / "run-metadata.json", metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    args = parser.parse_args()
    run(args.feature_dir, args.output_dir, args.epochs)


if __name__ == "__main__":
    main()
