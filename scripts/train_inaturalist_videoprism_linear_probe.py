"""Fit a small CPU linear probe on the streamed iNaturalist VideoPrism cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metrics(logits: torch.Tensor, targets: torch.Tensor, bins: int = 15) -> dict[str, Any]:
    probabilities = logits.softmax(dim=-1)
    confidence, predictions = probabilities.max(dim=-1)
    correct = predictions.eq(targets)
    one_hot = F.one_hot(targets, num_classes=logits.shape[-1]).to(probabilities.dtype)
    ece = logits.new_zeros(())
    bin_counts = []
    for bin_index in range(bins):
        lower, upper = bin_index / bins, (bin_index + 1) / bins
        in_bin = (confidence >= lower) & (
            confidence < upper if bin_index < bins - 1 else confidence <= upper
        )
        count = int(in_bin.sum())
        bin_counts.append(count)
        if count:
            ece += (
                in_bin.float().mean()
                * (confidence[in_bin].mean() - correct[in_bin].float().mean()).abs()
            )
    per_class = {}
    for label in range(logits.shape[-1]):
        mask = targets.eq(label)
        if mask.any():
            per_class[str(label)] = {
                "count": int(mask.sum()),
                "accuracy": float(correct[mask].float().mean()),
                "nll": float(F.cross_entropy(logits[mask], targets[mask])),
            }
    return {
        "count": len(targets),
        "accuracy": float(correct.float().mean()),
        "macro_accuracy": float(np.mean([v["accuracy"] for v in per_class.values()])),
        "nll": float(F.cross_entropy(logits, targets)),
        "brier": float((probabilities - one_hot).square().sum(dim=-1).mean()),
        "ece_15_bins": float(ece),
        "ece_bin_counts": bin_counts,
        "mean_confidence": float(confidence.mean()),
        "per_class": per_class,
    }


def load_verified_data(
    feature_dir: Path,
    metadata_dir: Path,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    result_path = feature_dir / "result.json"
    status = json.loads((feature_dir / "run-status.json").read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if status.get("status") != "COMPLETE" or result.get("status") != "complete":
        raise ValueError("feature extraction is not complete")
    if status.get("result_sha256") != sha256_file(result_path):
        raise ValueError("feature result hash mismatch")
    cache_path = feature_dir / "features.npy"
    rows_path = feature_dir / "feature-rows.jsonl"
    if result.get("feature_cache_sha256") != sha256_file(cache_path):
        raise ValueError("feature cache hash mismatch")
    if result.get("feature_rows_sha256") != sha256_file(rows_path):
        raise ValueError("feature row hash mismatch")

    metadata: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "validation"):
        source = metadata_dir / f"{split}.jsonl"
        if result.get("metadata_sha256", {}).get(split) != sha256_file(source):
            raise ValueError(f"{split} metadata hash mismatch")
        metadata[split] = [
            json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()
        ]
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()]
    expected_rows = [
        (split, row, f"inat:{int(row['taxon_id'])}:{row['observation_id']}")
        for split in ("train", "validation")
        for row in metadata[split]
    ]
    if len(rows) != len(expected_rows):
        raise ValueError("feature row count does not match metadata")
    for actual, (split, source, sample_id) in zip(rows, expected_rows, strict=True):
        if (
            actual.get("split") != split
            or actual.get("sample_id") != sample_id
            or int(actual.get("taxon_id", -1)) != int(source["taxon_id"])
            or actual.get("observation_id") != str(source["observation_id"])
        ):
            raise ValueError("feature row identity/order differs from frozen metadata")
    class_ids = sorted({int(row["taxon_id"]) for split in metadata.values() for row in split})
    if len(class_ids) != 8:
        raise ValueError("expected exactly eight fixed taxa")
    class_to_index = {class_id: index for index, class_id in enumerate(class_ids)}
    values = np.load(cache_path, mmap_mode="r")
    if values.shape != (len(rows), 768) or values.dtype != np.float32:
        raise ValueError("feature cache shape or dtype mismatch")
    if not np.isfinite(values).all():
        raise ValueError("feature cache contains non-finite values")
    train_count = len(metadata["train"])
    features = torch.from_numpy(np.array(values, dtype=np.float32, copy=True))
    targets = torch.tensor([class_to_index[int(row["taxon_id"])] for row in rows])
    return (
        features[:train_count],
        targets[:train_count],
        features[train_count:],
        targets[train_count:],
        class_ids,
    )


def run(
    feature_dir: Path,
    metadata_dir: Path,
    output_dir: Path,
    *,
    seed: int = 17,
    epochs: int = 200,
    patience: int = 25,
    min_delta: float = 1e-4,
    learning_rate: float = 0.03,
    weight_decay: float = 1e-4,
) -> dict[str, Any]:
    feature_dir = feature_dir.resolve(strict=True)
    metadata_dir = metadata_dir.resolve(strict=True)
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {output_dir}")
    started_utc = datetime.now(UTC).isoformat()
    started = time.perf_counter()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    device = torch.device("cpu")
    x_train, y_train, x_validation, y_validation, class_ids = load_verified_data(
        feature_dir, metadata_dir
    )
    mean = x_train.mean(dim=0)
    scale = x_train.std(dim=0, unbiased=False).clamp_min(1e-6)
    x_train = (x_train - mean) / scale
    x_validation = (x_validation - mean) / scale

    model = nn.Linear(x_train.shape[1], len(class_ids), device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    history = []
    best_state = None
    best_nll = float("inf")
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_logits = model(x_train)
        train_loss = F.cross_entropy(train_logits, y_train)
        train_loss.backward()
        optimizer.step()
        model.eval()
        with torch.inference_mode():
            train_metrics = metrics(model(x_train), y_train)
            validation_logits = model(x_validation)
            validation_metrics = metrics(validation_logits, y_validation)
        history.append(
            {
                "epoch": epoch,
                "train_ce": float(train_loss.detach()),
                "train_metrics": train_metrics,
                "validation_metrics": validation_metrics,
            }
        )
        if validation_metrics["nll"] < best_nll - min_delta:
            best_nll = validation_metrics["nll"]
            best_epoch = epoch
            stale_epochs = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break
    if best_state is None:
        raise RuntimeError("validation did not select a readout checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        selected_train_logits = model(x_train)
        selected_validation_logits = model(x_validation)
    selected_train_metrics = metrics(selected_train_logits, y_train)
    selected_validation_metrics = metrics(selected_validation_logits, y_validation)
    for metric_group in (selected_train_metrics, selected_validation_metrics):
        metric_group["per_class"] = {
            str(class_ids[int(index)]): values
            for index, values in metric_group["per_class"].items()
        }
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = output_dir / "best-linear-readout.pt"
    torch.save(
        {
            "state_dict": best_state,
            "train_mean": mean,
            "train_scale": scale,
            "class_ids": class_ids,
            "selected_epoch": best_epoch,
        },
        checkpoint_path,
    )
    predictions_path = output_dir / "validation-predictions.jsonl"
    probabilities = selected_validation_logits.softmax(dim=-1)
    with predictions_path.open("w", encoding="utf-8", newline="\n") as stream:
        for row, target, probability in zip(
            [
                json.loads(line)
                for line in (feature_dir / "feature-rows.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ][len(y_train) :],
            y_validation.tolist(),
            probabilities.tolist(),
            strict=True,
        ):
            predicted = int(np.argmax(probability))
            stream.write(
                json.dumps(
                    {
                        "sample_id": row["sample_id"],
                        "observation_id": row["observation_id"],
                        "photo_id": row["photo_id"],
                        "target_taxon_id": class_ids[target],
                        "prediction_taxon_id": class_ids[predicted],
                        "probabilities_by_taxon_id": {
                            str(class_id): probability[index]
                            for index, class_id in enumerate(class_ids)
                        },
                    },
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )
    history_path = output_dir / "training-history.jsonl"
    history_path.write_text(
        "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in history),
        encoding="utf-8",
    )
    config = {
        "seed": seed,
        "device": str(device),
        "architecture": "single affine 768-to-8 linear classifier",
        "normalization": "train-split mean/std only; std floor 1e-6",
        "optimizer": "AdamW",
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "max_epochs": epochs,
        "early_stopping": "validation NLL with min_delta and patience",
        "patience": patience,
        "min_delta": min_delta,
        "selection": "lowest validation NLL; selected checkpoint retained separately",
        "class_ids": class_ids,
        "feature_dir": str(feature_dir),
        "feature_cache_sha256": sha256_file(feature_dir / "features.npy"),
        "feature_rows_sha256": sha256_file(feature_dir / "feature-rows.jsonl"),
        "metadata_hashes": json.loads((feature_dir / "result.json").read_text())["metadata_sha256"],
    }
    config_path = output_dir / "config.json"
    config_path.write_text(
        json.dumps(config, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    train_counts = torch.bincount(y_train, minlength=len(class_ids))
    majority_class = int(train_counts.argmax())
    result = {
        "status": "complete_inaturalist_videoprism_linear_probe",
        "claim_scope": (
            "eight-taxon observer-disjoint image classification; not general Decision quality"
        ),
        "started_utc": started_utc,
        "completed_utc": datetime.now(UTC).isoformat(),
        "wall_seconds": time.perf_counter() - started,
        "device": str(device),
        "train_count": len(y_train),
        "validation_count": len(y_validation),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "epochs_run": len(history),
        "selected_epoch": best_epoch,
        "best_validation_nll": best_nll,
        "selected_train_metrics": selected_train_metrics,
        "selected_validation_metrics": selected_validation_metrics,
        "chance_accuracy": 1.0 / len(class_ids),
        "majority_train_prior_taxon_id": class_ids[majority_class],
        "majority_train_prior_accuracy": float((y_validation == majority_class).float().mean()),
        "config_sha256": sha256_file(config_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "history_sha256": sha256_file(history_path),
        "predictions_sha256": sha256_file(predictions_path),
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    args = parser.parse_args()
    result = run(
        args.feature_dir,
        args.metadata_dir,
        args.output_dir,
        seed=args.seed,
        epochs=args.epochs,
        patience=args.patience,
        min_delta=args.min_delta,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
