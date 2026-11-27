"""Fit and compare a small LSTM on pinned Physical Concepts frozen features."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from scripts.train_intphys_next_feature_predictor import (
    fit_predictor,
    load_feature_split,
    validate_group_disjointness,
)
from tiny_omni_decision.temporal_prediction import (
    NextFrameFeaturePredictor,
    cosine_prediction_loss,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def persistence_cosine_loss(features: np.ndarray) -> float:
    """Measure the no-training baseline that predicts each next feature unchanged."""
    if features.ndim != 3 or features.shape[0] < 1 or features.shape[1] < 2:
        raise ValueError("features must have shape [N,T,D] with N>=1 and T>=2")
    if not np.isfinite(features).all():
        raise ValueError("features must be finite")
    previous = features[:, :-1]
    following = features[:, 1:]
    previous_norm = np.linalg.norm(previous, axis=-1, keepdims=True)
    following_norm = np.linalg.norm(following, axis=-1, keepdims=True)
    if np.any(previous_norm == 0) or np.any(following_norm == 0):
        raise ValueError("persistence cosine loss is undefined for zero vectors")
    similarities = np.sum(previous * following, axis=-1) / (
        previous_norm[..., 0] * following_norm[..., 0]
    )
    return float(np.mean(1.0 - np.clip(similarities, -1.0, 1.0)))


def evaluate_validation_loss(
    model: Any, features: np.ndarray, batch_size: int = 64
) -> float:
    import torch

    if batch_size < 1:
        raise ValueError("batch size must be positive")
    values = torch.from_numpy(np.ascontiguousarray(features))
    total = 0.0
    count = 0
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            batch = values[start : start + batch_size]
            loss = cosine_prediction_loss(model(batch), batch)
            total += float(loss) * len(batch)
            count += len(batch)
    if count != len(features):
        raise ValueError("validation evaluation did not consume every sequence")
    return total / count


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--validation-features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    return parser.parse_args()


def main() -> None:
    import torch

    args = _args()
    train_path = args.train_features.resolve(strict=True)
    validation_path = args.validation_features.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    if train_path == validation_path:
        raise ValueError("training and validation features must be separate files")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    train, train_ids, train_groups = load_feature_split(train_path)
    validation, validation_ids, validation_groups = load_feature_split(validation_path)
    validate_group_disjointness(train_ids, train_groups, validation_ids, validation_groups)

    started_at = datetime.now(UTC).isoformat()
    start = time.perf_counter()
    model, history, selected_epoch = fit_predictor(
        train,
        validation,
        seed=args.seed,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        min_delta=args.min_delta,
    )
    elapsed_seconds = time.perf_counter() - start
    selected = history[selected_epoch - 1]
    persistence_loss = persistence_cosine_loss(validation)
    model_loss = float(selected["validation_cosine_loss"])
    if not np.isfinite(model_loss):
        raise FloatingPointError("selected validation loss is non-finite")

    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = output_dir / "best-next-feature-predictor.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_width": int(train.shape[-1]),
            "hidden_width": 64,
            "seed": args.seed,
            "selected_epoch": selected_epoch,
            "validation_cosine_loss": model_loss,
        },
        checkpoint_path,
    )
    saved_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    reloaded = NextFrameFeaturePredictor(feature_width=int(train.shape[-1]), hidden_width=64)
    reloaded.load_state_dict(saved_checkpoint["state_dict"], strict=True)
    reload_validation_loss = evaluate_validation_loss(reloaded, validation, args.batch_size)
    if abs(reload_validation_loss - model_loss) > 1e-7:
        raise RuntimeError("reloaded best checkpoint does not match its selected validation loss")
    history_path = output_dir / "training-history.jsonl"
    history_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in history),
        encoding="utf-8",
    )
    metadata: dict[str, Any] = {
        "status": "complete_physical_concepts_frozen_feature_next_step_fit",
        "started_at_utc": started_at,
        "ended_at_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "device": "cpu",
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "command": sys.argv,
        "seed": args.seed,
        "max_epochs": args.max_epochs,
        "epochs_completed": len(history),
        "selected_epoch": selected_epoch,
        "selection": "minimum internal shard-held-out validation cosine loss",
        "checkpoint_min_delta": 0.0,
        "patience_min_delta": args.min_delta,
        "optimizer": "AdamW",
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": args.batch_size,
        "early_stopping_patience": args.patience,
        "min_delta": args.min_delta,
        "train_examples": len(train),
        "validation_examples": len(validation),
        "sequence_length": int(train.shape[1]),
        "feature_width": int(train.shape[2]),
        "train_unique_groups": len(set(train_groups)),
        "validation_unique_groups": len(set(validation_groups)),
        "train_sample_ids_sha256": hashlib.sha256("\n".join(train_ids).encode()).hexdigest(),
        "validation_sample_ids_sha256": hashlib.sha256(
            "\n".join(validation_ids).encode()
        ).hexdigest(),
        "train_features_path": str(train_path),
        "validation_features_path": str(validation_path),
        "train_features_sha256": sha256_file(train_path),
        "validation_features_sha256": sha256_file(validation_path),
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "selected_train_cosine_loss": float(selected["train_cosine_loss"]),
        "selected_validation_cosine_loss": model_loss,
        "reloaded_validation_cosine_loss": reload_validation_loss,
        "validation_persistence_cosine_loss": persistence_loss,
        "validation_loss_minus_persistence": model_loss - persistence_loss,
        "best_checkpoint_sha256": sha256_file(checkpoint_path),
        "history_sha256": sha256_file(history_path),
        "quality_scope": (
            "self-supervised next-frozen-feature prediction only; no physical-event labels, "
            "calibrated probabilities, or product Decision accuracy"
        ),
        "probe_data_opened": False,
    }
    (output_dir / "run-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
