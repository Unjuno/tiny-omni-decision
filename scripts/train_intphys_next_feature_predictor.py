"""Fit a tiny VideoPrism next-feature predictor on train-only sequences."""

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


def load_feature_split(path: Path) -> tuple[np.ndarray, list[str], list[str]]:
    """Load validated `[N,T,D]` frozen features and group-aware identities."""
    with np.load(path, allow_pickle=False) as payload:
        required = {"features", "sample_ids", "group_ids"}
        if not required.issubset(payload.files):
            missing = sorted(required - set(payload.files))
            raise ValueError(f"feature archive missing arrays: {missing}")
        features = np.asarray(payload["features"], dtype=np.float32)
        sample_ids = payload["sample_ids"].astype(str).tolist()
        group_ids = payload["group_ids"].astype(str).tolist()
    if (
        features.ndim != 3
        or features.shape[0] < 2
        or features.shape[1] < 2
        or features.shape[2] < 1
        or not np.isfinite(features).all()
    ):
        raise ValueError("features must be finite nonempty [N,T,D] with N>=2 and T>=2")
    if len(sample_ids) != len(features) or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sample IDs must be unique and match the feature count")
    if len(group_ids) != len(features) or any(not item for item in group_ids):
        raise ValueError("group IDs must be nonempty and match the feature count")
    return features, sample_ids, group_ids


def validate_group_disjointness(
    train_ids: list[str],
    train_groups: list[str],
    validation_ids: list[str],
    validation_groups: list[str],
) -> None:
    if set(train_ids) & set(validation_ids):
        raise ValueError("sample IDs overlap between training and validation")
    if set(train_groups) & set(validation_groups):
        raise ValueError("underlying group IDs overlap between training and validation")


def _mean_loss(
    model: NextFrameFeaturePredictor,
    features: torch.Tensor,
    batch_size: int,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            batch = features[start : start + batch_size]
            loss = cosine_prediction_loss(model(batch), batch)
            total += float(loss) * len(batch)
            count += len(batch)
    if count == 0:
        raise ValueError("cannot evaluate an empty feature split")
    return total / count


def fit_predictor(
    train_features: np.ndarray,
    validation_features: np.ndarray,
    *,
    seed: int = 17,
    max_epochs: int = 100,
    batch_size: int = 64,
    patience: int = 8,
    min_delta: float = 1e-4,
) -> tuple[NextFrameFeaturePredictor, list[dict[str, float]], int]:
    if max_epochs < 1 or batch_size < 1 or patience < 1 or min_delta < 0:
        raise ValueError("epochs, batch size and patience must be positive; min_delta nonnegative")
    if train_features.ndim != 3 or validation_features.ndim != 3:
        raise ValueError("train and validation features must have shape [N,T,D]")
    if train_features.shape[1:] != validation_features.shape[1:]:
        raise ValueError("train and validation frame/feature dimensions must match")
    if train_features.shape[0] < 2 or validation_features.shape[0] < 1:
        raise ValueError("train needs at least 2 examples and validation at least 1")
    if not np.isfinite(train_features).all() or not np.isfinite(validation_features).all():
        raise ValueError("features must be finite")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    train = torch.from_numpy(np.ascontiguousarray(train_features))
    validation = torch.from_numpy(np.ascontiguousarray(validation_features))
    model = NextFrameFeaturePredictor(feature_width=train.shape[-1], hidden_width=64)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(len(train), generator=generator)
        total_train_loss = 0.0
        seen = 0
        for start in range(0, len(order), batch_size):
            batch = train[order[start : start + batch_size]]
            optimizer.zero_grad(set_to_none=True)
            loss = cosine_prediction_loss(model(batch), batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss at epoch {epoch}")
            loss.backward()
            gradients = [
                parameter.grad for parameter in model.parameters() if parameter.grad is not None
            ]
            if not gradients or any(not torch.isfinite(gradient).all() for gradient in gradients):
                raise FloatingPointError(f"non-finite or missing gradients at epoch {epoch}")
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0, error_if_nonfinite=True
            )
            optimizer.step()
            total_train_loss += float(loss.detach()) * len(batch)
            seen += len(batch)
        validation_loss = _mean_loss(model, validation, batch_size)
        row = {
            "epoch": float(epoch),
            "train_cosine_loss": total_train_loss / seen,
            "validation_cosine_loss": validation_loss,
        }
        history.append(row)
        if validation_loss < best_loss - min_delta:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is None:
        raise RuntimeError("no finite validation-selected checkpoint was produced")
    model.load_state_dict(best_state, strict=True)
    return model, history, best_epoch


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
    args = _args()
    train_path = args.train_features.resolve(strict=True)
    validation_path = args.validation_features.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    if train_path == validation_path:
        raise ValueError("training and validation feature files must be independent")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing run directory: {output_dir}")
    train, train_ids, train_groups = load_feature_split(train_path)
    validation, validation_ids, validation_groups = load_feature_split(validation_path)
    validate_group_disjointness(train_ids, train_groups, validation_ids, validation_groups)
    if train.shape[2] != validation.shape[2]:
        raise ValueError("feature widths differ between splits")
    started_at = datetime.now(UTC).isoformat()
    started = time.perf_counter()
    model, history, best_epoch = fit_predictor(
        train,
        validation,
        seed=args.seed,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        min_delta=args.min_delta,
    )
    elapsed_seconds = time.perf_counter() - started
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = output_dir / "best-next-feature-predictor.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_width": int(train.shape[2]),
            "hidden_width": 64,
            "seed": args.seed,
            "best_epoch": best_epoch,
        },
        checkpoint,
    )
    history_path = output_dir / "training-history.jsonl"
    history_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in history),
        encoding="utf-8",
    )
    metadata: dict[str, Any] = {
        "status": "complete_train_only_next_feature_predictor",
        "started_at_utc": started_at,
        "ended_at_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "device": "cpu",
        "seed": args.seed,
        "max_epochs": args.max_epochs,
        "epochs_completed": len(history),
        "selected_epoch": best_epoch,
        "selection": "minimum internal train-source validation cosine prediction loss",
        "optimizer": "AdamW",
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": args.batch_size,
        "early_stopping_patience": args.patience,
        "min_delta": args.min_delta,
        "train_examples": len(train),
        "validation_examples": len(validation),
        "train_unique_groups": len(set(train_groups)),
        "validation_unique_groups": len(set(validation_groups)),
        "train_features_sha256": sha256_file(train_path),
        "validation_features_sha256": sha256_file(validation_path),
        "best_checkpoint_sha256": sha256_file(checkpoint),
        "history_sha256": sha256_file(history_path),
        "best_validation_cosine_loss": history[best_epoch - 1]["validation_cosine_loss"],
        "train_sample_ids_sha256": hashlib.sha256("\n".join(train_ids).encode()).hexdigest(),
        "validation_sample_ids_sha256": hashlib.sha256(
            "\n".join(validation_ids).encode()
        ).hexdigest(),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "quality_scope": "next-frozen-feature prediction only; not a calibrated class probability",
    }
    (output_dir / "run-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
