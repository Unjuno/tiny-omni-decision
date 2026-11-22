"""Fit a small frozen-MiniLM readout on the pinned BANKING77 TRAIN split."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer

MODEL_REVISION = "4ca70771034acceecb2e72475f72050fcdde4ddc"
MODEL_SHA256 = "cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433"
SPLIT_FORMAT = "banking77_content_grouped_train_validation_v1"
CLASS_COUNT = 77


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def load_and_validate_data(
    data_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    manifest_path = data_dir / "split-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != SPLIT_FORMAT:
        raise ValueError("Unexpected split format")
    if manifest.get("official_test_loaded") is not False:
        raise ValueError("Refusing split manifest that opened official test")
    train = read_jsonl(data_dir / "train.jsonl")
    validation = read_jsonl(data_dir / "validation.jsonl")
    if len(train) != manifest["train"]["row_count"]:
        raise ValueError("Train row count mismatch")
    if len(validation) != manifest["validation"]["row_count"]:
        raise ValueError("Validation row count mismatch")
    for name, rows in (("train", train), ("validation", validation)):
        labels = {int(row["label"]) for row in rows}
        if labels != set(range(CLASS_COUNT)):
            raise ValueError(f"{name} does not contain all 77 expected classes")
    train_ids = {row["sample_id"] for row in train}
    validation_ids = {row["sample_id"] for row in validation}
    train_groups = {row["content_group_id"] for row in train}
    validation_groups = {row["content_group_id"] for row in validation}
    if train_ids & validation_ids or train_groups & validation_groups:
        raise ValueError("Split identity overlap detected")
    return train, validation, manifest


def extract_features(
    rows: list[dict[str, Any]], model_dir: Path, batch_size: int = 64
) -> torch.Tensor:
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    encoder = AutoModel.from_pretrained(model_dir, local_files_only=True).cpu().eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    features: list[torch.Tensor] = []
    texts = [str(row["text"]) for row in rows]
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(
                texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            output = encoder(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).to(output.dtype)
            pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            features.append(pooled.cpu().float())
    del encoder, tokenizer
    return torch.cat(features, dim=0)


def metric_summary(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    class_count = logits.shape[-1]
    if labels.numel() and (labels.min() < 0 or labels.max() >= class_count):
        raise ValueError("Label id is outside the logits class range")
    probabilities = torch.softmax(logits, dim=-1)
    predictions = probabilities.argmax(dim=-1)
    correct = predictions.eq(labels)
    accuracy = correct.float().mean().item()
    class_recalls: list[float] = []
    class_f1s: list[float] = []
    for label in range(class_count):
        target_mask = labels.eq(label)
        prediction_mask = predictions.eq(label)
        tp = (target_mask & prediction_mask).sum().item()
        support = target_mask.sum().item()
        predicted = prediction_mask.sum().item()
        recall = tp / support if support else 0.0
        precision = tp / predicted if predicted else 0.0
        class_recalls.append(recall)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        class_f1s.append(f1)
    one_hot = torch.nn.functional.one_hot(labels, class_count).to(probabilities.dtype)
    brier = (probabilities - one_hot).square().sum(dim=-1).mean().item()
    nll = torch.nn.functional.cross_entropy(logits, labels).item()
    confidence, _ = probabilities.max(dim=-1)
    ece = 0.0
    for bin_index in range(15):
        lower = bin_index / 15
        upper = (bin_index + 1) / 15
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            ece += mask.float().mean().item() * abs(
                confidence[mask].mean().item() - correct[mask].float().mean().item()
            )
    return {
        "accuracy": accuracy,
        "macro_recall": sum(class_recalls) / class_count,
        "macro_f1": sum(class_f1s) / class_count,
        "nll": nll,
        "brier": brier,
        "ece_15": ece,
    }


def train_readout(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    validation_features: torch.Tensor,
    validation_labels: torch.Tensor,
    *,
    seed: int = 17,
    learning_rate: float = 0.01,
    weight_decay: float = 0.01,
    batch_size: int = 256,
    max_epochs: int = 60,
    patience: int = 6,
    min_delta: float = 1e-5,
) -> tuple[nn.Module, dict[str, Any]]:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    train_mean = train_features.mean(dim=0)
    train_std = train_features.std(dim=0, unbiased=False).clamp_min(1e-5)
    x_train = (train_features - train_mean) / train_std
    x_validation = (validation_features - train_mean) / train_std
    model = nn.Linear(x_train.shape[1], CLASS_COUNT)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(x_train, train_labels),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    history: list[dict[str, Any]] = []
    best_nll = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, float] | None = None
    stale_epochs = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        batch_losses: list[float] = []
        for batch_features, batch_labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(batch_features), batch_labels)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        model.eval()
        with torch.inference_mode():
            validation_logits = model(x_validation)
            metrics = metric_summary(validation_logits, validation_labels)
        history.append(
            {
                "epoch": epoch,
                "train_ce": sum(batch_losses) / len(batch_losses),
                "validation": metrics,
            }
        )
        if metrics["nll"] < best_nll - min_delta:
            best_nll = metrics["nll"]
            best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            best_metrics = metrics
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break
    if best_state is None or best_metrics is None:
        raise RuntimeError("No validation checkpoint was selected")
    model.load_state_dict(best_state)
    model.eval()
    return model, {
        "history": history,
        "best_epoch": best_epoch,
        "best_validation_metrics": best_metrics,
        "epochs_completed": len(history),
        "train_mean": train_mean,
        "train_std": train_std,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output_dir}")
    model_weights = args.model_dir / "model.safetensors"
    actual_model_hash = sha256_file(model_weights)
    if actual_model_hash != MODEL_SHA256:
        raise ValueError("MiniLM-L3 weights do not match the pinned checkpoint")

    train_rows, validation_rows, _ = load_and_validate_data(args.data_dir)
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
    )
    with torch.inference_mode():
        logits = model((validation_features - fit["train_mean"]) / fit["train_std"])
        probabilities = torch.softmax(logits, dim=-1)
        predictions = logits.argmax(dim=-1)

    args.output_dir.mkdir(parents=True)
    head_path = args.output_dir / "best-readout.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "train_mean": fit["train_mean"],
            "train_std": fit["train_std"],
            "class_count": CLASS_COUNT,
            "best_epoch": fit["best_epoch"],
        },
        head_path,
    )
    history_path = args.output_dir / "training-history.jsonl"
    with history_path.open("w", encoding="utf-8", newline="\n") as stream:
        for item in fit["history"]:
            stream.write(json.dumps(item, sort_keys=True) + "\n")
    predictions_path = args.output_dir / "validation-predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8", newline="\n") as stream:
        for index, row in enumerate(validation_rows):
            record = {
                "sample_id": row["sample_id"],
                "content_group_id": row["content_group_id"],
                "target": int(validation_labels[index]),
                "prediction": int(predictions[index]),
                "probabilities": probabilities[index].tolist(),
            }
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
    metrics = {
        **fit["best_validation_metrics"],
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
    target_counts = Counter(int(row["label"]) for row in validation_rows)
    for label in range(CLASS_COUNT):
        mask = validation_labels.eq(label)
        metrics["per_class"][str(label)] = {
            "count": target_counts[label],
            "accuracy": float(predictions[mask].eq(validation_labels[mask]).float().mean()),
            "nll": nn.functional.cross_entropy(logits[mask], validation_labels[mask]).item(),
        }
    metrics_path = args.output_dir / "validation-metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    ended = datetime.now(UTC).isoformat()
    metadata = {
        "experiment_id": "banking77-minilm-l3-frozen-readout-v1",
        "model_id": "sentence-transformers/paraphrase-MiniLM-L3-v2",
        "model_revision": MODEL_REVISION,
        "model_weights_sha256": actual_model_hash,
        "split_manifest_sha256": sha256_file(args.data_dir / "split-manifest.json"),
        "base_encoder_parameters": 17389824,
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "train_unique_groups": len({row["content_group_id"] for row in train_rows}),
        "validation_unique_groups": len({row["content_group_id"] for row in validation_rows}),
        "official_test_loaded": False,
        "device": "cpu",
        "dtype": "float32",
        "seed": 17,
        "selection_metric": "minimum validation NLL",
        "best_epoch": fit["best_epoch"],
        "epochs_completed": fit["epochs_completed"],
        "started_at_utc": started,
        "ended_at_utc": ended,
        "elapsed_seconds": time.perf_counter() - time_start,
        "head_sha256": sha256_file(head_path),
        "metrics_sha256": sha256_file(metrics_path),
    }
    (args.output_dir / "run-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({**metadata, "validation_metrics": fit["best_validation_metrics"]}, indent=2))


if __name__ == "__main__":
    main()
