"""Train a frozen MiniLM linear readout on grouped CLINC150 development data."""

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

CLASS_COUNT = 151
CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs/pretrained_reuse/clinc150_minilm_frozen_readout_v1.json"
)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def validate_data(
    data_dir: Path, config: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    manifest_path = data_dir / "split-manifest.json"
    manifest_hash = file_hash(manifest_path)
    if manifest_hash != config["development_split"]["manifest_sha256"]:
        raise ValueError("Grouped CLINC development manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("official_test_downloaded_or_opened") is not False:
        raise ValueError("Refusing a split that opened the official test")
    train, validation = (
        read_jsonl(data_dir / "train.jsonl"),
        read_jsonl(data_dir / "validation.jsonl"),
    )
    expected = config["development_split"]
    if len(train) != expected["train_rows"] or len(validation) != expected["validation_rows"]:
        raise ValueError("Grouped CLINC row count mismatch")
    if file_hash(data_dir / "train.jsonl") != manifest["train_jsonl_sha256"]:
        raise ValueError("Train data hash mismatch")
    if file_hash(data_dir / "validation.jsonl") != manifest["validation_jsonl_sha256"]:
        raise ValueError("Validation data hash mismatch")
    train_groups = {row["content_group_id"] for row in train}
    validation_groups = {row["content_group_id"] for row in validation}
    if train_groups & validation_groups:
        raise ValueError("Normalized content group overlap detected")
    for rows in (train, validation):
        if {int(row["label"]) for row in rows} != set(range(CLASS_COUNT)):
            raise ValueError("Split does not contain all 151 CLINC classes")
    return train, validation, manifest_hash


def extract_features(rows: list[dict[str, Any]], model_dir: Path) -> torch.Tensor:
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    encoder = AutoModel.from_pretrained(model_dir, local_files_only=True).cpu().eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    features: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(rows), 64):
            batch = tokenizer(
                [row["text"] for row in rows[start : start + 64]],
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            output = encoder(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).to(output.dtype)
            features.append(
                ((output * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)).cpu().float()
            )
    return torch.cat(features)


def metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    probs = torch.softmax(logits, dim=-1)
    pred = probs.argmax(dim=-1)
    correct = pred.eq(labels)
    recalls, f1s = [], []
    for label in range(CLASS_COUNT):
        positive, chosen = labels.eq(label), pred.eq(label)
        tp = (positive & chosen).sum().item()
        recall = tp / positive.sum().item() if positive.any() else 0.0
        precision = tp / chosen.sum().item() if chosen.any() else 0.0
        recalls.append(recall)
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    one_hot = nn.functional.one_hot(labels, CLASS_COUNT).to(probs.dtype)
    confidence = probs.max(dim=-1).values
    ece = 0.0
    for index in range(15):
        mask = (confidence > index / 15) & (confidence <= (index + 1) / 15)
        if mask.any():
            ece += mask.float().mean().item() * abs(
                confidence[mask].mean().item() - correct[mask].float().mean().item()
            )
    return {
        "accuracy": correct.float().mean().item(),
        "macro_recall": sum(recalls) / CLASS_COUNT,
        "macro_f1": sum(f1s) / CLASS_COUNT,
        "nll": nn.functional.cross_entropy(logits, labels).item(),
        "brier": (probs - one_hot).square().sum(dim=-1).mean().item(),
        "ece_15": ece,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=("minilm_l6", "minilm_l12"), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output_dir}")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    candidate = config["model_candidates"][args.candidate]
    for relative, expected in candidate["files_sha256"].items():
        if file_hash(args.model_dir / relative) != expected:
            raise ValueError(f"Pinned model file hash mismatch: {relative}")
    train_rows, val_rows, split_hash = validate_data(args.data_dir, config)
    started_at = datetime.now(UTC).isoformat()
    start = time.perf_counter()
    train_features = extract_features(train_rows, args.model_dir)
    val_features = extract_features(val_rows, args.model_dir)
    train_labels = torch.tensor([int(row["label"]) for row in train_rows])
    val_labels = torch.tensor([int(row["label"]) for row in val_rows])
    seed = config["training"]["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    mean = train_features.mean(dim=0)
    std = train_features.std(dim=0, unbiased=False).clamp_min(1e-5)
    x_train, x_val = (train_features - mean) / std, (val_features - mean) / std
    model = nn.Linear(x_train.shape[1], CLASS_COUNT)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(x_train, train_labels),
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        generator=generator,
    )
    history, best_state, best_nll, best_epoch, stale = [], None, float("inf"), 0, 0
    best_metrics: dict[str, float] | None = None
    for epoch in range(1, config["training"]["max_epochs"] + 1):
        model.train()
        losses = []
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        model.eval()
        with torch.inference_mode():
            val_logits = model(x_val)
            val_metrics = metrics(val_logits, val_labels)
        history.append(
            {"epoch": epoch, "train_ce": sum(losses) / len(losses), "validation": val_metrics}
        )
        if val_metrics["nll"] < best_nll - config["training"]["early_stopping_min_delta"]:
            best_nll, best_epoch, best_metrics, stale = val_metrics["nll"], epoch, val_metrics, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= config["training"]["early_stopping_patience"]:
            break
    if best_state is None or best_metrics is None:
        raise RuntimeError("No validation checkpoint selected")
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        logits = model(x_val)
        probabilities = torch.softmax(logits, dim=-1)
        predictions = logits.argmax(dim=-1)
    args.output_dir.mkdir(parents=True)
    head = args.output_dir / "best-readout.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "train_mean": mean,
            "train_std": std,
            "class_count": CLASS_COUNT,
            "best_epoch": best_epoch,
        },
        head,
    )
    with (args.output_dir / "training-history.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as stream:
        for item in history:
            stream.write(json.dumps(item, sort_keys=True) + "\n")
    with (args.output_dir / "validation-predictions.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as stream:
        for i, row in enumerate(val_rows):
            stream.write(
                json.dumps(
                    {
                        "sample_id": row["sample_id"],
                        "content_group_id": row["content_group_id"],
                        "target": int(val_labels[i]),
                        "prediction": int(predictions[i]),
                        "probabilities": probabilities[i].tolist(),
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
    label_counts = Counter(int(row["label"]) for row in val_rows)
    per_class = {}
    for label in range(CLASS_COUNT):
        mask = val_labels.eq(label)
        per_class[str(label)] = {
            "count": label_counts[label],
            "accuracy": predictions[mask].eq(val_labels[mask]).float().mean().item(),
            "nll": nn.functional.cross_entropy(logits[mask], val_labels[mask]).item(),
        }
    result = {
        **best_metrics,
        "validation_rows": len(val_rows),
        "class_count": CLASS_COUNT,
        "oos": {
            "count": label_counts[150],
            "accuracy": per_class["150"]["accuracy"],
            "nll": per_class["150"]["nll"],
        },
        "per_class": per_class,
    }
    metrics_path = args.output_dir / "validation-metrics.json"
    metrics_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    ended_at = datetime.now(UTC).isoformat()
    metadata = {
        "experiment_id": config["experiment_id"],
        "candidate": args.candidate,
        "model_id": candidate["id"],
        "model_revision": candidate["revision"],
        "model_weight_sha256": candidate["files_sha256"]["model.safetensors"],
        "split_manifest_sha256": split_hash,
        "base_encoder_parameters": candidate["parameters"],
        "trainable_parameters": trainable,
        "train_rows": len(train_rows),
        "validation_rows": len(val_rows),
        "official_test_loaded": False,
        "device": "cpu",
        "dtype": "float32",
        "seed": seed,
        "selection_metric": "minimum validation multiclass NLL",
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "started_at_utc": started_at,
        "ended_at_utc": ended_at,
        "elapsed_seconds": time.perf_counter() - start,
        "head_sha256": file_hash(head),
        "metrics_sha256": file_hash(metrics_path),
    }
    (args.output_dir / "run-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({**metadata, "validation_metrics": result}, indent=2))


if __name__ == "__main__":
    main()
