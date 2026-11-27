"""Fit a linear readout on a pinned encoder and frozen near-duplicate split."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_state_hash(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def load_and_validate_data(
    data_dir: Path, config: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    split_config = config["development_split"]
    manifest_path = data_dir / "split-manifest.json"
    manifest_hash = file_hash(manifest_path)
    if manifest_hash != split_config["manifest_sha256"]:
        raise ValueError("Near-duplicate split manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["protocol_id"] != split_config["protocol_id"]:
        raise ValueError("Unexpected near-duplicate split protocol")
    if manifest.get("official_test_loaded") is not False:
        raise ValueError("Refusing data split that loaded official test records")
    if manifest.get("sealed_audit_loaded") is not False:
        raise ValueError("Refusing data split that loaded sealed-audit records")

    train = read_jsonl(data_dir / "train.jsonl")
    validation = read_jsonl(data_dir / "validation.jsonl")
    if len(train) != split_config["train_rows"]:
        raise ValueError("Near-duplicate train row count mismatch")
    if len(validation) != split_config["validation_rows"]:
        raise ValueError("Near-duplicate validation row count mismatch")
    for split, rows, expected_hash in (
        ("train", train, split_config["train_sha256"]),
        ("validation", validation, split_config["validation_sha256"]),
    ):
        if file_hash(data_dir / f"{split}.jsonl") != expected_hash:
            raise ValueError(f"Near-duplicate {split} file hash mismatch")
        if any(not row.get("near_duplicate_group_id") for row in rows):
            raise ValueError(f"Near-duplicate group ID missing in {split}")

    train_groups = {row["near_duplicate_group_id"] for row in train}
    validation_groups = {row["near_duplicate_group_id"] for row in validation}
    if train_groups & validation_groups:
        raise ValueError("Near-duplicate group overlap detected")
    class_count = split_config["class_count"]
    expected_labels = set(range(class_count))
    for rows in (train, validation):
        if {int(row["label"]) for row in rows} != expected_labels:
            raise ValueError("Near-duplicate split does not contain every class")
    return train, validation, manifest_hash


def extract_features(
    rows: list[dict[str, Any]], model_dir: Path, max_length: int
) -> tuple[torch.Tensor, dict[str, int]]:
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    encoder = AutoModel.from_pretrained(model_dir, local_files_only=True).cpu().eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    before_hash = tensor_state_hash(encoder)
    token_counts = [
        len(tokenizer(row["text"], truncation=False)["input_ids"]) for row in rows
    ]
    features: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(rows), 64):
            batch = tokenizer(
                [row["text"] for row in rows[start : start + 64]],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            output = encoder(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).to(output.dtype)
            features.append(
                ((output * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)).cpu().float()
            )
    after_hash = tensor_state_hash(encoder)
    if before_hash != after_hash:
        raise RuntimeError("Frozen MiniLM encoder weights changed during feature extraction")
    if max(token_counts) > max_length:
        raise ValueError(
            f"Found a candidate truncated by max_length={max_length}: {max(token_counts)} tokens"
        )
    return torch.cat(features), {
        "token_count_max": max(token_counts),
        "token_count_over_max_length": sum(count > max_length for count in token_counts),
    }


def metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    probs = torch.softmax(logits, dim=-1)
    pred = probs.argmax(dim=-1)
    correct = pred.eq(labels)
    recalls, f1s = [], []
    for label in range(logits.shape[-1]):
        positive, chosen = labels.eq(label), pred.eq(label)
        tp = (positive & chosen).sum().item()
        recall = tp / positive.sum().item() if positive.any() else 0.0
        precision = tp / chosen.sum().item() if chosen.any() else 0.0
        recalls.append(recall)
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    one_hot = nn.functional.one_hot(labels, logits.shape[-1]).to(probs.dtype)
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
        "macro_recall": sum(recalls) / len(recalls),
        "macro_f1": sum(f1s) / len(f1s),
        "nll": nn.functional.cross_entropy(logits, labels).item(),
        "brier": (probs - one_hot).square().sum(dim=-1).mean().item(),
        "ece_15": ece,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite run output: {args.output_dir}")
    config_hash = file_hash(args.config)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    candidate = config["model"]
    for relative, expected in candidate["files_sha256"].items():
        if file_hash(args.model_dir / relative) != expected:
            raise ValueError(f"Pinned MiniLM file hash mismatch: {relative}")
    train, validation, manifest_hash = load_and_validate_data(args.data_dir, config)

    started_at = datetime.now(UTC).isoformat()
    started_clock = time.perf_counter()
    max_length = config["training"]["max_length"]
    train_features, train_token_audit = extract_features(train, args.model_dir, max_length)
    validation_features, validation_token_audit = extract_features(
        validation, args.model_dir, max_length
    )
    train_labels = torch.tensor([int(row["label"]) for row in train])
    validation_labels = torch.tensor([int(row["label"]) for row in validation])
    training = config["training"]
    seed = int(training["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    mean = train_features.mean(dim=0)
    std = train_features.std(dim=0, unbiased=False).clamp_min(1e-5)
    x_train, x_validation = (train_features - mean) / std, (validation_features - mean) / std
    head = nn.Linear(x_train.shape[1], config["readout"]["output_classes"])
    trainable_parameters = sum(parameter.numel() for parameter in head.parameters())
    if trainable_parameters != config["readout"]["trainable_parameters"]:
        raise ValueError("Unexpected readout parameter count")
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=training["learning_rate"],
        weight_decay=training["weight_decay"],
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(x_train, train_labels),
        batch_size=training["batch_size"],
        shuffle=True,
        generator=generator,
    )
    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_nll, best_epoch, stale = float("inf"), 0, 0
    best_metrics: dict[str, float] | None = None
    for epoch in range(1, training["max_epochs"] + 1):
        head.train()
        losses = []
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(head(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        head.eval()
        with torch.inference_mode():
            validation_logits = head(x_validation)
            validation_metrics = metrics(validation_logits, validation_labels)
        history.append(
            {
                "epoch": epoch,
                "train_ce": sum(losses) / len(losses),
                "validation": validation_metrics,
            }
        )
        if validation_metrics["nll"] < best_nll - training["early_stopping_min_delta"]:
            best_nll, best_epoch, best_metrics, stale = (
                validation_metrics["nll"],
                epoch,
                validation_metrics,
                0,
            )
            best_state = {key: value.detach().clone() for key, value in head.state_dict().items()}
        else:
            stale += 1
        if stale >= training["early_stopping_patience"]:
            break
    if best_state is None or best_metrics is None:
        raise RuntimeError("No validation-selected checkpoint")
    head.load_state_dict(best_state)
    head.eval()
    with torch.inference_mode():
        logits = head(x_validation)
        probabilities = torch.softmax(logits, dim=-1)
        predictions = logits.argmax(dim=-1)

    args.output_dir.mkdir(parents=True)
    head_path = args.output_dir / "best-readout.pt"
    torch.save(
        {
            "state_dict": head.state_dict(),
            "train_mean": mean,
            "train_std": std,
            "class_count": config["readout"]["output_classes"],
            "best_epoch": best_epoch,
        },
        head_path,
    )
    history_path = args.output_dir / "training-history.jsonl"
    with history_path.open("w", encoding="utf-8", newline="\n") as stream:
        for item in history:
            stream.write(json.dumps(item, sort_keys=True) + "\n")
    prediction_path = args.output_dir / "validation-predictions.jsonl"
    with prediction_path.open("w", encoding="utf-8", newline="\n") as stream:
        for index, row in enumerate(validation):
            stream.write(
                json.dumps(
                    {
                        "sample_id": row["sample_id"],
                        "near_duplicate_group_id": row["near_duplicate_group_id"],
                        "label": int(validation_labels[index]),
                        "prediction": int(predictions[index]),
                        "probabilities": probabilities[index].tolist(),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    ended_at = datetime.now(UTC).isoformat()
    metadata = {
        "experiment_id": config["experiment_id"],
        "config_sha256": config_hash,
        "source_commit": "64a499ee877d2d651b5cd2801cc0a72120cc9151",
        "source": config["source"],
        "candidate": candidate,
        "data_manifest_sha256": manifest_hash,
        "split": config["development_split"],
        "initialization": (
            f"pinned pretrained frozen encoder {candidate['id']}; new random linear readout"
        ),
        "base_encoder_parameters": candidate["parameters"],
        "trainable_parameters": trainable_parameters,
        "encoder_frozen": True,
        "encoder_state_unchanged": True,
        "training": training,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "train_unique_near_duplicate_groups": len(
            {row["near_duplicate_group_id"] for row in train}
        ),
        "validation_unique_near_duplicate_groups": len(
            {row["near_duplicate_group_id"] for row in validation}
        ),
        "tokenization": {
            "train": train_token_audit,
            "validation": validation_token_audit,
            "truncated_rows": 0,
        },
        "selected_epoch": best_epoch,
        "epochs_completed": len(history),
        "selected_validation_metrics": best_metrics,
        "started_at_utc": started_at,
        "ended_at_utc": ended_at,
        "elapsed_seconds": time.perf_counter() - started_clock,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "torch_threads": torch.get_num_threads(),
        "files_sha256": {
            "best-readout.pt": file_hash(head_path),
            "training-history.jsonl": file_hash(history_path),
            "validation-predictions.jsonl": file_hash(prediction_path),
        },
        "official_test_loaded": False,
        "sealed_audit_loaded": False,
    }
    metadata_path = args.output_dir / "run-metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
