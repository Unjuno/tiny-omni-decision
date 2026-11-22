"""Train a compact three-way head over frozen Japanese BERT paired-text features."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import random
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import AutoModel, AutoTokenizer

MODEL_ID = "tohoku-nlp/bert-base-japanese-v3"
REVISION = "65243d6e5629b969c77309f217bd7b1a79d43c7e"
MODEL_WEIGHT_SHA256 = "e172862e0674054d65e0ba40d67df2a4687982f589db44aa27091c386e5450a4"
LABELS = ("entailment", "contradiction", "neutral")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics(logits: Tensor, targets: Tensor, bins: int = 15) -> dict[str, Any]:
    probabilities = logits.float().softmax(dim=-1)
    predictions = probabilities.argmax(dim=-1)
    correct = predictions.eq(targets)
    confidence = probabilities.max(dim=-1).values
    ece = torch.zeros((), dtype=torch.float32)
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        mask = (confidence >= low) & (confidence < high if index < bins - 1 else confidence <= high)
        if mask.any():
            ece += (
                mask.float().mean() * (correct[mask].float().mean() - confidence[mask].mean()).abs()
            )
    one_hot = F.one_hot(targets, num_classes=logits.shape[-1]).float()
    return {
        "count": int(targets.numel()),
        "accuracy": float(correct.float().mean()),
        "nll": float(F.cross_entropy(logits.float(), targets)),
        "brier": float((probabilities - one_hot).square().sum(dim=-1).mean()),
        "ece_15_bins": float(ece),
        "per_class": {
            label: {
                "count": int((targets == index).sum()),
                "accuracy": (
                    float(correct[targets == index].float().mean())
                    if (targets == index).any()
                    else None
                ),
            }
            for index, label in enumerate(LABELS)
        },
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def args_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--encode-batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    return parser.parse_args()


def encode_pairs(
    rows: list[dict[str, Any]], tokenizer: Any, encoder: nn.Module, batch_size: int, max_length: int
) -> Tensor:
    pair_features = []
    encoder.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            encoded = tokenizer(
                [row["sentence1"] for row in batch],
                [row["sentence2"] for row in batch],
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            )
            output = encoder(**encoded).last_hidden_state[:, 0].float()
            if not torch.isfinite(output).all():
                raise ValueError("BERT returned non-finite paired features")
            pair_features.append(output.cpu())
    # Joint pair encoding lets BERT's attention represent premise-hypothesis relations.
    return torch.cat(pair_features)


def main() -> None:
    args = args_parser()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    model_dir = args.model.resolve()
    if model_dir.is_symlink():
        raise ValueError("model path must be a concrete local directory")
    if sha256_file(model_dir / "pytorch_model.bin") != MODEL_WEIGHT_SHA256:
        raise ValueError("BERT weights do not match pinned upstream blob")
    split = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    if split["official_test_downloaded_or_opened"]:
        raise ValueError("official test access is prohibited")
    for path, key in (
        (args.train, "train_jsonl_sha256"),
        (args.validation, "validation_jsonl_sha256"),
    ):
        if sha256_file(path) != split[key]:
            raise ValueError(f"data hash mismatch: {path}")
    train, validation = read_jsonl(args.train), read_jsonl(args.validation)
    train_groups = {row["content_group_id"] for row in train}
    valid_groups = {row["content_group_id"] for row in validation}
    train_images = {item for row in train for item in row["image_ids"]}
    valid_images = {item for row in validation for item in row["image_ids"]}
    if train_groups & valid_groups or train_images & valid_images:
        raise ValueError("group or source-image leakage detected")
    if any(row["label"] not in LABELS for row in train + validation):
        raise ValueError("invalid label")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = torch.device("cpu")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True)
    started = datetime.now(UTC).isoformat()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    encoder = AutoModel.from_pretrained(
        model_dir, local_files_only=True, dtype=torch.float32
    ).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    model_started = time.perf_counter()
    # BERT pair input uses joint encoding; retain one [CLS] vector per paired example.
    train_x = encode_pairs(train, tokenizer, encoder, args.encode_batch_size, args.max_length)
    valid_x = encode_pairs(validation, tokenizer, encoder, args.encode_batch_size, args.max_length)
    feature_seconds = time.perf_counter() - model_started
    train_y = torch.tensor([LABELS.index(row["label"]) for row in train])
    valid_y = torch.tensor([LABELS.index(row["label"]) for row in validation])
    torch.save(
        {
            "train_cls": train_x,
            "train_targets": train_y,
            "validation_cls": valid_x,
            "validation_targets": valid_y,
        },
        output_dir / "frozen-features.pt",
    )
    scorer = nn.Sequential(
        nn.Linear(train_x.shape[-1], 128), nn.Tanh(), nn.Dropout(0.1), nn.Linear(128, 3)
    )
    trainable_parameters = sum(parameter.numel() for parameter in scorer.parameters())
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=2e-3, weight_decay=1e-4)
    order_rng = random.Random(args.seed)
    best_nll, best_epoch, best_state = float("inf"), 0, None
    history = []
    training_started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        order = list(range(len(train_x)))
        order_rng.shuffle(order)
        scorer.train()
        losses = []
        for start in range(0, len(order), args.batch_size):
            indices = torch.tensor(order[start : start + args.batch_size])
            optimizer.zero_grad(set_to_none=True)
            logits = scorer(train_x[indices])
            loss = F.cross_entropy(logits, train_y[indices])
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        scorer.eval()
        with torch.inference_mode():
            valid_logits = scorer(valid_x)
        val_metrics = metrics(valid_logits, valid_y)
        history.append({"epoch": epoch, "train_ce": sum(losses) / len(losses), **val_metrics})
        if val_metrics["nll"] < best_nll:
            best_nll, best_epoch = val_metrics["nll"], epoch
            best_state = {key: value.detach().clone() for key, value in scorer.state_dict().items()}
    training_seconds = time.perf_counter() - training_started
    if best_state is None:
        raise RuntimeError("no validation-selected checkpoint")
    scorer.load_state_dict(best_state)
    scorer.eval()
    with torch.inference_mode():
        selected_logits = scorer(valid_x)
    selected_metrics = metrics(selected_logits, valid_y)
    torch.save(best_state, output_dir / "best-readout.pt")
    predictions = []
    probs = selected_logits.float().softmax(-1)
    for row, probability, target in zip(validation, probs.tolist(), valid_y.tolist(), strict=True):
        prediction = max(range(3), key=probability.__getitem__)
        predictions.append(
            {
                "sample_id": row["sample_id"],
                "content_group_id": row["content_group_id"],
                "target": LABELS[target],
                "prediction": LABELS[prediction],
                "probabilities": dict(zip(LABELS, probability, strict=True)),
            }
        )
    pred_path = output_dir / "validation-predictions.jsonl"
    pred_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions), encoding="utf-8"
    )
    report = {
        "status": "complete_development_validation_only",
        "task": "Japanese NLI (three-way paired sentence reasoning)",
        "started_at_utc": started,
        "ended_at_utc": datetime.now(UTC).isoformat(),
        "seed": args.seed,
        "epochs": args.epochs,
        "best_epoch_by_validation_nll": best_epoch,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "encode_batch_size": args.encode_batch_size,
        "data": {
            "train_sha256": sha256_file(args.train),
            "validation_sha256": sha256_file(args.validation),
            "split_manifest_sha256": sha256_file(args.split_manifest),
            "train_rows": len(train),
            "validation_rows": len(validation),
            "train_groups": len(train_groups),
            "validation_groups": len(valid_groups),
            "cross_split_group_overlap": 0,
            "cross_split_image_overlap": 0,
            "official_test_downloaded_or_opened": False,
        },
        "model": {
            "id": MODEL_ID,
            "revision": REVISION,
            "weights_sha256": sha256_file(model_dir / "pytorch_model.bin"),
            "weight_bytes": (model_dir / "pytorch_model.bin").stat().st_size,
            "license": "Apache-2.0",
            "frozen_parameters": sum(parameter.numel() for parameter in encoder.parameters()),
            "architecture": {
                "layers": encoder.config.num_hidden_layers,
                "hidden_size": encoder.config.hidden_size,
            },
        },
        "readout": {
            "architecture": "128-hidden tanh MLP on jointly encoded pair [CLS]",
            "trainable_parameters": trainable_parameters,
            "total_prequantization_parameters": sum(
                parameter.numel() for parameter in encoder.parameters()
            )
            + trainable_parameters,
        },
        "optimizer": "AdamW(lr=0.002, weight_decay=0.0001)",
        "history": history,
        "selected_validation_metrics": selected_metrics,
        "timing": {
            "feature_extraction_seconds": feature_seconds,
            "readout_training_seconds": training_seconds,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "fugashi": importlib.metadata.version("fugashi"),
            "unidic-lite": importlib.metadata.version("unidic-lite"),
            "device": str(device),
        },
        "artifacts": {
            name: sha256_file(output_dir / name)
            for name in ("frozen-features.pt", "best-readout.pt", "validation-predictions.jsonl")
        },
        "note": (
            "Checkpoint selection used grouped development validation. This is not blind "
            "or multimodal product-quality evidence."
        ),
    }
    (output_dir / "run-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
