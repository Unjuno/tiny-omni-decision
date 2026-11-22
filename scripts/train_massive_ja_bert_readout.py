"""Fit a small Japanese intent readout over a frozen BERT-base representation."""

from __future__ import annotations

import argparse
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

if __package__:
    from scripts.prepare_massive_ja_intents import INTENTS, sha256_file
else:
    from prepare_massive_ja_intents import INTENTS, sha256_file

MODEL_ID = "tohoku-nlp/bert-base-japanese-v3"
MODEL_REVISION = "65243d6e5629b969c77309f217bd7b1a79d43c7e"
MODEL_SHA256 = "e172862e0674054d65e0ba40d67df2a4687982f589db44aa27091c386e5450a4"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def classification_metrics(logits: Tensor, targets: Tensor, bins: int = 15) -> dict[str, Any]:
    probs = logits.float().softmax(-1)
    predictions = probs.argmax(-1)
    correct = predictions.eq(targets)
    confidence = probs.max(-1).values
    one_hot = F.one_hot(targets, len(INTENTS)).float()
    ece = torch.zeros((), dtype=torch.float32)
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        mask = (confidence >= low) & (
            confidence < high if index < bins - 1 else confidence <= high
        )
        if mask.any():
            ece += mask.float().mean() * (
                correct[mask].float().mean() - confidence[mask].mean()
            ).abs()
    per_intent = {}
    for index, intent in enumerate(INTENTS):
        mask = targets == index
        per_intent[intent] = {
            "count": int(mask.sum()),
            "accuracy": float(correct[mask].float().mean()) if mask.any() else None,
        }
    recalls = [entry["accuracy"] for entry in per_intent.values() if entry["accuracy"] is not None]
    return {
        "count": int(targets.numel()),
        "accuracy": float(correct.float().mean()),
        "macro_accuracy_present_intents": sum(recalls) / len(recalls),
        "intents_present": len(recalls),
        "nll": float(F.cross_entropy(logits.float(), targets)),
        "brier": float((probs - one_hot).square().sum(-1).mean()),
        "ece_15_bins": float(ece),
        "mean_confidence": float(confidence.mean()),
        "per_intent": per_intent,
    }


def encode_utterances(
    rows: list[dict[str, Any]], tokenizer: Any, encoder: nn.Module, batch_size: int
) -> Tensor:
    result = []
    encoder.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            tokens = tokenizer(
                [row["utterance"] for row in batch],
                truncation=True,
                max_length=128,
                padding=True,
                return_tensors="pt",
            )
            cls = encoder(**tokens).last_hidden_state[:, 0].float()
            if not torch.isfinite(cls).all():
                raise ValueError("BERT emitted non-finite frozen features")
            result.append(cls.cpu())
    return torch.cat(result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--encode-batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    if manifest["publisher_test_downloaded_or_opened"]:
        raise ValueError("publisher test access is prohibited")
    if manifest["intent_names"] != list(INTENTS):
        raise ValueError("intent taxonomy does not match the pinned MASSIVE schema")
    if sha256_file(args.train) != manifest["train_sha256"]:
        raise ValueError("train hash does not match frozen split manifest")
    if sha256_file(args.validation) != manifest["validation_sha256"]:
        raise ValueError("validation hash does not match frozen split manifest")
    if sha256_file(args.model / "pytorch_model.bin") != MODEL_SHA256:
        raise ValueError("base weights do not match pinned Japanese BERT revision")
    train, validation = read_jsonl(args.train), read_jsonl(args.validation)
    train_ids, validation_ids = ({row["source_id"] for row in rows} for rows in (train, validation))
    train_groups, validation_groups = (
        {row["content_group_id"] for row in rows} for rows in (train, validation)
    )
    if train_ids & validation_ids or train_groups & validation_groups:
        raise ValueError("source-ID or normalized-utterance overlap detected")
    if any(
        not 0 <= row["target_index"] < len(INTENTS)
        or INTENTS[row["target_index"]] != row["intent"]
        for row in train + validation
    ):
        raise ValueError("intent target/name mismatch")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    output = args.output_dir.resolve()
    output.mkdir(parents=True)
    started = datetime.now(UTC).isoformat()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    encoder = AutoModel.from_pretrained(
        args.model, local_files_only=True, dtype=torch.float32
    ).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    feature_start = time.perf_counter()
    train_x = encode_utterances(train, tokenizer, encoder, args.encode_batch_size)
    validation_x = encode_utterances(validation, tokenizer, encoder, args.encode_batch_size)
    feature_seconds = time.perf_counter() - feature_start
    train_y = torch.tensor([row["target_index"] for row in train], dtype=torch.long)
    validation_y = torch.tensor([row["target_index"] for row in validation], dtype=torch.long)
    cache_path = output / "frozen-features.pt"
    torch.save(
        {"train_features": train_x, "train_targets": train_y,
         "validation_features": validation_x, "validation_targets": validation_y},
        cache_path,
    )

    readout = nn.Sequential(
        nn.Linear(int(encoder.config.hidden_size), 128),
        nn.Tanh(),
        nn.Dropout(0.1),
        nn.Linear(128, len(INTENTS)),
    )
    trainable_parameters = sum(parameter.numel() for parameter in readout.parameters())
    optimizer = torch.optim.AdamW(readout.parameters(), lr=2e-3, weight_decay=1e-4)
    order_rng = random.Random(args.seed)
    best_nll, best_epoch, best_state = float("inf"), 0, None
    history = []
    fit_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        order = list(range(len(train)))
        order_rng.shuffle(order)
        readout.train()
        losses = []
        for start in range(0, len(order), args.batch_size):
            ids = torch.tensor(order[start : start + args.batch_size])
            optimizer.zero_grad(set_to_none=True)
            logits = readout(train_x[ids])
            loss = F.cross_entropy(logits, train_y[ids])
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        readout.eval()
        with torch.inference_mode():
            validation_logits = readout(validation_x)
        metrics = classification_metrics(validation_logits, validation_y)
        history.append({"epoch": epoch, "train_ce": sum(losses) / len(losses), **metrics})
        if metrics["nll"] < best_nll:
            best_nll, best_epoch = metrics["nll"], epoch
            best_state = {
                key: value.detach().clone() for key, value in readout.state_dict().items()
            }
    fit_seconds = time.perf_counter() - fit_start
    if best_state is None:
        raise RuntimeError("no validation-selected checkpoint")
    checkpoint = output / "best-readout.pt"
    torch.save(best_state, checkpoint)
    reloaded_state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    readout.load_state_dict(reloaded_state)
    readout.eval()
    with torch.inference_mode():
        selected_logits = readout(validation_x)
    selected_metrics = classification_metrics(selected_logits, validation_y)
    probs = selected_logits.float().softmax(-1)
    predictions = []
    for row, probability in zip(validation, probs.tolist(), strict=True):
        prediction = max(range(len(INTENTS)), key=probability.__getitem__)
        predictions.append({
            "sample_id": row["sample_id"],
            "source_id": row["source_id"],
            "content_group_id": row["content_group_id"],
            "target_index": row["target_index"],
            "target_intent": row["intent"],
            "prediction_index": prediction,
            "prediction_intent": INTENTS[prediction],
            "probabilities": probability,
        })
    predictions_path = output / "validation-predictions.jsonl"
    predictions_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions),
        encoding="utf-8",
    )
    run = {
        "status": "complete_development_validation_only",
        "task": "Japanese 60-intent voice-assistant decision",
        "started_at_utc": started,
        "ended_at_utc": datetime.now(UTC).isoformat(),
        "seed": args.seed,
        "epochs": args.epochs,
        "selected_epoch_by_validation_nll": best_epoch,
        "data": {
            "train_sha256": sha256_file(args.train),
            "validation_sha256": sha256_file(args.validation),
            "split_manifest_sha256": sha256_file(args.split_manifest),
            "train_rows": len(train),
            "validation_rows": len(validation),
            "train_source_ids": len(train_ids),
            "validation_source_ids": len(validation_ids),
            "cross_split_source_id_overlap": 0,
            "cross_split_normalized_utterance_overlap": 0,
            "publisher_test_downloaded_or_opened": False,
        },
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "weights_sha256": MODEL_SHA256,
            "frozen_parameters": sum(parameter.numel() for parameter in encoder.parameters()),
            "weight_bytes": (args.model / "pytorch_model.bin").stat().st_size,
            "declared_license": "Apache-2.0",
        },
        "readout": {
            "architecture": "frozen joint text [CLS] -> 128 tanh -> 60 intent logits",
            "trainable_parameters": trainable_parameters,
            "total_prequantization_parameters": sum(
                parameter.numel() for parameter in encoder.parameters()
            ) + trainable_parameters,
            "optimizer": "AdamW(lr=0.002, weight_decay=0.0001)",
            "max_length": 128,
        },
        "history": history,
        "selected_validation_metrics": selected_metrics,
        "timing_seconds": {"feature_extraction": feature_seconds, "readout_fit": fit_seconds},
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "fugashi": importlib.metadata.version("fugashi"),
            "unidic-lite": importlib.metadata.version("unidic-lite"),
            "device": "cpu",
        },
        "artifact_sha256": {
            name: sha256_file(output / name)
            for name in ("frozen-features.pt", "best-readout.pt", "validation-predictions.jsonl")
        },
        "checkpoint_reload_verified": True,
        "note": (
            "Validation selected the checkpoint; publisher test and project sealed data "
            "remained unopened."
        ),
    }
    (output / "run-report.json").write_text(
        json.dumps(run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(run, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
