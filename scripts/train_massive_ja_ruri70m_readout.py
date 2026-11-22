"""Fit a small intent readout over frozen Ruri-v3-70m sentence features."""

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
    from scripts.train_massive_ja_bert_readout import classification_metrics, read_jsonl
else:
    from prepare_massive_ja_intents import INTENTS, sha256_file
    from train_massive_ja_bert_readout import classification_metrics, read_jsonl

MODEL_ID = "cl-nagoya/ruri-v3-70m"
MODEL_REVISION = "07a8b0aba47d29d2ca21f89b915c1efe2c23d1cc"
MODEL_SHA256 = "c959b76a535eee838b1f1e6b8999ba513ef8d87a6f8c3534c50dcda1eb5f6a62"
MODEL_PARAMETERS = 70_002_816


def encode_utterances(
    rows: list[dict[str, Any]], tokenizer: Any, encoder: nn.Module, batch_size: int
) -> Tensor:
    features = []
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
            output = encoder(**tokens).last_hidden_state.float()
            mask = tokens["attention_mask"].unsqueeze(-1).to(output.dtype)
            pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            pooled = F.normalize(pooled, dim=-1)
            if not torch.isfinite(pooled).all():
                raise ValueError("Ruri emitted non-finite pooled features")
            features.append(pooled.cpu())
    return torch.cat(features)


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
        raise ValueError("intent taxonomy does not match pinned MASSIVE schema")
    for path, key in (
        (args.train, "train_sha256"),
        (args.validation, "validation_sha256"),
    ):
        if sha256_file(path) != manifest[key]:
            raise ValueError(f"frozen data hash mismatch: {path}")
    if sha256_file(args.model / "model.safetensors") != MODEL_SHA256:
        raise ValueError("Ruri weights do not match pinned upstream revision")

    train, validation = read_jsonl(args.train), read_jsonl(args.validation)
    train_ids = {row["source_id"] for row in train}
    validation_ids = {row["source_id"] for row in validation}
    train_groups = {row["content_group_id"] for row in train}
    validation_groups = {row["content_group_id"] for row in validation}
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
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True)
    started = datetime.now(UTC).isoformat()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    encoder = AutoModel.from_pretrained(
        args.model, local_files_only=True, dtype=torch.float32
    ).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)

    feature_started = time.perf_counter()
    train_x = encode_utterances(train, tokenizer, encoder, args.encode_batch_size)
    validation_x = encode_utterances(validation, tokenizer, encoder, args.encode_batch_size)
    feature_seconds = time.perf_counter() - feature_started
    train_y = torch.tensor([row["target_index"] for row in train], dtype=torch.long)
    validation_y = torch.tensor([row["target_index"] for row in validation], dtype=torch.long)
    torch.save(
        {
            "train_features": train_x,
            "train_targets": train_y,
            "validation_features": validation_x,
            "validation_targets": validation_y,
        },
        output_dir / "frozen-features.pt",
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
    fit_started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        order = list(range(len(train)))
        order_rng.shuffle(order)
        readout.train()
        losses = []
        for start in range(0, len(order), args.batch_size):
            indices = torch.tensor(order[start : start + args.batch_size])
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(readout(train_x[indices]), train_y[indices])
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
    fit_seconds = time.perf_counter() - fit_started
    if best_state is None:
        raise RuntimeError("no validation-selected checkpoint")

    checkpoint = output_dir / "best-readout.pt"
    torch.save(best_state, checkpoint)
    readout.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    readout.eval()
    with torch.inference_mode():
        selected_logits = readout(validation_x)
    selected_metrics = classification_metrics(selected_logits, validation_y)
    probabilities = selected_logits.float().softmax(-1)
    predictions = []
    for row, probability in zip(validation, probabilities.tolist(), strict=True):
        prediction = max(range(len(INTENTS)), key=probability.__getitem__)
        predictions.append(
            {
                "sample_id": row["sample_id"],
                "source_id": row["source_id"],
                "content_group_id": row["content_group_id"],
                "target_index": row["target_index"],
                "target_intent": row["intent"],
                "prediction_index": prediction,
                "prediction_intent": INTENTS[prediction],
                "probabilities": probability,
            }
        )
    (output_dir / "validation-predictions.jsonl").write_text(
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
            "cross_split_source_id_overlap": 0,
            "cross_split_normalized_utterance_overlap": 0,
            "publisher_test_downloaded_or_opened": False,
        },
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "weights_sha256": MODEL_SHA256,
            "frozen_parameters": sum(parameter.numel() for parameter in encoder.parameters()),
            "weight_bytes": (args.model / "model.safetensors").stat().st_size,
            "declared_license": "Apache-2.0",
            "pooling": "attention-mask mean pooling followed by L2 normalization",
        },
        "readout": {
            "architecture": "frozen sentence embedding -> 128 tanh -> 60 intent logits",
            "trainable_parameters": trainable_parameters,
            "total_prequantization_parameters": MODEL_PARAMETERS + trainable_parameters,
            "optimizer": "AdamW(lr=0.002, weight_decay=0.0001)",
            "max_length": 128,
        },
        "history": history,
        "selected_validation_metrics": selected_metrics,
        "timing_seconds": {
            "feature_extraction": feature_seconds,
            "readout_fit": fit_seconds,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "device": "cpu",
        },
        "artifact_sha256": {
            name: sha256_file(output_dir / name)
            for name in (
                "frozen-features.pt",
                "best-readout.pt",
                "validation-predictions.jsonl",
            )
        },
        "checkpoint_reload_verified": True,
        "note": (
            "Same exposed validation IDs as the BERT-base comparison; publisher test "
            "and project sealed audit remained unopened."
        ),
    }
    (output_dir / "run-report.json").write_text(
        json.dumps(run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": run["status"],
                "selected_epoch_by_validation_nll": best_epoch,
                "selected_validation_metrics": selected_metrics,
                "timing_seconds": run["timing_seconds"],
                "artifact_sha256": run["artifact_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
