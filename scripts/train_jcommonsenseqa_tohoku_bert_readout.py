"""Fit a small option scorer over frozen Japanese BERT paired-text features."""

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
    from scripts.prepare_jcommonsenseqa_grouped_split import (
        sha256_file,
        target_text_is_ambiguous,
    )
    from scripts.train_jnli_tohoku_bert_readout import MODEL_ID, REVISION
else:
    from prepare_jcommonsenseqa_grouped_split import sha256_file, target_text_is_ambiguous
    from train_jnli_tohoku_bert_readout import MODEL_ID, REVISION


def parse_args() -> argparse.Namespace:
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
    parser.add_argument("--max-length", type=int, default=192)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def encode_options(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    encoder: nn.Module,
    batch_size: int,
    max_length: int,
) -> Tensor:
    pairs = [(row["question"], row[f"choice{choice}"]) for row in rows for choice in range(5)]
    chunks = []
    encoder.eval()
    with torch.inference_mode():
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            encoded = tokenizer(
                [pair[0] for pair in batch],
                [pair[1] for pair in batch],
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            )
            cls = encoder(**encoded).last_hidden_state[:, 0].float()
            if not torch.isfinite(cls).all():
                raise ValueError("BERT emitted non-finite option features")
            chunks.append(cls.cpu())
    return torch.cat(chunks).reshape(len(rows), 5, -1)


def choice_metrics(logits: Tensor, targets: Tensor, bins: int = 15) -> dict[str, Any]:
    probabilities = logits.float().softmax(dim=-1)
    confidence = probabilities.max(dim=-1).values
    correct = logits.argmax(dim=-1).eq(targets)
    ece = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        mask = (confidence >= low) & (confidence < high if index < bins - 1 else confidence <= high)
        if mask.any():
            ece += float(mask.float().mean()) * abs(
                float(correct[mask].float().mean() - confidence[mask].mean())
            )
    one_hot = F.one_hot(targets, num_classes=logits.shape[-1]).float()
    return {
        "count": int(targets.numel()),
        "accuracy": float(correct.float().mean()),
        "nll": float(F.cross_entropy(logits.float(), targets)),
        "brier": float((probabilities - one_hot).square().sum(-1).mean()),
        "ece_15_bins": ece,
        "mean_confidence": float(confidence.mean()),
        "expected_random_accuracy": 0.2,
        "macro_accuracy_by_target_position": sum(
            float(correct[targets == index].float().mean())
            for index in range(5)
            if (targets == index).any()
        )
        / sum(bool((targets == index).any()) for index in range(5)),
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    split = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    if split["official_test_downloaded_or_opened"]:
        raise ValueError("official test access is prohibited")
    for path, key in (
        (args.train, "train_jsonl_sha256"),
        (args.validation, "validation_jsonl_sha256"),
    ):
        if sha256_file(path) != split[key]:
            raise ValueError(f"split hash mismatch: {path}")
    train, validation = read_rows(args.train), read_rows(args.validation)
    train_groups = {row["content_group_id"] for row in train}
    valid_groups = {row["content_group_id"] for row in validation}
    if train_groups & valid_groups:
        raise ValueError("question-group overlap detected")
    if any(not 0 <= int(row["label"]) < 5 for row in train + validation):
        raise ValueError("invalid five-choice target")

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
    feature_started = time.perf_counter()
    train_x = encode_options(train, tokenizer, encoder, args.encode_batch_size, args.max_length)
    valid_x = encode_options(
        validation, tokenizer, encoder, args.encode_batch_size, args.max_length
    )
    feature_seconds = time.perf_counter() - feature_started
    train_y = torch.tensor([int(row["label"]) for row in train])
    valid_y = torch.tensor([int(row["label"]) for row in validation])
    torch.save(
        {
            "train_features": train_x,
            "train_targets": train_y,
            "validation_features": valid_x,
            "validation_targets": valid_y,
        },
        output / "frozen-features.pt",
    )
    scorer = nn.Sequential(
        nn.Linear(train_x.shape[-1], 128), nn.Tanh(), nn.Dropout(0.1), nn.Linear(128, 1)
    )
    trainable = sum(parameter.numel() for parameter in scorer.parameters())
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=2e-3, weight_decay=1e-4)
    rng = random.Random(args.seed)
    best_nll, best_epoch, best_state = float("inf"), 0, None
    history = []
    training_started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        order = list(range(len(train)))
        rng.shuffle(order)
        scorer.train()
        losses = []
        for start in range(0, len(order), args.batch_size):
            indices = torch.tensor(order[start : start + args.batch_size])
            features = train_x[indices].reshape(-1, train_x.shape[-1])
            logits = scorer(features).reshape(len(indices), 5)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(logits, train_y[indices])
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        scorer.eval()
        with torch.inference_mode():
            val_logits = scorer(valid_x.reshape(-1, valid_x.shape[-1])).reshape(-1, 5)
        val_result = choice_metrics(val_logits, valid_y)
        history.append({"epoch": epoch, "train_ce": sum(losses) / len(losses), **val_result})
        if val_result["nll"] < best_nll:
            best_nll, best_epoch = val_result["nll"], epoch
            best_state = {key: value.detach().clone() for key, value in scorer.state_dict().items()}
    training_seconds = time.perf_counter() - training_started
    if best_state is None:
        raise RuntimeError("no best validation checkpoint")
    scorer.load_state_dict(best_state)
    scorer.eval()
    with torch.inference_mode():
        final_logits = scorer(valid_x.reshape(-1, valid_x.shape[-1])).reshape(-1, 5)
    final_metrics = choice_metrics(final_logits, valid_y)
    torch.save(best_state, output / "best-readout.pt")
    probabilities = final_logits.float().softmax(-1)
    predictions = []
    for row, probs, target in zip(
        validation, probabilities.tolist(), valid_y.tolist(), strict=True
    ):
        predictions.append(
            {
                "q_id": row["q_id"],
                "sample_id": row["sample_id"],
                "content_group_id": row["content_group_id"],
                "target_index": target,
                "prediction_index": max(range(5), key=probs.__getitem__),
                "ambiguous_target_text": target_text_is_ambiguous(row),
                "probabilities": probs,
            }
        )
    pred_path = output / "validation-predictions.jsonl"
    pred_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions),
        encoding="utf-8",
    )
    report = {
        "status": "complete_development_validation_only",
        "task": "Japanese five-choice commonsense question answering",
        "started_at_utc": started,
        "ended_at_utc": datetime.now(UTC).isoformat(),
        "seed": args.seed,
        "epochs": args.epochs,
        "best_epoch_by_validation_nll": best_epoch,
        "optimizer": "AdamW(lr=0.002, weight_decay=0.0001)",
        "data": {
            "train_sha256": sha256_file(args.train),
            "validation_sha256": sha256_file(args.validation),
            "split_manifest_sha256": sha256_file(args.split_manifest),
            "train_rows": len(train),
            "validation_rows": len(validation),
            "train_groups": len(train_groups),
            "validation_groups": len(valid_groups),
            "cross_split_question_overlap": 0,
            "official_test_downloaded_or_opened": False,
        },
        "model": {
            "id": MODEL_ID,
            "revision": REVISION,
            "weights_sha256": sha256_file(args.model / "pytorch_model.bin"),
            "license": "Apache-2.0",
            "frozen_parameters": sum(parameter.numel() for parameter in encoder.parameters()),
            "weight_bytes": (args.model / "pytorch_model.bin").stat().st_size,
        },
        "readout": {
            "architecture": "shared 768->128 tanh->1 scorer over joint question-option [CLS]",
            "trainable_parameters": trainable,
            "total_prequantization_parameters": sum(
                parameter.numel() for parameter in encoder.parameters()
            )
            + trainable,
            "max_length": args.max_length,
        },
        "history": history,
        "selected_validation_metrics": final_metrics,
        "ambiguous_target_text_count": sum(target_text_is_ambiguous(row) for row in validation),
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
            "device": "cpu",
        },
        "artifacts": {
            name: sha256_file(output / name)
            for name in ("frozen-features.pt", "best-readout.pt", "validation-predictions.jsonl")
        },
        "note": (
            "Validation selected the checkpoint. This is development evidence, not blind "
            "or multimodal quality evidence."
        ),
    }
    (output / "run-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
