"""Fit a small scorer over frozen Ruri joint question-option features."""

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
from prepare_jcommonsenseqa_grouped_split import sha256_file
from torch import Tensor, nn
from train_jcommonsenseqa_ruri_readout import _metrics
from transformers import AutoModel, AutoTokenizer

MODEL_REVISION = "24899e5de370b56d179604a007c0d727bf144504"
MODEL_SHA256 = "e94155e342cbfc33280c345c6e2912905fb5de6e1a3adeb938f395aafd2f8832"


class PairScorer(nn.Module):
    def __init__(self, feature_size: int = 256, hidden_size: int = 509) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_size, hidden_size), nn.GELU(), nn.Linear(hidden_size, 1)
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features).squeeze(-1)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-questions", type=int, default=64)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _encode_pairs(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    encoder: nn.Module,
    device: torch.device,
    batch_size: int,
) -> list[Tensor]:
    questions = [row["question"] for row in rows for _ in range(5)]
    options = [row[f"choice{i}"] for row in rows for i in range(5)]
    chunks = []
    encoder.eval()
    with torch.inference_mode():
        for start in range(0, len(questions), batch_size):
            encoded = tokenizer(
                questions[start : start + batch_size],
                options[start : start + batch_size],
                max_length=512,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(device)
            hidden = encoder(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            pooled = F.normalize(pooled.float(), dim=-1)
            if not torch.isfinite(pooled).all():
                raise ValueError("joint Ruri features contain non-finite values")
            chunks.append(pooled.cpu())
    if len(rows) == 0:
        raise ValueError("cannot encode zero question-option pairs")
    flat = torch.cat(chunks)
    return list(flat.reshape(len(rows), 5, -1))


def _predict(model: nn.Module, features: list[Tensor], device: torch.device) -> list[Tensor]:
    model.eval()
    with torch.inference_mode():
        scores = model(torch.cat(features).to(device)).cpu()
    return list(scores.reshape(len(features), 5))


def _fit_head(
    train_features: list[Tensor],
    train_targets: list[int],
    validation_features: list[Tensor],
    validation_targets: list[int],
    *,
    seed: int,
    epochs: int,
    batch_questions: int,
) -> tuple[PairScorer, dict[str, Tensor], list[dict[str, Any]], int, float]:
    torch.manual_seed(seed)
    randomizer = random.Random(seed)
    scorer = PairScorer()
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=1e-3, weight_decay=1e-4)
    history = []
    best_state = None
    best_nll = float("inf")
    best_epoch = 0
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        order = list(range(len(train_features)))
        randomizer.shuffle(order)
        scorer.train()
        losses = []
        for start in range(0, len(order), batch_questions):
            indices = order[start : start + batch_questions]
            batch = torch.cat([train_features[index] for index in indices])
            logits = scorer(batch).reshape(len(indices), 5)
            targets = torch.tensor([train_targets[index] for index in indices])
            loss = F.cross_entropy(logits, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        validation_logits = _predict(scorer, validation_features, torch.device("cpu"))
        metrics = _metrics(validation_logits, validation_targets)
        history.append({"epoch": epoch, "train_ce": sum(losses) / len(losses), **metrics})
        if metrics["nll"] < best_nll:
            best_nll = metrics["nll"]
            best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in scorer.state_dict().items()}
    if best_state is None:
        raise RuntimeError("validation selector did not produce a checkpoint")
    return scorer, best_state, history, best_epoch, time.perf_counter() - started


def main() -> None:
    args = _args()
    if args.revision != MODEL_REVISION:
        raise ValueError("unexpected Ruri revision")
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")
    if sha256_file(args.model / "model.safetensors") != MODEL_SHA256:
        raise ValueError("local Ruri weights do not match pinned SHA-256")
    receipt = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    if receipt["official_test_downloaded_or_opened"]:
        raise ValueError("official test access is forbidden")
    if receipt["cross_split_question_group_overlap"] != 0:
        raise ValueError("split receipt contains question-group overlap")
    if sha256_file(args.train) != receipt["train_jsonl_sha256"]:
        raise ValueError("train file does not match the split receipt")
    if sha256_file(args.validation) != receipt["validation_jsonl_sha256"]:
        raise ValueError("validation file does not match the split receipt")
    train, validation = _load_rows(args.train), _load_rows(args.validation)
    if args.smoke:
        train = train[:8]
        validation = validation[:8]
    groups_train = {row["content_group_id"] for row in train}
    groups_validation = {row["content_group_id"] for row in validation}
    if groups_train & groups_validation:
        raise ValueError("train/validation exact-question group overlap detected")
    for row in train + validation:
        if not 0 <= int(row["label"]) < 5 or any(f"choice{i}" not in row for i in range(5)):
            raise ValueError("invalid five-choice row")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(
        "cuda"
        if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    output.mkdir(parents=True)
    started_at = datetime.now(UTC).isoformat()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    encoder = AutoModel.from_pretrained(args.model, local_files_only=True, dtype=torch.float32)
    encoder.eval().to(device)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in encoder.parameters()):
        raise AssertionError("Ruri encoder unexpectedly trainable")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    feature_started = time.perf_counter()
    train_features = _encode_pairs(train, tokenizer, encoder, device, args.feature_batch_size)
    validation_features = _encode_pairs(
        validation, tokenizer, encoder, device, args.feature_batch_size
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    feature_seconds = time.perf_counter() - feature_started
    train_targets = [int(row["label"]) for row in train]
    validation_targets = [int(row["label"]) for row in validation]

    if args.smoke:
        receipt_out = {
            "status": "smoke_pass",
            "rows_per_split": len(train),
            "pairs_per_question": 5,
            "train_feature_shape": list(train_features[0].shape),
            "validation_feature_shape": list(validation_features[0].shape),
            "train_features_finite": bool(torch.isfinite(torch.stack(train_features)).all()),
            "validation_features_finite": bool(
                torch.isfinite(torch.stack(validation_features)).all()
            ),
            "device": str(device),
            "feature_extraction_seconds": feature_seconds,
        }
        (output / "smoke-report.json").write_text(
            json.dumps(receipt_out, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(receipt_out, indent=2))
        return

    feature_path = output / "frozen-joint-features.pt"
    torch.save({"train": train_features, "validation": validation_features}, feature_path)
    scorer, best_state, history, best_epoch, training_seconds = _fit_head(
        train_features,
        train_targets,
        validation_features,
        validation_targets,
        seed=args.seed,
        epochs=args.epochs,
        batch_questions=args.batch_questions,
    )
    scorer.load_state_dict(best_state)
    best_logits = _predict(scorer, validation_features, torch.device("cpu"))
    best_train_logits = _predict(scorer, train_features, torch.device("cpu"))
    best_metrics = _metrics(best_logits, validation_targets)
    train_metrics = _metrics(best_train_logits, train_targets)
    checkpoint = output / "best-readout.pt"
    torch.save(best_state, checkpoint)
    reloaded = PairScorer()
    reloaded.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    replay_logits = _predict(reloaded, validation_features, torch.device("cpu"))
    replay_delta = max(
        float(
            (torch.softmax(first.float(), dim=-1) - torch.softmax(second.float(), dim=-1))
            .abs()
            .max()
        )
        for first, second in zip(best_logits, replay_logits, strict=True)
    )
    if replay_delta != 0:
        raise ValueError(f"readout reload probability drift: {replay_delta}")
    predictions = []
    for row, logits, target in zip(validation, best_logits, validation_targets, strict=True):
        probabilities = torch.softmax(logits.float(), dim=-1).tolist()
        prediction = int(logits.argmax())
        predictions.append(
            {
                "sample_id": row["sample_id"],
                "content_group_id": row["content_group_id"],
                "target_index": target,
                "prediction_index": prediction,
                "target_text": row[f"choice{target}"],
                "prediction_text": row[f"choice{prediction}"],
                "probabilities": probabilities,
            }
        )
    predictions_path = output / "validation-predictions.jsonl"
    predictions_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions),
        encoding="utf-8",
    )
    report = {
        "status": "complete_development_validation_only",
        "task": "Japanese five-choice commonsense QA",
        "source_commit": receipt["source"]["commit"],
        "source_license": receipt["source"]["license"],
        "official_test_downloaded_or_opened": False,
        "seed": args.seed,
        "started_at_utc": started_at,
        "ended_at_utc": datetime.now(UTC).isoformat(),
        "train_rows": len(train),
        "train_groups": len(groups_train),
        "validation_rows": len(validation),
        "validation_groups": len(groups_validation),
        "train_sha256": sha256_file(args.train),
        "validation_sha256": sha256_file(args.validation),
        "split_manifest_sha256": sha256_file(args.split_manifest),
        "model_revision": args.revision,
        "model_weight_sha256": MODEL_SHA256,
        "encoder_parameters_frozen": sum(parameter.numel() for parameter in encoder.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in scorer.parameters()),
        "scorer": "shared 256->509 GELU->1 over L2-normalized joint question-option pair embedding",
        "representation_change": (
            "Ruri sees each question and option together; backbone remains frozen"
        ),
        "training": {
            "optimizer": "AdamW(lr=1e-3, weight_decay=1e-4)",
            "epochs": args.epochs,
            "batch_questions": args.batch_questions,
            "best_epoch_by_validation_nll": best_epoch,
            "history": history,
        },
        "training_metrics_selected": train_metrics,
        "validation_metrics_selected": best_metrics,
        "reload_maximum_probability_delta": replay_delta,
        "feature_extraction_seconds": feature_seconds,
        "readout_training_seconds": training_seconds,
        "device": str(device),
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else None,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "features_sha256": sha256_file(feature_path),
        "checkpoint_sha256": sha256_file(checkpoint),
        "predictions_sha256": sha256_file(predictions_path),
        "note": (
            "Same grouped development validation as the previous candidate; "
            "not blind or independent audit evidence."
        ),
    }
    (output / "run-report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
