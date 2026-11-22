"""Train a small option scorer over frozen Ruri features on grouped JCommonsenseQA."""

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
    from scripts.prepare_jcommonsenseqa_grouped_split import sha256_file, target_text_is_ambiguous
else:
    from prepare_jcommonsenseqa_grouped_split import sha256_file, target_text_is_ambiguous

MODEL_PINS = {
    "cl-nagoya/ruri-v3-30m": {
        "revision": "24899e5de370b56d179604a007c0d727bf144504",
        "license": "Apache-2.0",
        "weights_sha256": "e94155e342cbfc33280c345c6e2912905fb5de6e1a3adeb938f395aafd2f8832",
    },
    "cl-nagoya/ruri-v3-70m": {
        "revision": "07a8b0aba47d29d2ca21f89b915c1efe2c23d1cc",
        "license": "Apache-2.0",
        "weights_sha256": "c959b76a535eee838b1f1e6b8999ba513ef8d87a6f8c3534c50dcda1eb5f6a62",
    },
}


class FrozenFeatureCandidateScorer(nn.Module):
    def __init__(self, feature_size: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_size, hidden_size), nn.GELU(), nn.Linear(hidden_size, 1)
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features).squeeze(-1)


def _metrics(logits: list[Tensor], targets: list[int], bins: int = 15) -> dict[str, Any]:
    probabilities = [torch.softmax(item.float(), dim=-1) for item in logits]
    correct = torch.tensor(
        [
            int(int(prob.argmax()) == target)
            for prob, target in zip(probabilities, targets, strict=True)
        ]
    )
    nll = torch.tensor(
        [
            -float(prob[target].clamp_min(1e-12).log())
            for prob, target in zip(probabilities, targets, strict=True)
        ]
    )
    brier = torch.tensor(
        [
            float((prob - F.one_hot(torch.tensor(target), prob.numel())).square().sum())
            for prob, target in zip(probabilities, targets, strict=True)
        ]
    )
    confidence = torch.tensor([float(prob.max()) for prob in probabilities])
    ece = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        mask = (confidence >= low) & (confidence < high if index < bins - 1 else confidence <= high)
        if mask.any():
            ece += (
                float(mask.sum())
                / len(targets)
                * abs(float(correct[mask].float().mean() - confidence[mask].mean()))
            )
    return {
        "count": len(targets),
        "accuracy": float(correct.float().mean()),
        "nll": float(nll.mean()),
        "brier": float(brier.mean()),
        "ece_15_bins": ece,
    }


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-id", choices=tuple(MODEL_PINS), default="cl-nagoya/ruri-v3-30m")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-questions", type=int, default=64)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    return parser.parse_args()


def _load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _encode(texts: list[str], tokenizer: Any, encoder: Any, batch_size: int) -> Tensor:
    chunks = []
    encoder.eval()
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(
                texts[start : start + batch_size],
                truncation=True,
                max_length=512,
                padding=True,
                return_tensors="pt",
            )
            hidden = encoder(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            pooled = F.normalize(pooled.float(), dim=-1)
            if not torch.isfinite(pooled).all():
                raise ValueError("Ruri produced non-finite frozen features")
            chunks.append(pooled.cpu())
    return torch.cat(chunks)


def _features(
    rows: list[dict[str, Any]], query: Tensor, choices: Tensor
) -> tuple[list[Tensor], list[int]]:
    feature_rows = []
    targets = []
    for index, row in enumerate(rows):
        candidate = choices[index * 5 : (index + 1) * 5]
        context = query[index].expand_as(candidate)
        feature_rows.append(
            torch.cat(
                [context, candidate, (context - candidate).abs(), context * candidate], dim=-1
            )
        )
        targets.append(int(row["label"]))
    return feature_rows, targets


def _group_logits(model: Any, features: list[Tensor], device: torch.device) -> list[Tensor]:
    model.eval()
    with torch.inference_mode():
        flat_logits = model(torch.cat(features).to(device)).cpu()
    return list(flat_logits.reshape(len(features), 5))


def _ambiguous(rows: list[dict[str, Any]]) -> list[int]:
    result = []
    for index, row in enumerate(rows):
        if target_text_is_ambiguous(row):
            result.append(index)
    return result


def main() -> None:
    args = _args()
    model_pin = MODEL_PINS[args.model_id]
    if args.revision != model_pin["revision"]:
        raise ValueError(f"unexpected revision for {args.model_id}")
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output}")
    if args.model.resolve().is_symlink():
        raise ValueError("model path must be a concrete local directory")
    weights_sha256 = sha256_file(args.model / "model.safetensors")
    if weights_sha256 != model_pin["weights_sha256"]:
        raise ValueError(f"local {args.model_id} weights do not match the pinned SHA-256")
    split_receipt = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    if split_receipt["official_test_downloaded_or_opened"]:
        raise ValueError("official test access is prohibited")
    if split_receipt["cross_split_question_group_overlap"] != 0:
        raise ValueError("group leakage detected in split receipt")
    if sha256_file(args.train) != split_receipt["train_jsonl_sha256"]:
        raise ValueError("train data does not match the pinned split manifest")
    if sha256_file(args.validation) != split_receipt["validation_jsonl_sha256"]:
        raise ValueError("validation data does not match the pinned split manifest")
    train, validation = _load(args.train), _load(args.validation)
    if len({row["sample_id"] for row in train + validation}) != len(train) + len(validation):
        raise ValueError("duplicate sample IDs across splits")
    train_groups = {row["content_group_id"] for row in train}
    validation_groups = {row["content_group_id"] for row in validation}
    if train_groups & validation_groups:
        raise ValueError("normalized question group overlap detected")
    if any(
        any(f"choice{i}" not in row for i in range(5)) or not 0 <= int(row["label"]) < 5
        for row in train + validation
    ):
        raise ValueError("five-choice input malformed")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = torch.device("cpu")
    output.mkdir(parents=True)
    started = datetime.now(UTC).isoformat()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    encoder = AutoModel.from_pretrained(
        args.model, local_files_only=True, dtype=torch.float32
    ).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in encoder.parameters()):
        raise AssertionError("Ruri encoder was not frozen")

    feature_started = time.perf_counter()
    all_rows = train + validation
    query_texts = [row["question"] for row in all_rows]
    choice_texts = [row[f"choice{i}"] for row in all_rows for i in range(5)]
    query_vectors = _encode(query_texts, tokenizer, encoder, args.feature_batch_size)
    choice_vectors = _encode(choice_texts, tokenizer, encoder, args.feature_batch_size)
    split_at = len(train)
    train_features, train_targets = _features(
        train, query_vectors[:split_at], choice_vectors[: split_at * 5]
    )
    val_features, val_targets = _features(
        validation, query_vectors[split_at:], choice_vectors[split_at * 5 :]
    )
    feature_seconds = time.perf_counter() - feature_started
    feature_path = output / "frozen-features.pt"
    torch.save({"train": train_features, "validation": val_features}, feature_path)

    scorer = FrozenFeatureCandidateScorer(int(encoder.config.hidden_size) * 4)
    trainable_parameters = sum(parameter.numel() for parameter in scorer.parameters())
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = random.Random(args.seed)
    history = []
    best_state = None
    best_nll = float("inf")
    best_epoch = 0
    train_started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        order = list(range(len(train_features)))
        rng.shuffle(order)
        scorer.train()
        losses = []
        for start in range(0, len(order), args.batch_questions):
            indices = order[start : start + args.batch_questions]
            packed = torch.cat([train_features[index] for index in indices])
            scores = scorer(packed)
            scores = scores.reshape(len(indices), 5)
            targets = torch.tensor([train_targets[index] for index in indices])
            loss = F.cross_entropy(scores, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        val_logits = _group_logits(scorer, val_features, device)
        val_metrics = _metrics(val_logits, val_targets)
        history.append({"epoch": epoch, "train_ce": sum(losses) / len(losses), **val_metrics})
        if val_metrics["nll"] < best_nll:
            best_nll = val_metrics["nll"]
            best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in scorer.state_dict().items()}
    train_seconds = time.perf_counter() - train_started
    if best_state is None:
        raise RuntimeError("no best validation checkpoint")
    scorer.load_state_dict(best_state)
    val_logits = _group_logits(scorer, val_features, device)
    metrics = _metrics(val_logits, val_targets)
    ambiguous = set(_ambiguous(validation))
    unambiguous_logits = [value for i, value in enumerate(val_logits) if i not in ambiguous]
    unambiguous_targets = [value for i, value in enumerate(val_targets) if i not in ambiguous]
    predictions = []
    for row, logits, target, index in zip(
        validation, val_logits, val_targets, range(len(validation)), strict=True
    ):
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
                "ambiguous_target_text": index in ambiguous,
                "probabilities": probabilities,
            }
        )
    torch.save(best_state, output / "best-readout.pt")
    (output / "validation-predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions),
        encoding="utf-8",
    )
    report = {
        "status": "complete_development_validation_only",
        "claim": "Japanese five-choice commonsense QA, not general multimodal quality",
        "started_at_utc": started,
        "ended_at_utc": datetime.now(UTC).isoformat(),
        "seed": args.seed,
        "source_commit": split_receipt["source"]["commit"],
        "source_license": split_receipt["source"]["license"],
        "source_files_sha256": split_receipt["source"]["source_files_sha256"],
        "train_sha256": sha256_file(args.train),
        "validation_sha256": sha256_file(args.validation),
        "split_manifest_sha256": sha256_file(args.split_manifest),
        "official_test_downloaded_or_opened": False,
        "train_rows": len(train),
        "train_question_groups": len(train_groups),
        "validation_rows": len(validation),
        "validation_question_groups": len(validation_groups),
        "group_overlap": 0,
        "model_id": args.model_id,
        "model_revision": args.revision,
        "model_license_declared": model_pin["license"],
        "model_weights_sha256": weights_sha256,
        "model_weights_bytes": (args.model / "model.safetensors").stat().st_size,
        "model_config_sha256": sha256_file(args.model / "config.json"),
        "encoder_class": type(encoder).__name__,
        "encoder_hidden_size": int(encoder.config.hidden_size),
        "encoder_layers": int(encoder.config.num_hidden_layers),
        "encoder_parameters_frozen": sum(parameter.numel() for parameter in encoder.parameters()),
        "trainable_readout_parameters": trainable_parameters,
        "total_prequantization_parameters": (
            sum(parameter.numel() for parameter in encoder.parameters()) + trainable_parameters
        ),
        "readout": "shared 1024->128 GELU->1 option scorer over [q,c,abs(q-c),q*c]",
        "optimizer": "AdamW(lr=1e-3, weight_decay=1e-4)",
        "epochs": args.epochs,
        "batch_questions": args.batch_questions,
        "history": history,
        "best_epoch_by_validation_nll": best_epoch,
        "validation_metrics_all_rows": metrics,
        "validation_metrics_unambiguous_target_text_only": _metrics(
            unambiguous_logits, unambiguous_targets
        ),
        "validation_ambiguous_target_text_count": len(ambiguous),
        "feature_extraction_seconds": feature_seconds,
        "training_seconds": train_seconds,
        "device": str(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "feature_cache_sha256": sha256_file(feature_path),
        "checkpoint_sha256": sha256_file(output / "best-readout.pt"),
        "predictions_sha256": sha256_file(output / "validation-predictions.jsonl"),
        "note": (
            "Validation chose the checkpoint; this is development evidence, "
            "not blind audit evidence."
        ),
    }
    (output / "run-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
