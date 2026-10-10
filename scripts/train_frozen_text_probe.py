"""Train a small Decision scorer over frozen, locally pinned text features.

This is a bounded Path A probe for the synthetic Typed Decisions Synth task;
it is not a quality claim for real-world text decisions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer

from tiny_omni_decision.dataset import iter_local_rows, normalize_jsonl, sha256_file
from tiny_omni_decision.schema import DatasetManifest


class CandidateScorer(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features).squeeze(-1)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-questions", type=int, default=64)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    return parser.parse_args()


def _load_examples(path: Path, manifest: DatasetManifest) -> list[Any]:
    return list(normalize_jsonl(iter_local_rows(path), manifest, "typed-decisions-synth"))


def _identity_sets(examples: list[Any]) -> tuple[set[str], set[str]]:
    state_ids = {example.source_record_id.split(":", 1)[0] for example in examples}
    content = {
        hashlib.sha256(
            json.dumps(
                {
                    "state": example.state,
                    "question": example.question,
                    "options": sorted(example.options),
                },
                sort_keys=True,
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()
        for example in examples
    }
    return state_ids, content


def _assert_disjoint(train: list[Any], validation: list[Any]) -> None:
    train_groups, train_content = _identity_sets(train)
    validation_groups, validation_content = _identity_sets(validation)
    if train_groups & validation_groups or train_content & validation_content:
        raise ValueError("train/validation state or normalized-content overlap detected")


def _format_context(example: Any) -> str:
    return f"State: {example.state}\nQuestion: {example.question}"


def _extract_features(
    examples: list[Any],
    tokenizer: Any,
    encoder: nn.Module,
    device: torch.device,
    batch_size: int,
) -> tuple[list[Tensor], list[int]]:
    text_rows: list[str] = []
    question_ranges: list[tuple[int, int]] = []
    contexts: list[str] = []
    targets: list[int] = []
    for example in examples:
        context = _format_context(example)
        contexts.append(context)
        start = len(text_rows)
        text_rows.extend(f"Candidate answer: {option}" for option in example.options)
        question_ranges.append((start, len(text_rows)))
        targets.append(example.options.index(example.target))

    all_text = contexts + text_rows
    embeddings: list[Tensor] = []
    encoder.eval()
    with torch.inference_mode():
        for start in range(0, len(all_text), batch_size):
            encoded = tokenizer(
                all_text[start : start + batch_size],
                max_length=256,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(device)
            hidden = encoder(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = ((hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)).float().cpu()
            if not torch.isfinite(pooled).all():
                raise ValueError("frozen encoder produced a non-finite candidate feature")
            embeddings.append(pooled)
    pooled = torch.cat(embeddings)
    context_embeddings = pooled[: len(contexts)]
    option_embeddings = pooled[len(contexts) :]
    features = []
    offset = 0
    for context, (lo, hi) in zip(context_embeddings, question_ranges, strict=True):
        options = option_embeddings[lo:hi]
        expanded_context = context.unsqueeze(0).expand(options.shape[0], -1)
        features.append(
            torch.cat(
                [
                    expanded_context,
                    options,
                    (expanded_context - options).abs(),
                    expanded_context * options,
                ],
                dim=-1,
            )
        )
        offset = hi
    if offset != len(option_embeddings):
        raise AssertionError("candidate feature ranges do not cover encoded options")
    return features, targets


def _metrics(logits: list[Tensor], targets: list[int], bins: int = 15) -> dict[str, Any]:
    probabilities = [torch.softmax(item.float(), dim=-1) for item in logits]
    correct = torch.tensor(
        [
            int(int(prob.argmax()) == target)
            for prob, target in zip(probabilities, targets, strict=True)
        ],
        dtype=torch.float32,
    )
    nll = torch.tensor(
        [
            -float(prob[target].clamp_min(1e-12).log())
            for prob, target in zip(probabilities, targets, strict=True)
        ],
        dtype=torch.float32,
    )
    brier = torch.tensor(
        [
            float((prob - F.one_hot(torch.tensor(target), prob.numel())).square().sum())
            for prob, target in zip(probabilities, targets, strict=True)
        ],
        dtype=torch.float32,
    )
    confidence = torch.tensor([float(prob.max()) for prob in probabilities])
    ece = 0.0
    bin_counts = []
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        mask = (confidence >= low) & (confidence < high if index < bins - 1 else confidence <= high)
        count = int(mask.sum())
        bin_counts.append(count)
        if count:
            ece += count / len(targets) * abs(float(correct[mask].mean() - confidence[mask].mean()))
    return {
        "count": len(targets),
        "accuracy": float(correct.mean()),
        "nll": float(nll.mean()),
        "brier": float(brier.mean()),
        "ece_15_bins": ece,
        "ece_bin_counts": bin_counts,
    }


def _predict(scorer: nn.Module, features: list[Tensor], device: torch.device) -> list[Tensor]:
    scorer.eval()
    with torch.inference_mode():
        widths = [feature.shape[0] for feature in features]
        flat = torch.cat(features).to(device)
        flat_logits = scorer(flat).cpu()
        result = []
        offset = 0
        for width in widths:
            result.append(flat_logits[offset : offset + width])
            offset += width
        return result


def main() -> None:
    args = _args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        device = torch.device("cuda")
        torch.cuda.reset_peak_memory_stats(device)
    else:
        device = torch.device("cpu")
    torch.set_num_threads(4)

    manifest_data = __import__("yaml").safe_load(args.manifest.read_text(encoding="utf-8"))
    manifest = DatasetManifest.model_validate(manifest_data)
    train_manifest = manifest.model_copy(update={"split": "train"})
    validation_manifest = manifest.model_copy(update={"split": "validation"})
    train = _load_examples(args.train, train_manifest)
    validation = _load_examples(args.validation, validation_manifest)
    _assert_disjoint(train, validation)
    train_groups, _ = _identity_sets(train)
    validation_groups, _ = _identity_sets(validation)

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    encoder = AutoModel.from_pretrained(args.model, local_files_only=True).eval().to(device)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    feature_started = time.perf_counter()
    train_features, train_targets = _extract_features(
        train, tokenizer, encoder, device, args.feature_batch_size
    )
    validation_features, validation_targets = _extract_features(
        validation, tokenizer, encoder, device, args.feature_batch_size
    )
    feature_seconds = time.perf_counter() - feature_started
    feature_path = output / "frozen-features.pt"
    torch.save(
        {
            "train_features": train_features,
            "train_targets": train_targets,
            "validation_features": validation_features,
            "validation_targets": validation_targets,
        },
        feature_path,
    )
    counts = Counter(len(example.options) for example in train)
    scorer = CandidateScorer(int(encoder.config.hidden_size) * 4).to(device)
    trainable_parameters = sum(parameter.numel() for parameter in scorer.parameters())
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=1e-3, weight_decay=1e-4)
    order_rng = random.Random(args.seed)
    history: list[dict[str, float | int]] = []
    best_nll = float("inf")
    best_state: dict[str, Tensor] | None = None
    train_started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        order = list(range(len(train_features)))
        order_rng.shuffle(order)
        scorer.train()
        losses = []
        for start in range(0, len(order), args.batch_questions):
            indices = order[start : start + args.batch_questions]
            rows = [train_features[index].to(device) for index in indices]
            targets = torch.tensor([train_targets[index] for index in indices], device=device)
            widths = [row.shape[0] for row in rows]
            candidates = torch.cat(rows, dim=0)
            scores = scorer(candidates)
            max_width = max(widths)
            padded = scores.new_full((len(rows), max_width), -1e4)
            offset = 0
            for row_index, width in enumerate(widths):
                padded[row_index, :width] = scores[offset : offset + width]
                offset += width
            loss = F.cross_entropy(padded, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        val_logits = _predict(scorer, validation_features, device)
        val_metrics = _metrics(val_logits, validation_targets)
        history.append(
            {
                "epoch": epoch,
                "train_ce": sum(losses) / len(losses),
                **{
                    f"validation_{key}": value
                    for key, value in val_metrics.items()
                    if isinstance(value, (int, float))
                },
            }
        )
        if val_metrics["nll"] < best_nll:
            best_nll = val_metrics["nll"]
            best_state = {
                key: value.detach().cpu().clone() for key, value in scorer.state_dict().items()
            }
    training_seconds = time.perf_counter() - train_started
    if best_state is None:
        raise RuntimeError("no validation-selected checkpoint was produced")
    scorer.load_state_dict(best_state)
    best_logits = _predict(scorer, validation_features, device)
    best_metrics = _metrics(best_logits, validation_targets)
    predictions = [
        {
            "id": example.id,
            "source_record_id": example.source_record_id,
            "target": example.target,
            "options": example.options,
            "prediction": example.options[int(logit.argmax())],
            "probabilities": torch.softmax(logit.float(), dim=-1).tolist(),
        }
        for example, logit in zip(validation, best_logits, strict=True)
    ]
    torch.save(best_state, output / "best-readout.pt")
    (output / "validation-predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions),
        encoding="utf-8",
    )
    report = {
        "schema_version": 1,
        "status": "complete_text_only_synthetic_probe",
        "task_claim": "Typed Decisions Synth is synthetic and not a real-world task benchmark.",
        "seed": args.seed,
        "source_revision": manifest.revision,
        "model_revision": args.revision,
        "model_weights_sha256": sha256_file(args.model / "model.safetensors"),
        "train_file_sha256": sha256_file(args.train),
        "validation_file_sha256": sha256_file(args.validation),
        "train_examples": len(train),
        "validation_examples": len(validation),
        "train_unique_states": len(train_groups),
        "validation_unique_states": len(validation_groups),
        "train_validation_state_overlap": 0,
        "train_validation_content_overlap": 0,
        "train_option_count_distribution": dict(sorted(counts.items())),
        "feature_cache_bytes": feature_path.stat().st_size,
        "feature_cache_sha256": sha256_file(feature_path),
        "feature_extraction_seconds": feature_seconds,
        "training_seconds": training_seconds,
        "device": str(device),
        "encoder_parameters_frozen": sum(parameter.numel() for parameter in encoder.parameters()),
        "trainable_readout_parameters": trainable_parameters,
        "readout": "MLP([context, option, |difference|, product]->128 GELU->1); option softmax",
        "optimizer": "AdamW(lr=0.001, weight_decay=0.0001)",
        "epochs": args.epochs,
        "history": history,
        "best_validation": best_metrics,
        "best_epoch": int(
            history[min(range(len(history)), key=lambda i: float(history[i]["validation_nll"]))][
                "epoch"
            ]
        ),
        "predictions_path": "validation-predictions.jsonl",
        "head_path": "best-readout.pt",
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else None,
        "note": (
            "Validation is used for checkpoint selection; it is development data, "
            "not a blind audit."
        ),
    }
    (output / "run-report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
