"""Small shared scoring utilities for Social IQa supplied-option experiments."""

from __future__ import annotations

import hashlib
import unicodedata
from collections import Counter
from typing import Any


def normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def context_hash(context: str) -> str:
    return hashlib.sha256(normalize_text(context).encode("utf-8")).hexdigest()


def validate_splits(
    train: list[dict[str, Any]], validation: list[dict[str, Any]]
) -> dict[str, Any]:
    """Fail closed on malformed examples and exact cross-split content leakage."""
    splits = {"train": train, "validation": validation}
    ids_by_split: dict[str, set[str]] = {}
    groups_by_split: dict[str, set[str]] = {}
    examples_by_split: dict[str, set[tuple[str, ...]]] = {}
    qoptions_by_split: dict[str, set[tuple[str, ...]]] = {}
    targets: dict[str, Counter[int]] = {}
    for split, rows in splits.items():
        ids: set[str] = set()
        groups: set[str] = set()
        examples: set[tuple[str, ...]] = set()
        qoptions: set[tuple[str, ...]] = set()
        counts: Counter[int] = Counter()
        for row in rows:
            required = {
                "sample_id",
                "context",
                "question",
                "options",
                "target",
                "source_context_sha256",
            }
            if not required.issubset(row):
                raise ValueError(f"{split} row is missing required fields")
            sample_id = str(row["sample_id"])
            if not sample_id or sample_id in ids:
                raise ValueError(f"{split} sample IDs must be nonempty and unique")
            ids.add(sample_id)
            context = str(row["context"])
            question = str(row["question"])
            options = row["options"]
            target = row["target"]
            if not context.strip() or not question.strip() or not isinstance(options, list):
                raise ValueError(f"{split} row has empty text or malformed options")
            if len(options) != 3 or any(
                not isinstance(option, str) or not option.strip() for option in options
            ):
                raise ValueError(f"{split} row must have three nonempty text options")
            if not isinstance(target, int) or isinstance(target, bool) or target not in range(3):
                raise ValueError(f"{split} target must be a zero-based option index")
            expected_hash = context_hash(context)
            if row["source_context_sha256"] != expected_hash:
                raise ValueError(f"{split} context hash mismatch")
            groups.add(expected_hash)
            normalized_question = normalize_text(question)
            normalized_options = tuple(normalize_text(option) for option in options)
            examples.add((normalize_text(context), normalized_question, *normalized_options))
            qoptions.add((normalized_question, *normalized_options))
            counts[target] += 1
        ids_by_split[split] = ids
        groups_by_split[split] = groups
        examples_by_split[split] = examples
        qoptions_by_split[split] = qoptions
        targets[split] = counts
    if groups_by_split["train"] & groups_by_split["validation"]:
        raise ValueError("normalized context identity crosses train/validation")
    if examples_by_split["train"] & examples_by_split["validation"]:
        raise ValueError("normalized full example content crosses train/validation")
    if qoptions_by_split["train"] & qoptions_by_split["validation"]:
        raise ValueError("normalized question/options content crosses train/validation")
    return {
        "train_rows": len(train),
        "validation_rows": len(validation),
        "train_context_groups": len(groups_by_split["train"]),
        "validation_context_groups": len(groups_by_split["validation"]),
        "context_group_overlap": 0,
        "normalized_full_example_overlap": 0,
        "normalized_question_options_overlap": 0,
        "target_counts": {split: dict(sorted(count.items())) for split, count in targets.items()},
    }


def classification_metrics(logits: Any, targets: Any, *, bins: int = 15) -> dict[str, Any]:
    """Compute accuracy and calibration metrics from [N,3] CPU logits."""
    import torch
    import torch.nn.functional as F

    scores = torch.as_tensor(logits, dtype=torch.float64, device="cpu")
    labels = torch.as_tensor(targets, dtype=torch.long, device="cpu")
    if scores.ndim != 2 or scores.shape[1] != 3 or labels.shape != (scores.shape[0],):
        raise ValueError("expected logits [N,3] and targets [N]")
    if (
        scores.shape[0] == 0
        or not torch.isfinite(scores).all()
        or not torch.all((labels >= 0) & (labels < 3))
    ):
        raise ValueError("metrics require finite nonempty logits and valid targets")
    probabilities = torch.softmax(scores, dim=1)
    predictions = probabilities.argmax(dim=1)
    confidence = probabilities.max(dim=1).values
    correct = predictions.eq(labels)
    one_hot = F.one_hot(labels, num_classes=3).to(dtype=probabilities.dtype)
    ece = torch.zeros((), dtype=probabilities.dtype)
    bin_counts: list[int] = []
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        selected = (confidence >= lower) & (
            (confidence < upper) if index < bins - 1 else (confidence <= upper)
        )
        count = int(selected.sum())
        bin_counts.append(count)
        if count:
            ece += selected.to(probabilities.dtype).mean() * torch.abs(
                confidence[selected].mean() - correct[selected].to(probabilities.dtype).mean()
            )
    per_target: dict[str, dict[str, float | int | None]] = {}
    for target in range(3):
        selected = labels.eq(target)
        count = int(selected.sum())
        per_target[str(target)] = {
            "count": count,
            "accuracy": float(correct[selected].to(torch.float64).mean()) if count else None,
        }
    class_acc = [float(entry["accuracy"]) for entry in per_target.values() if entry["count"]]
    return {
        "count": int(scores.shape[0]),
        "accuracy": float(correct.to(torch.float64).mean()),
        "macro_accuracy_by_correct_option_position": float(sum(class_acc) / len(class_acc)),
        "nll": float(F.cross_entropy(scores, labels)),
        "brier": float(((probabilities - one_hot).square().sum(dim=1)).mean()),
        "ece_15": float(ece),
        "mean_confidence": float(confidence.mean()),
        "ece_bin_counts": bin_counts,
        "per_target": per_target,
    }
