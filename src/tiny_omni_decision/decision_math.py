from __future__ import annotations

import math
from collections.abc import Sequence
from string import ascii_lowercase, ascii_uppercase, digits

from .schema import TextDecision


def option_labels(option_count: int) -> list[str]:
    """Return distinct single-character labels for every supplied option."""
    labels = list(ascii_uppercase + ascii_lowercase + digits)
    if option_count < 2 or option_count > len(labels):
        raise ValueError(f"option count must be between 2 and {len(labels)}")
    return labels[:option_count]


def prompt_for_decision(sample: TextDecision, labels: list[str]) -> str:
    choices = "\n".join(
        f"{label}. {option}" for label, option in zip(labels, sample.options, strict=True)
    )
    return (
        f"State: {sample.state}\nQuestion: {sample.question}\n"
        f"Choose the best option.\n{choices}\nAnswer:"
    )


def option_label_ids(tokenizer: object, labels: list[str]) -> list[int]:
    """Resolve labels to single vocabulary IDs; refuse ambiguous/multitoken labels."""
    ids: list[int] = []
    for label in labels:
        encoded = tokenizer.encode(label, add_special_tokens=False)  # type: ignore[attr-defined]
        if len(encoded) != 1:
            raise ValueError(f"Option label {label!r} must tokenize to exactly one token")
        ids.append(int(encoded[0]))
    if len(set(ids)) != len(ids):
        raise ValueError("Option labels must map to distinct token IDs")
    return ids


def label_token_ids_from_prompt(
    tokenizer: object, prompt: str, labels: list[str]
) -> tuple[list[int], int]:
    """Validate one-token label continuations and return IDs plus prompt length."""
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)  # type: ignore[attr-defined]
    ids: list[int] = []
    for label in labels:
        full_ids = tokenizer.encode(prompt + label, add_special_tokens=False)  # type: ignore[attr-defined]
        if len(full_ids) != len(prompt_ids) + 1 or full_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError(f"Option label {label!r} is not one continuation token")
        ids.append(int(full_ids[-1]))
    if len(set(ids)) != len(ids):
        raise ValueError("Option labels must map to distinct next-token IDs")
    return ids, len(prompt_ids)


def normalize_scores(scores: list[float]) -> list[float]:
    if len(scores) < 2:
        raise ValueError("scores must contain at least two options")
    maximum = max(scores)
    exps = [math.exp(score - maximum) for score in scores]
    total = sum(exps)
    return [value / total for value in exps]


def brier_score(probabilities: list[list[float]], targets: list[int]) -> float:
    if not probabilities or len(probabilities) != len(targets):
        raise ValueError("probability batch and targets must be nonempty and have equal lengths")
    total = 0.0
    for row, target in zip(probabilities, targets, strict=True):
        if target < 0 or target >= len(row):
            raise ValueError("target index is out of range")
        total += sum((value - float(index == target)) ** 2 for index, value in enumerate(row))
    return total / len(probabilities)


def negative_log_likelihood(
    probabilities: Sequence[Sequence[float]], targets: Sequence[int]
) -> float:
    if not probabilities or len(probabilities) != len(targets):
        raise ValueError("probability batch and targets must be nonempty and have equal lengths")
    total = 0.0
    for row, target in zip(probabilities, targets, strict=True):
        if target < 0 or target >= len(row):
            raise ValueError("target index is out of range")
        if any(not math.isfinite(value) or value < 0.0 for value in row):
            raise ValueError("probabilities must be finite and nonnegative")
        total -= math.log(max(float(row[target]), 1e-12))
    return total / len(probabilities)


def expected_calibration_error(
    probabilities: Sequence[Sequence[float]],
    targets: Sequence[int],
    n_bins: int = 15,
) -> float:
    """Top-label ECE with equal-width confidence bins on [0, 1]."""
    if n_bins < 1:
        raise ValueError("n_bins must be at least one")
    if not probabilities or len(probabilities) != len(targets):
        raise ValueError("probability batch and targets must be nonempty and have equal lengths")
    bins: list[list[float]] = [[] for _ in range(n_bins)]
    for row, target in zip(probabilities, targets, strict=True):
        if len(row) < 2 or target < 0 or target >= len(row):
            raise ValueError("each target must index a probability row with at least two options")
        if any(not math.isfinite(value) or value < 0.0 for value in row):
            raise ValueError("probabilities must be finite and nonnegative")
        prediction = max(range(len(row)), key=row.__getitem__)
        confidence = float(row[prediction])
        if confidence > 1.0:
            raise ValueError("probabilities must be in [0, 1]")
        index = min(int(confidence * n_bins), n_bins - 1)
        bins[index].append(float(prediction == target) - confidence)
    count = len(targets)
    return sum(len(items) * abs(sum(items) / len(items)) for items in bins if items) / count


def reorder_target(
    options: list[str], target: str, permutation: list[int]
) -> tuple[list[str], int]:
    if sorted(permutation) != list(range(len(options))):
        raise ValueError("permutation must contain each option index exactly once")
    if target not in options:
        raise ValueError("target must match one of the options")
    reordered = [options[index] for index in permutation]
    return reordered, reordered.index(target)
