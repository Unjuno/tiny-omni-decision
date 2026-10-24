"""Deterministic supplied-option adapter for CLEVR VQA answer labels."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence

import torch

ANSWER_VOCABULARIES: dict[str, tuple[str, ...]] = {
    "boolean": ("no", "yes"),
    "size": ("large", "small"),
    "material": ("metal", "rubber"),
    "shape": ("cube", "cylinder", "sphere"),
    "color": ("blue", "brown", "cyan", "gray", "green", "purple", "red", "yellow"),
    "count": tuple(str(value) for value in range(11)),
}
OPTION_SEED = 17

ANSWER_LABELS: tuple[str, ...] = tuple(
    label for vocabulary in ANSWER_VOCABULARIES.values() for label in vocabulary
)
ANSWER_TO_KIND: dict[str, str] = {
    label: kind for kind, labels in ANSWER_VOCABULARIES.items() for label in labels
}


def answer_kind(answer: str) -> str:
    """Return the fixed CLEVR answer kind, rejecting labels outside its vocabulary."""
    try:
        return ANSWER_TO_KIND[answer]
    except KeyError as exc:
        raise ValueError(f"unsupported CLEVR answer: {answer!r}") from exc


def make_options(
    *,
    image_filename: str,
    question_index: int,
    answer: str,
    seed: int = OPTION_SEED,
) -> tuple[str, ...]:
    """Make a deterministic 2–4-way option set from the answer's semantic type."""
    kind = answer_kind(answer)
    vocabulary = ANSWER_VOCABULARIES[kind]
    key = f"clevr-options-v1|{seed}|{image_filename}|{question_index}".encode()
    rng_seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
    rng = random.Random(rng_seed)
    distractors = [label for label in vocabulary if label != answer]
    chosen = [answer, *rng.sample(distractors, min(3, len(distractors)))]
    rng.shuffle(chosen)
    return tuple(chosen)


def option_probabilities(
    answer_logits: torch.Tensor,
    options: Sequence[str],
) -> torch.Tensor:
    """Softmax only the supplied CLEVR answer options, preserving their order."""
    if answer_logits.ndim != 1 or answer_logits.numel() != len(ANSWER_LABELS):
        raise ValueError(f"expected one logit per {len(ANSWER_LABELS)} CLEVR answer labels")
    if not torch.isfinite(answer_logits).all():
        raise FloatingPointError("answer logits must be finite")
    if not 2 <= len(options) <= 4:
        raise ValueError("between two and four options are required")
    if len(options) != len(set(options)):
        raise ValueError("options must be unique")
    try:
        indices = [ANSWER_LABELS.index(option) for option in options]
    except ValueError as exc:
        raise ValueError("option is outside the CLEVR answer vocabulary") from exc
    if len({ANSWER_TO_KIND[option] for option in options}) != 1:
        raise ValueError("options must share one answer kind")
    return torch.softmax(answer_logits[indices], dim=0)
