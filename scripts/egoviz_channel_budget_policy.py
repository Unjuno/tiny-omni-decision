"""Fixed validation schedule and best-checkpoint rule."""

import math


def evaluation_epochs(max_epochs: int) -> list[int]:
    if max_epochs <= 0 or max_epochs % 8:
        raise ValueError("max epochs must be a positive multiple of eight")
    return [1, *range(8, max_epochs + 1, 8)]


def improves(value: float, best: float, min_delta: float = 1e-6) -> bool:
    if not math.isfinite(value):
        raise ValueError("nonfinite validation NLL")
    return value < best - min_delta
