"""Train-group folds for the task-native RoboMNIST 10-option digit Decision."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict

import numpy as np

_GROUP = re.compile(r"^Ar(?P<arm>\d+)_V(?P<speed>[^_]+)_Ac(?P<digit>\d+)_Rp\d+_UNC\d+$")


def parse_group(group_id: str) -> tuple[str, int]:
    match = _GROUP.fullmatch(group_id)
    if not match:
        raise ValueError(f"unrecognized RoboMNIST group identity: {group_id}")
    digit = int(match.group("digit"))
    if digit not in range(10):
        raise ValueError("activity/digit label outside [0,9]")
    return f"Ar{match.group('arm')}_V{match.group('speed')}_Ac{digit}", digit


def condition_folds(group_ids: set[str], *, folds=5, seed=17) -> dict[str, int]:
    if folds < 2 or len(group_ids) < folds:
        raise ValueError("insufficient groups/folds")
    conditions = defaultdict(set)
    observed = {}
    for group in sorted(group_ids):
        condition, digit = parse_group(group)
        conditions[digit].add(condition)
        observed[group] = digit
    if set(conditions) != set(range(10)):
        raise ValueError("TRAIN must contain all ten digit classes")
    assigned = {}
    offset = 0
    for digit in range(10):
        ordered = sorted(
            conditions[digit], key=lambda c: (hashlib.sha256(f"{seed}:{c}".encode()).hexdigest(), c)
        )
        if len(ordered) < folds:
            raise ValueError("every digit needs at least one held condition per fold")
        for idx, condition in enumerate(ordered):
            fold = (offset + idx) % folds
            for group in group_ids:
                parsed, _ = parse_group(group)
                if parsed == condition:
                    if group in assigned:
                        raise ValueError("duplicate group assignment")
                    assigned[group] = fold
        offset = (offset + len(ordered)) % folds
    if assigned.keys() != group_ids:
        raise ValueError("incomplete group assignment")
    return assigned


def option_ordered_probabilities(logits, option_ids):
    import torch

    ids = list(option_ids)
    if len(ids) != 10 or set(ids) != set(range(10)):
        raise ValueError("options must be a permutation of 0..9")
    if logits.ndim != 2 or logits.shape[1] != 10 or not torch.isfinite(logits).all():
        raise ValueError("logits must be finite [batch,10]")
    return torch.softmax(logits.float(), dim=-1)[:, ids]


def multiclass_metrics(probabilities, targets, *, bins=15):
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(targets, dtype=np.int64)
    if (
        p.shape != (len(y), 10)
        or np.any(y < 0)
        or np.any(y > 9)
        or not np.isfinite(p).all()
        or np.any(p < 0)
        or not np.allclose(p.sum(1), 1)
    ):
        raise ValueError("expected normalized finite ten-choice probabilities and labels")
    pred = p.argmax(1)
    correct = pred == y
    confidence = p.max(1)
    nll = -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1))
    brier = ((p - np.eye(10)[y]) ** 2).sum(1)
    ece = 0.0
    for idx in range(bins):
        mask = (confidence >= idx / bins) & (
            (confidence < (idx + 1) / bins) if idx < bins - 1 else (confidence <= 1)
        )
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    by_class = {str(c): int(np.sum((pred == y) & (y == c))) for c in range(10)}
    counts = {str(c): int(np.sum(y == c)) for c in range(10)}
    accuracy_by_class = {
        str(c): (by_class[str(c)] / counts[str(c)] if counts[str(c)] else None) for c in range(10)
    }
    return {
        "count": int(len(y)),
        "accuracy": float(correct.mean()),
        "nll": float(nll.mean()),
        "brier": float(brier.mean()),
        "ece_15": float(ece),
        "mean_confidence": float(confidence.mean()),
        "class_count": counts,
        "accuracy_by_class": accuracy_by_class,
    }
