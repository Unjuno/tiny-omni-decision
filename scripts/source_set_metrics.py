"""Compute per-record set and Bernoulli metrics for candidate marginals."""

from __future__ import annotations

import math


def binary_metrics(probabilities, labels):
    if not probabilities or len(probabilities) != len(labels):
        raise ValueError("invalid probability and label lengths")
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
        raise ValueError("invalid probability")
    if any(y not in (0, 1) for y in labels):
        raise ValueError("invalid binary label")
    n = len(labels)
    brier = sum((p - y) ** 2 for p, y in zip(probabilities, labels, strict=True)) / n
    eps = 1e-12
    nll = (
        -sum(
            y * math.log(max(p, eps)) + (1 - y) * math.log(max(1 - p, eps))
            for p, y in zip(probabilities, labels, strict=True)
        )
        / n
    )
    ece = 0.0
    for i in range(10):
        values = [
            (p, y)
            for p, y in zip(probabilities, labels, strict=True)
            if i / 10 <= p and (p < (i + 1) / 10 if i < 9 else p <= 1)
        ]
        if values:
            ece += (
                len(values)
                / n
                * abs(
                    sum(p for p, _ in values) / len(values)
                    - sum(y for _, y in values) / len(values)
                )
            )
    return {"n": n, "nll": nll, "brier": brier, "ece_10": ece}


def score_record(candidate_ids, targets, selected, conditional, none_probability):
    ids = list(candidate_ids)
    if not ids or len(set(ids)) != len(ids) or len(ids) != len(conditional):
        raise ValueError("invalid candidate identities or probabilities")
    true_set, selected_set = set(targets), set(selected)
    if not true_set <= set(ids) or not selected_set <= set(ids) | {"none_visible"}:
        raise ValueError("unknown candidate")
    if not selected_set or ("none_visible" in selected_set and len(selected_set) != 1):
        raise ValueError("invalid selected set")
    for p in [none_probability, *conditional]:
        if not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("invalid probability")
    expected = true_set if true_set else {"none_visible"}
    candidate_probabilities = [(1 - none_probability) * p for p in conditional]
    membership = binary_metrics(candidate_probabilities, [int(i in true_set) for i in ids])
    return {
        "exact": selected_set == expected,
        "none_correct": ("none_visible" in selected_set) == (not true_set),
        "membership_nll": membership["nll"],
        "membership_brier": membership["brier"],
        "none_nll": binary_metrics([none_probability], [int(not true_set)])["nll"],
        "none_brier": (none_probability - int(not true_set)) ** 2,
    }
