"""Paired metrics for frozen R2-M attempt outputs."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np


def binary_metric_summary(
    probabilities: np.ndarray,
    targets: np.ndarray,
    *,
    bins: int = 15,
) -> dict[str, float | int]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    if (probabilities.ndim != 2 or probabilities.shape != (len(targets), 2)
            or not len(targets) or bins < 1):
        raise ValueError("expected nonempty binary probabilities and aligned targets")
    if (not np.isfinite(probabilities).all() or (probabilities < 0).any()
            or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
            or not np.isin(targets, (0, 1)).all()):
        raise ValueError("probabilities must be normalized and targets binary")
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = (predictions == targets).astype(np.float64)
    one_hot = np.eye(2, dtype=np.float64)[targets]
    chosen = np.maximum(probabilities[np.arange(len(targets)), targets], 1e-12)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for index, lower in enumerate(edges[:-1]):
        upper = edges[index + 1]
        upper_mask = confidence <= upper if index == bins - 1 else confidence < upper
        mask = (confidence >= lower) & upper_mask
        if mask.any():
            ece += float(mask.mean() * abs(confidence[mask].mean() - correct[mask].mean()))
    return {
        "examples": int(len(targets)),
        "accuracy": float(correct.mean()),
        "nll": float(-np.log(chosen).mean()),
        "brier": float(((probabilities - one_hot) ** 2).sum(axis=1).mean()),
        "ece_15bin": ece,
        "mean_confidence": float(confidence.mean()),
    }


def paired_choice_rows(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[str, dict[int, Mapping[str, object]]] = defaultdict(dict)
    for row in rows:
        pair_id = str(row.get("pair_id", ""))
        target = row.get("target")
        if not pair_id or type(target) is not int or target not in (0, 1):
            raise ValueError("prediction rows require pair IDs and binary targets")
        if target in grouped[pair_id]:
            raise ValueError(f"duplicate target for pair {pair_id}")
        grouped[pair_id][target] = row

    result: list[dict[str, object]] = []
    for pair_id, candidates in sorted(grouped.items()):
        if set(candidates) != {0, 1}:
            raise ValueError(f"pair {pair_id} must have one row for each candidate")
        scores = []
        for target in (0, 1):
            probabilities = np.asarray(candidates[target].get("probabilities"), dtype=np.float64)
            if (probabilities.shape != (2,) or not np.isfinite(probabilities).all()
                    or (probabilities < 0).any()
                    or not np.isclose(probabilities.sum(), 1.0, atol=1e-6)):
                raise ValueError(f"pair {pair_id} has invalid candidate probabilities")
            probability = float(np.clip(probabilities[1], 1e-12, 1.0 - 1e-12))
            scores.append(float(np.log(probability / (1.0 - probability))))
        score_array = np.asarray(scores, dtype=np.float64)
        shifted = score_array - score_array.max()
        pair_probabilities = np.exp(shifted)
        pair_probabilities /= pair_probabilities.sum()
        positive = candidates[1]
        negative = candidates[0]
        if (positive.get("anchor_sample_id") != negative.get("anchor_sample_id")
                or positive.get("negative_relation") != negative.get("negative_relation")):
            raise ValueError(f"pair {pair_id} candidate identity metadata disagrees")
        result.append({
            "pair_id": pair_id,
            "anchor_sample_id": positive.get("anchor_sample_id"),
            "negative_relation": positive.get("negative_relation"),
            "target": 1,
            "prediction": int(pair_probabilities.argmax()),
            "probabilities": pair_probabilities.tolist(),
        })
    return result
