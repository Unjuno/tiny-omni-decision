"""Aggregation helpers for the fixed synthetic AV validation cohort."""

from __future__ import annotations

import numpy as np


def group_accuracy(correct: np.ndarray, group_ids: np.ndarray) -> dict[str, float]:
    correct = np.asarray(correct, dtype=np.float64)
    group_ids = np.asarray(group_ids).astype(str)
    if correct.ndim != 1 or group_ids.ndim != 1 or len(correct) != len(group_ids):
        raise ValueError("correct flags and group IDs must be matching vectors")
    if len(correct) == 0 or not np.isfinite(correct).all():
        raise ValueError("group accuracy requires non-empty finite values")
    values = {}
    for group in sorted(set(group_ids.tolist())):
        selected = correct[group_ids == group]
        values[group] = float(selected.mean())
    return values


def paired_group_bootstrap_difference(
    reference: dict[str, float],
    candidate: dict[str, float],
    *,
    replicates: int = 10_000,
    seed: int = 17,
) -> dict[str, float | int]:
    if replicates < 1 or seed < 0:
        raise ValueError("replicates must be positive and seed non-negative")
    if not reference or reference.keys() != candidate.keys():
        raise ValueError("paired bootstrap inputs must have the same non-empty group IDs")
    groups = sorted(reference)
    ref = np.asarray([reference[group] for group in groups], dtype=np.float64)
    cand = np.asarray([candidate[group] for group in groups], dtype=np.float64)
    if not np.isfinite(ref).all() or not np.isfinite(cand).all():
        raise ValueError("bootstrap values must be finite")
    differences = cand - ref
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, len(groups), size=(replicates, len(groups)))
    boot = differences[draws].mean(axis=1)
    lower, upper = np.percentile(boot, [2.5, 97.5])
    return {
        "group_count": len(groups),
        "replicates": replicates,
        "seed": seed,
        "mean_accuracy_difference": float(differences.mean()),
        "lower_95_percentile": float(lower),
        "upper_95_percentile": float(upper),
    }


def replay_prediction_metrics(records: list[dict], ece_bins: int = 15) -> dict[str, float]:
    """Recompute published metrics from serialized probabilities without model code."""
    if not records or ece_bins < 1:
        raise ValueError("prediction replay requires records and positive ECE bins")
    probabilities = np.asarray([row["probabilities"] for row in records], dtype=np.float64)
    targets = np.asarray([row["target"] for row in records], dtype=np.int64)
    if probabilities.shape != (len(records), 2):
        raise ValueError("each prediction must contain two class probabilities")
    if (
        not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
        or np.any(probabilities > 1)
        or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
        or np.any((targets < 0) | (targets > 1))
    ):
        raise ValueError("saved probabilities or targets are invalid")
    predictions = probabilities.argmax(axis=1)
    correct = predictions == targets
    confidence = probabilities.max(axis=1)
    ece = 0.0
    for bin_index in range(ece_bins):
        lower = bin_index / ece_bins
        upper = (bin_index + 1) / ece_bins
        selected = (confidence > lower) & (confidence <= upper)
        if selected.any():
            observed_accuracy = float(correct[selected].mean())
            observed_confidence = float(confidence[selected].mean())
            ece += float(selected.mean()) * abs(observed_accuracy - observed_confidence)
    one_hot = np.eye(2, dtype=np.float64)[targets]
    group_ids = np.asarray([str(row["group_id"]) for row in records])
    pair_ids = np.asarray([str(row["pair_id"]) for row in records])
    groups = group_accuracy(correct.astype(np.float64), group_ids)
    pair_accuracy = []
    for pair_id in sorted(set(pair_ids.tolist())):
        pair_correct = correct[pair_ids == pair_id]
        if len(pair_correct) != 2:
            raise ValueError("each saved pair must contain both query polarities")
        pair_accuracy.append(float(pair_correct.all()))
    return {
        "accuracy": float(correct.mean()),
        "nll": float(
            -np.log(np.maximum(probabilities[np.arange(len(records)), targets], 1e-15)).mean()
        ),
        "brier": float(((probabilities - one_hot) ** 2).sum(axis=1).mean()),
        "ece": ece,
        "mean_confidence": float(confidence.mean()),
        "pair_exact_accuracy": float(np.mean(pair_accuracy)),
        "group_macro_accuracy": float(np.mean(list(groups.values()))),
        "group_count": len(groups),
    }
