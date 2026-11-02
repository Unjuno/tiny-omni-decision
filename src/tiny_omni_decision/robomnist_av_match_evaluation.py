"""Dependency-light validation and metric contracts for RoboMNIST AV matching."""

from __future__ import annotations

import numpy as np


def validate_examples(manifest: dict) -> dict[str, list[dict]]:
    """Fail closed on malformed 2×2 pair units or cross-fold source groups."""
    output = {}
    for split in ("train", "validation"):
        rows = manifest["examples"][split]
        pair_rows: dict[str, list[dict]] = {}
        for row in rows:
            if row["target"] not in (0, 1):
                raise ValueError("AV match target must be binary")
            if (row["audio_group"] == row["video_group"]) != (row["target"] == 1):
                raise ValueError("binary target disagrees with source-group match identity")
            pair_rows.setdefault(row["pair_unit_id"], []).append(row)
        for pair_id, examples in pair_rows.items():
            if len(examples) != 4 or sum(row["target"] for row in examples) != 2:
                raise ValueError(f"pair unit must contain four balanced examples: {pair_id}")
            if len({row["condition"] for row in examples}) != 1:
                raise ValueError(f"pair unit spans multiple conditions: {pair_id}")
            groups = sorted(
                {
                    group_id
                    for row in examples
                    for group_id in (row["audio_group"], row["video_group"])
                }
            )
            observed = {(row["audio_group"], row["video_group"]) for row in examples}
            expected = (
                {
                    (groups[0], groups[0]),
                    (groups[0], groups[1]),
                    (groups[1], groups[1]),
                    (groups[1], groups[0]),
                }
                if len(groups) == 2
                else set()
            )
            if observed != expected:
                raise ValueError(f"pair unit is not the frozen 2x2 AV match design: {pair_id}")
        output[split] = rows
    train_groups = {row[key] for row in output["train"] for key in ("audio_group", "video_group")}
    validation_groups = {
        row[key] for row in output["validation"] for key in ("audio_group", "video_group")
    }
    if train_groups & validation_groups:
        raise ValueError("audio/video source group overlap between train and validation")
    return output


def classification_metrics(
    probabilities: np.ndarray, target: np.ndarray, *, bins: int = 15
) -> dict[str, float | int]:
    target = np.asarray(target, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if (
        probabilities.ndim != 2
        or probabilities.shape != (len(target), 2)
        or not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
        or np.any(probabilities > 1)
        or bins < 1
        or not np.isclose(probabilities.sum(-1), 1).all()
    ):
        raise ValueError("binary probabilities must be finite normalized two-option rows")
    if np.any((target != 0) & (target != 1)):
        raise ValueError("binary targets must be zero or one")
    predictions = probabilities.argmax(-1)
    confidence = probabilities.max(-1)
    correct = predictions == target
    nll = -np.log(np.clip(probabilities[np.arange(len(target)), target], 1e-12, 1.0))
    one_hot = np.eye(2, dtype=np.float64)[target]
    brier = np.square(probabilities - one_hot).sum(-1)
    ece = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for index in range(bins):
        mask = (confidence >= edges[index]) & (
            confidence < edges[index + 1] if index < bins - 1 else confidence <= 1.0
        )
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return {
        "count": len(target),
        "accuracy": float(correct.mean()),
        "nll": float(nll.mean()),
        "brier": float(brier.mean()),
        "ece_15": float(ece),
        "mean_confidence": float(confidence.mean()),
    }


def paired_bootstrap_delta(
    correct_joint: np.ndarray,
    correct_unimodal: np.ndarray,
    pair_ids: np.ndarray,
    *,
    replicates: int = 2000,
    seed: int = 17,
) -> dict:
    if correct_joint.shape != correct_unimodal.shape or correct_joint.shape != pair_ids.shape:
        raise ValueError("paired bootstrap inputs must have matching one-dimensional shapes")
    unique_pairs = np.unique(pair_ids)
    if len(unique_pairs) < 200:
        raise ValueError("paired validation bootstrap requires at least 200 independent pair units")
    pair_to_rows = [np.flatnonzero(pair_ids == pair_id) for pair_id in unique_pairs]
    if any(len(indices) != 4 for indices in pair_to_rows):
        raise ValueError("each bootstrap unit must contain exactly four decision rows")
    if replicates < 1:
        raise ValueError("bootstrap replicate count must be positive")
    rng = np.random.default_rng(seed)
    deltas = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sample = rng.integers(0, len(pair_to_rows), size=len(pair_to_rows))
        rows = np.concatenate([pair_to_rows[unit] for unit in sample])
        deltas[index] = correct_joint[rows].mean() - correct_unimodal[rows].mean()
    return {
        "independent_pair_units": len(unique_pairs),
        "replicates": replicates,
        "seed": seed,
        "unit": "paired 4-example source pair unit sampled with replacement",
        "accuracy_delta_joint_minus_best_unimodal": float(
            correct_joint.mean() - correct_unimodal.mean()
        ),
        "percentile_95_ci": [
            float(np.quantile(deltas, 0.025)),
            float(np.quantile(deltas, 0.975)),
        ],
    }
