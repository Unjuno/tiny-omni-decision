"""Paired session-cluster bootstrap for saved HoloAssist validation predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from scripts.audit_holoassist_metadata import sha256_file


def load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("prediction file must contain a nonempty JSON list")
    result = {}
    for row in rows:
        sample_id = str(row.get("id", ""))
        probabilities = np.asarray(row.get("probabilities"), dtype=np.float64)
        if (
            not sample_id
            or sample_id in result
            or not str(row.get("session_id", ""))
            or row.get("target") not in (0, 1)
            or probabilities.shape != (2,)
            or not np.isfinite(probabilities).all()
            or np.any(probabilities < 0)
            or not math.isclose(float(probabilities.sum()), 1.0, abs_tol=1e-6)
        ):
            raise ValueError(f"invalid or duplicate validation prediction row: {sample_id}")
        result[sample_id] = {
            "session_id": str(row["session_id"]),
            "target": int(row["target"]),
            "probabilities": probabilities,
        }
    return result


def _summary(target: np.ndarray, probabilities: np.ndarray) -> dict[str, float | None]:
    prediction = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = prediction == target
    positive = target == 1
    negative = target == 0
    sensitivity = float((prediction[positive] == 1).mean()) if positive.any() else float("nan")
    specificity = float((prediction[negative] == 0).mean()) if negative.any() else float("nan")
    nll = -np.log(np.clip(probabilities[np.arange(len(target)), target], 1e-12, 1.0))
    one_hot = np.eye(2, dtype=np.float64)[target]
    brier = np.square(probabilities - one_hot).sum(axis=1)
    ece = 0.0
    for index in range(15):
        lower, upper = index / 15, (index + 1) / 15
        selected = (confidence >= lower) & (
            (confidence < upper) if index < 14 else (confidence <= upper)
        )
        if selected.any():
            ece += selected.mean() * abs(confidence[selected].mean() - correct[selected].mean())
    return {
        "accuracy": float(correct.mean()),
        "nll": float(nll.mean()),
        "brier": float(brier.mean()),
        "ece_15": float(ece),
        "balanced_accuracy": (
            float((sensitivity + specificity) / 2)
            if positive.any() and negative.any()
            else None
        ),
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def paired_session_bootstrap(
    baseline: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
    *,
    replicates: int = 10_000,
    seed: int = 1701,
) -> dict[str, Any]:
    if set(baseline) != set(candidate):
        raise ValueError("paired prediction IDs must match exactly")
    ids = sorted(baseline)
    targets = np.asarray([baseline[key]["target"] for key in ids], dtype=np.int64)
    sessions = [baseline[key]["session_id"] for key in ids]
    if any(
        baseline[key]["target"] != candidate[key]["target"]
        or baseline[key]["session_id"] != candidate[key]["session_id"]
        for key in ids
    ):
        raise ValueError("paired predictions disagree on target or session identity")
    base_probability = np.stack([baseline[key]["probabilities"] for key in ids])
    candidate_probability = np.stack([candidate[key]["probabilities"] for key in ids])
    by_session: dict[str, list[int]] = defaultdict(list)
    for index, session in enumerate(sessions):
        by_session[session].append(index)
    session_names = sorted(by_session)
    if len(session_names) < 2 or replicates < 1:
        raise ValueError("at least two sessions and one bootstrap replicate are required")

    baseline_full = _summary(targets, base_probability)
    candidate_full = _summary(targets, candidate_probability)
    metric_names = tuple(baseline_full)
    rng = random.Random(seed)
    deltas = {
        metric: np.full(replicates, np.nan, dtype=np.float64) for metric in metric_names
    }
    for replicate in range(replicates):
        sampled_sessions = [rng.choice(session_names) for _ in session_names]
        indices = np.concatenate(
            [np.asarray(by_session[session], dtype=np.int64) for session in sampled_sessions]
        )
        base_metrics = _summary(targets[indices], base_probability[indices])
        candidate_metrics = _summary(targets[indices], candidate_probability[indices])
        for metric in metric_names:
            left, right = base_metrics[metric], candidate_metrics[metric]
            if left is not None and right is not None:
                deltas[metric][replicate] = right - left

    return {
        "replicates": replicates,
        "seed": seed,
        "sampling_unit": "session_id; sample sessions with replacement and retain all windows",
        "interval": "percentile 2.5th and 97.5th percentiles",
        "window_count": len(ids),
        "session_count": len(session_names),
        "paired_prediction_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "baseline_metrics": baseline_full,
        "candidate_metrics": candidate_full,
        "candidate_minus_baseline": {
            metric: (
                {
                    "median": float(np.nanmedian(values)),
                    "lower_95": float(np.nanquantile(values, 0.025)),
                    "upper_95": float(np.nanquantile(values, 0.975)),
                    "valid_replicates": int(np.isfinite(values).sum()),
                }
                if np.isfinite(values).any()
                else None
            )
            for metric, values in deltas.items()
        },
        "interpretation_limit": (
            "Validation-only descriptive uncertainty. If checkpoints were selected on these IDs, "
            "intervals are selection-biased and are not independent confirmation."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=1701)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite bootstrap report: {args.output}")
    result = paired_session_bootstrap(
        load_predictions(args.baseline),
        load_predictions(args.candidate),
        replicates=args.replicates,
        seed=args.seed,
    )
    result["baseline_predictions_sha256"] = sha256_file(args.baseline)
    result["candidate_predictions_sha256"] = sha256_file(args.candidate)
    result["baseline_prediction_path"] = str(args.baseline.resolve(strict=True))
    result["candidate_prediction_path"] = str(args.candidate.resolve(strict=True))
    result["created_at_utc"] = datetime.now(UTC).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
