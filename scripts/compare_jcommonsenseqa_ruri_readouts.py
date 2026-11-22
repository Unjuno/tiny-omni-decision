"""Compare two saved JCommonsenseQA readouts with question-group bootstrap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _paired_group_bootstrap(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    if len(baseline) != len(candidate):
        raise ValueError("candidate prediction counts differ")
    for left, right in zip(baseline, candidate, strict=True):
        keys = ("sample_id", "content_group_id", "target_index")
        if any(left[key] != right[key] for key in keys):
            raise ValueError("prediction rows do not align on sample/group/target")
    groups = sorted({row["content_group_id"] for row in baseline})
    group_to_index = {group: index for index, group in enumerate(groups)}
    sample_groups = np.asarray([group_to_index[row["content_group_id"]] for row in baseline])
    targets = np.asarray([int(row["target_index"]) for row in baseline])

    def row_values(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        probabilities = np.asarray([row["probabilities"] for row in rows], dtype=np.float64)
        if probabilities.shape != (len(rows), 5) or not np.isfinite(probabilities).all():
            raise ValueError("invalid five-choice probabilities")
        if np.any(probabilities < 0) or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5):
            raise ValueError("probabilities must be normalized")
        prediction = probabilities.argmax(axis=1)
        accuracy = (prediction == targets).astype(np.float64)
        nll = -np.log(np.maximum(probabilities[np.arange(len(rows)), targets], 1e-12))
        one_hot = np.eye(5, dtype=np.float64)[targets]
        brier = ((probabilities - one_hot) ** 2).sum(axis=1)
        return accuracy, nll, brier

    base_values = row_values(baseline)
    cand_values = row_values(candidate)
    differences = [right - left for left, right in zip(base_values, cand_values, strict=True)]
    group_sums = np.zeros((len(groups), 3), dtype=np.float64)
    group_counts = np.zeros(len(groups), dtype=np.int64)
    np.add.at(group_counts, sample_groups, 1)
    for metric_index, values in enumerate(differences):
        np.add.at(group_sums[:, metric_index], sample_groups, values)
    estimates = np.asarray([float(values.mean()) for values in differences])
    rng = np.random.default_rng(seed)
    samples = np.empty((iterations, 3), dtype=np.float64)
    for start in range(0, iterations, 128):
        count = min(128, iterations - start)
        picked = rng.integers(0, len(groups), size=(count, len(groups)), dtype=np.int32)
        denominator = group_counts[picked].sum(axis=1)
        samples[start : start + count] = group_sums[picked].sum(axis=1) / denominator[:, None]
    names = ("accuracy", "nll", "brier")
    return {
        "group_count": len(groups),
        "iterations": iterations,
        "seed": seed,
        "resampling_unit": (
            "normalized exact-question content_group_id; all rows in each group travel together"
        ),
        "candidate_minus_baseline": {
            name: {
                "point_difference": float(estimates[index]),
                "bootstrap_95_percentile_interval": [
                    float(np.quantile(samples[:, index], 0.025)),
                    float(np.quantile(samples[:, index], 0.975)),
                ],
            }
            for index, name in enumerate(names)
        },
        "note": (
            "Descriptive only: this validation was used for checkpoint selection "
            "and candidate comparison."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    baseline, candidate = _load(args.baseline), _load(args.candidate)
    result = _paired_group_bootstrap(
        baseline, candidate, iterations=args.iterations, seed=args.seed
    )
    result["baseline_rows"] = len(baseline)
    result["candidate_rows"] = len(candidate)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
