"""Compare saved MASSIVE readout predictions with a normalized-text group bootstrap."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def read_predictions(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def paired_group_bootstrap(
    bert: list[dict[str, Any]],
    ruri: list[dict[str, Any]],
    *,
    iterations: int = 10_000,
    seed: int = 17,
) -> dict[str, Any]:
    if len(bert) != len(ruri) or not bert:
        raise ValueError("prediction sets must be nonempty and equal length")
    group_metrics: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"bert": [], "ruri": []}
    )
    for left, right in zip(bert, ruri, strict=True):
        for field in ("sample_id", "source_id", "content_group_id", "target_index"):
            if left[field] != right[field]:
                raise ValueError(f"prediction rows are misaligned at {field}")
        target = int(left["target_index"])
        group = str(left["content_group_id"])
        for name, row in (("bert", left), ("ruri", right)):
            probabilities = np.asarray(row["probabilities"], dtype=np.float64)
            if probabilities.shape != (60,) or not np.isfinite(probabilities).all():
                raise ValueError("each prediction must contain 60 finite probabilities")
            if not np.isclose(probabilities.sum(), 1.0, atol=1e-5):
                raise ValueError("probabilities must sum to one")
            brier = float(np.square(probabilities - np.eye(60)[target]).sum())
            group_metrics[group][name].append(
                float(probabilities.argmax() == target),
            )
            group_metrics[group][name].append(-float(np.log(max(probabilities[target], 1e-15))))
            group_metrics[group][name].append(brier)

    groups = sorted(group_metrics)
    n_metrics = 3
    sums = np.empty((len(groups), 2, n_metrics), dtype=np.float64)
    counts = np.empty(len(groups), dtype=np.int64)
    for index, group in enumerate(groups):
        # Each model list stores [accuracy, NLL, Brier] triples, flattened by row.
        for model_index, name in enumerate(("bert", "ruri")):
            values = np.asarray(group_metrics[group][name], dtype=np.float64).reshape(-1, 3)
            sums[index, model_index] = values.sum(axis=0)
        counts[index] = len(group_metrics[group]["bert"]) // n_metrics

    rng = np.random.Generator(np.random.PCG64(seed))
    differences = np.empty((iterations, n_metrics), dtype=np.float64)
    for iteration in range(iterations):
        sampled = rng.integers(0, len(groups), size=len(groups))
        denominator = counts[sampled].sum()
        model_means = sums[sampled].sum(axis=0) / denominator
        differences[iteration] = model_means[0] - model_means[1]
    names = ("accuracy", "nll", "brier")
    point = sums.sum(axis=0) / counts.sum()
    intervals = {
        name: {
            "difference_bert_minus_ruri": float((point[0] - point[1])[index]),
            "ci95_percentile": [
                float(np.quantile(differences[:, index], 0.025)),
                float(np.quantile(differences[:, index], 0.975)),
            ],
        }
        for index, name in enumerate(names)
    }
    return {
        "rows": len(bert),
        "normalized_text_groups": len(groups),
        "bootstrap": {
            "unit": "normalized exact utterance group",
            "iterations": iterations,
            "seed": seed,
            "rng": "NumPy PCG64",
            "interval": "percentile 95%",
        },
        "group_weighted_point_metrics": {
            model: {name: float(point[model_index, index]) for index, name in enumerate(names)}
            for model_index, model in enumerate(("bert", "ruri"))
        },
        "paired_differences": intervals,
        "selection_affected_validation": True,
        "independent_test": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bert-predictions", type=Path, required=True)
    parser.add_argument("--ruri-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    result = paired_group_bootstrap(
        read_predictions(args.bert_predictions),
        read_predictions(args.ruri_predictions),
        iterations=args.iterations,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
