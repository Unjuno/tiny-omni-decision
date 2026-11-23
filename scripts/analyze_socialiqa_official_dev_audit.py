"""Context-cluster uncertainty for one fixed Social IQa official-dev audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines()]
    grouped: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
    seen_ids: set[str] = set()
    for row in rows:
        if row["sample_id"] in seen_ids:
            raise ValueError("duplicate public-dev sample ID")
        seen_ids.add(row["sample_id"])
        probs = np.asarray(row["probabilities"], dtype=np.float64)
        target = int(row["target"])
        if probs.shape != (3,) or not np.isfinite(probs).all() or target not in range(3):
            raise ValueError("malformed saved prediction")
        one_hot = np.eye(3, dtype=np.float64)[target]
        nll = -float(np.log(max(float(probs[target]), 1e-12)))
        brier = float(np.square(probs - one_hot).sum())
        accuracy = float(int(np.argmax(probs)) == target)
        grouped[row["source_context_sha256"]].append((accuracy, nll, brier))
    if len(rows) != 1_954 or len(grouped) != 1_930:
        raise ValueError("unexpected frozen official-dev prediction/group count")

    states = sorted(grouped)
    cluster_sums = np.asarray(
        [
            [len(grouped[state]), *np.asarray(grouped[state], dtype=np.float64).sum(axis=0)]
            for state in states
        ],
        dtype=np.float64,
    )
    observed = cluster_sums.sum(axis=0)[1:] / cluster_sums[:, 0].sum()
    baseline = np.asarray([1 / 3, np.log(3), 2 / 3], dtype=np.float64)

    rng = np.random.Generator(np.random.PCG64(args.seed))
    samples = np.empty((args.replicates, 3), dtype=np.float64)
    for index in range(args.replicates):
        selected = rng.integers(0, len(states), size=len(states))
        counts = np.bincount(selected, minlength=len(states))
        sums = (cluster_sums * counts[:, None]).sum(axis=0)
        samples[index] = sums[1:] / sums[0]

    names = ("accuracy", "nll", "brier")
    metrics: dict[str, Any] = {}
    for column, name in enumerate(names):
        values = samples[:, column]
        diffs = values - baseline[column]
        metrics[name] = {
            "observed": float(observed[column]),
            "baseline": float(baseline[column]),
            "observed_minus_uniform": float(observed[column] - baseline[column]),
            "context_cluster_percentile_95_interval": [
                float(np.quantile(values, 0.025)),
                float(np.quantile(values, 0.975)),
            ],
            "difference_from_uniform_percentile_95_interval": [
                float(np.quantile(diffs, 0.025)),
                float(np.quantile(diffs, 0.975)),
            ],
        }
    report = {
        "status": "complete_posthoc_context_cluster_bootstrap",
        "method": (
            "PCG64 percentile bootstrap resampling normalized context groups "
            "with replacement; all questions within each selected context remain together"
        ),
        "replicates": args.replicates,
        "seed": args.seed,
        "prediction_sha256": _sha256(args.predictions),
        "questions": len(rows),
        "context_groups": len(states),
        "metrics": metrics,
        "limits": (
            "descriptive uncertainty for the single already-opened public-dev "
            "audit; not for further model selection or retuning"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
