"""Paired connected-group uncertainty analysis for the KUCI validation probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scripts.build_kuci_train_grouped_split import (
        _UnionFind,
        content_key,
        normalize_text,
        sha256_file,
    )
except ModuleNotFoundError:
    from build_kuci_train_grouped_split import _UnionFind, content_key, normalize_text, sha256_file


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def connected_groups(rows: list[dict[str, Any]]) -> list[int]:
    union_find = _UnionFind(len(rows))
    first_by_key: dict[tuple[str, str], int] = {}
    for index, row in enumerate(rows):
        keys = (
            ("event_pair", normalize_text(str(row["core_event_pair"]))),
            ("context", normalize_text(str(row["context"]))),
            ("content", content_key(row)),
        )
        for key in keys:
            prior = first_by_key.setdefault(key, index)
            union_find.union(index, prior)
    roots = [union_find.find(index) for index in range(len(rows))]
    root_to_group = {root: group for group, root in enumerate(sorted(set(roots)))}
    return [root_to_group[root] for root in roots]


def _prediction_arrays(
    rows: list[dict[str, Any]], predictions: list[dict[str, Any]]
) -> tuple[np.ndarray, np.ndarray]:
    expected_ids = [str(row["id"]) for row in rows]
    observed_ids = [str(row["id"]) for row in predictions]
    if expected_ids != observed_ids:
        raise ValueError("prediction IDs/order do not match frozen validation rows")
    targets = np.asarray(["abcd".index(str(row["label"])) for row in rows], dtype=np.int64)
    for row, prediction in zip(rows, predictions, strict=True):
        if str(prediction["target"]) != str(row["label"]):
            raise ValueError("prediction target differs from frozen validation label")
    probabilities = np.asarray([row["probabilities_abcd"] for row in predictions], dtype=np.float64)
    if probabilities.shape != (len(rows), 4) or not np.isfinite(probabilities).all():
        raise ValueError("probabilities must be finite [N, 4]")
    if np.any(probabilities < 0) or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5):
        raise ValueError("probabilities must be normalized")
    return targets, probabilities


def _per_example_losses(
    targets: np.ndarray, probabilities: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(targets))
    correct = (probabilities.argmax(axis=1) == targets).astype(np.float64)
    nll = -np.log(np.clip(probabilities[indices, targets], 1e-12, 1.0))
    one_hot = np.eye(4, dtype=np.float64)[targets]
    brier = np.square(probabilities - one_hot).sum(axis=1)
    return correct, nll, brier


def paired_group_bootstrap(
    rows: list[dict[str, Any]],
    baseline_predictions: list[dict[str, Any]],
    candidate_predictions: list[dict[str, Any]],
    *,
    replicates: int = 10_000,
    seed: int = 1_729,
) -> dict[str, Any]:
    if not rows or replicates < 1:
        raise ValueError("bootstrap requires validation rows and positive replicates")
    targets, baseline = _prediction_arrays(rows, baseline_predictions)
    candidate_targets, candidate = _prediction_arrays(rows, candidate_predictions)
    if not np.array_equal(targets, candidate_targets):
        raise ValueError("models do not share validation labels")

    groups = np.asarray(connected_groups(rows), dtype=np.int64)
    group_count = int(groups.max()) + 1
    group_sizes = np.bincount(groups, minlength=group_count).astype(np.float64)
    metrics = {}
    rng = np.random.default_rng(seed)
    deltas = []
    for model in (baseline, candidate):
        deltas.append(_per_example_losses(targets, model))
    names = ("accuracy", "nll", "brier")
    group_delta_sums: dict[str, np.ndarray] = {}
    point_deltas: dict[str, float] = {}
    for metric_index, name in enumerate(names):
        difference = deltas[1][metric_index] - deltas[0][metric_index]
        group_delta_sums[name] = np.bincount(groups, weights=difference, minlength=group_count)
        point_deltas[name] = float(difference.mean())

    bootstrap_values = {name: np.empty(replicates, dtype=np.float64) for name in names}
    offset, batch_size = 0, 128
    probabilities = np.full(group_count, 1.0 / group_count)
    while offset < replicates:
        count = min(batch_size, replicates - offset)
        multiplicities = rng.multinomial(group_count, probabilities, size=count)
        denominators = multiplicities @ group_sizes
        for name in names:
            numerators = multiplicities @ group_delta_sums[name]
            bootstrap_values[name][offset : offset + count] = numerators / denominators
        offset += count

    for name in names:
        values = bootstrap_values[name]
        metrics[name] = {
            "candidate_minus_baseline": point_deltas[name],
            "percentile_95_ci": [
                float(np.quantile(values, 0.025)),
                float(np.quantile(values, 0.975)),
            ],
        }
    return {
        "baseline": "ruri-v3-30m",
        "candidate": "line-distilbert-ja-first4",
        "resampling_unit": (
            "connected validation component (event pair, normalized context, option content)"
        ),
        "connected_groups": group_count,
        "validation_rows": len(rows),
        "replicates": replicates,
        "seed": seed,
        "metrics": metrics,
        "limitations": [
            "development validation was also used for checkpoint selection",
            "intervals are descriptive and do not constitute a blind test",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--comparison-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=1_729)
    args = parser.parse_args()
    rows = read_jsonl(args.data_dir / "validation.jsonl")
    baseline_path = args.comparison_dir / "ruri-v3-30m" / "best-validation-predictions.jsonl"
    candidate_path = (
        args.comparison_dir / "line-distilbert-ja-first4" / "best-validation-predictions.jsonl"
    )
    result = paired_group_bootstrap(
        rows,
        read_jsonl(baseline_path),
        read_jsonl(candidate_path),
        replicates=args.replicates,
        seed=args.seed,
    )
    result["validation_jsonl_sha256"] = sha256_file(args.data_dir / "validation.jsonl")
    result["baseline_predictions_sha256"] = sha256_file(baseline_path)
    result["candidate_predictions_sha256"] = sha256_file(candidate_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
