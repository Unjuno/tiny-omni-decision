"""Compute paired component-cluster bootstrap deltas from train-only HABIT OOF."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def connected_components(records: list[dict[str, Any]]) -> list[list[str]]:
    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        if parent[node] != node:
            parent[node] = find(parent[node])
        return parent[node]

    def union(left: str, right: str) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[b] = a

    for row in records:
        union(f"task:{row['sid']}", f"video:{row['video_sha256']}")
    groups: dict[str, list[str]] = defaultdict(list)
    for row in records:
        groups[find(f"task:{row['sid']}")].append(row["id"])
    return sorted((sorted(ids) for ids in groups.values()), key=lambda ids: ids[0])


def _row_losses(prediction: dict[str, Any]) -> tuple[float, float, float]:
    probabilities = prediction["probabilities"]
    target = int(prediction["target_index"])
    accuracy = float(int(prediction["prediction_index"]) == target)
    nll = -__import__("math").log(max(float(probabilities[target]), 1e-12))
    brier = sum((float(p) - float(i == target)) ** 2
                for i, p in enumerate(probabilities))
    return accuracy, nll, brier


def paired_component_bootstrap(
    report: dict[str, Any], records: list[dict[str, Any]], *, draws: int = 10_000,
    seed: int = 1701,
) -> dict[str, Any]:
    train = [row for row in records if row.get("split") == "train"]
    components = connected_components(train)
    component_by_id = {sample_id: component for component, ids in enumerate(components)
                       for sample_id in ids}
    results: dict[str, Any] = {}
    metrics_by_mode = report["out_of_fold_metrics_by_mode_and_epoch"]
    for epoch in report["curve_epochs"]:
        for comparator in ("video_only", "state_only"):
            left = metrics_by_mode["video_state"][str(epoch)]["predictions"]
            right = metrics_by_mode[comparator][str(epoch)]["predictions"]
            if ([row["sample_id"] for row in left] != [row["sample_id"] for row in right]
                    or len(left) != len(train)):
                raise ValueError("paired OOF predictions do not align exactly with train rows")
            left_by_id = {row["sample_id"]: row for row in left}
            right_by_id = {row["sample_id"]: row for row in right}
            if set(left_by_id) != set(component_by_id):
                raise ValueError("OOF sample IDs differ from train package")
            grouped: dict[
                int, list[tuple[tuple[float, ...], tuple[float, ...]]]
            ] = defaultdict(list)
            for sample_id in sorted(component_by_id):
                left_loss = _row_losses(left_by_id[sample_id])
                right_loss = _row_losses(right_by_id[sample_id])
                grouped[component_by_id[sample_id]].append((left_loss, right_loss))
            point = []
            for metric_index in range(3):
                lsum = sum(pair[0][metric_index] for rows in grouped.values() for pair in rows)
                rsum = sum(pair[1][metric_index] for rows in grouped.values() for pair in rows)
                point.append((lsum - rsum) / len(train))
            rng = random.Random(seed + int(epoch) * 11 + (0 if comparator == "video_only" else 1))
            samples: list[list[float]] = [[], [], []]
            component_ids = list(range(len(components)))
            for _ in range(draws):
                chosen = [rng.choice(component_ids) for _ in component_ids]
                sampled_pairs = [pair for component in chosen for pair in grouped[component]]
                for metric_index in range(3):
                    samples[metric_index].append(
                        sum(pair[0][metric_index] - pair[1][metric_index]
                            for pair in sampled_pairs) / len(sampled_pairs)
                    )
            contrasts = {}
            for name, values, point_delta in zip(
                ("accuracy", "nll", "brier"), samples, point, strict=True
            ):
                values.sort()
                contrasts[name] = {
                    "video_state_minus_comparator": point_delta,
                    "cluster_bootstrap_95pct_percentile_ci": [
                        values[int(0.025 * draws)], values[min(draws - 1, int(0.975 * draws))]
                    ],
                }
            results[f"epoch_{epoch}_video_state_minus_{comparator}"] = contrasts
    return {
        "method": "paired cluster bootstrap over connected task/video identity components",
        "draws": draws,
        "seed": seed,
        "component_count": len(components),
        "train_examples": len(train),
        "validation_split_used": False,
        "sealed_audit_accessed": False,
        "contrasts": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    examples_path = args.package_dir / "transition-examples.json"
    examples = json.loads(examples_path.read_text(encoding="utf-8"))["records"]
    result = paired_component_bootstrap(report, examples)
    result["oof_report_sha256"] = hashlib.sha256(args.report.read_bytes()).hexdigest()
    result["train_examples_sha256"] = hashlib.sha256(examples_path.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "contrasts"}, indent=2))


if __name__ == "__main__":
    main()
