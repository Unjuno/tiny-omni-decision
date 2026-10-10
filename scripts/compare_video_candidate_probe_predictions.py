"""Paired scene bootstrap for two saved CLEVRER candidate-probe predictions."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from statistics import fmean
from typing import Any


def _read_predictions(path: Path) -> dict[str, dict[str, Any]]:
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    indexed = {str(row["id"]): row for row in rows}
    if not rows or len(indexed) != len(rows):
        raise ValueError(f"empty predictions or duplicate IDs: {path}")
    return indexed


def _validate_pair(
    reference: dict[str, dict[str, Any]], candidate: dict[str, dict[str, Any]]
) -> None:
    if set(reference) != set(candidate):
        raise ValueError("prediction sample IDs differ")
    for sample_id, ref in reference.items():
        got = candidate[sample_id]
        for key in ("target", "options", "scene_index", "question_type"):
            if ref[key] != got[key]:
                raise ValueError(f"paired prediction metadata differs for {sample_id}: {key}")
        for row in (ref, got):
            probabilities = [float(value) for value in row["probabilities"]]
            if len(probabilities) != len(row["options"]):
                raise ValueError(f"probability width mismatch for {sample_id}")
            if any(not math.isfinite(value) or value < 0 for value in probabilities):
                raise ValueError(f"invalid probabilities for {sample_id}")
            if not math.isclose(math.fsum(probabilities), 1.0, abs_tol=1e-5):
                raise ValueError(f"probabilities do not sum to one for {sample_id}")


def _metrics(rows: list[dict[str, Any]], bins: int = 15) -> dict[str, float | int]:
    if not rows:
        raise ValueError("cannot measure an empty prediction sample")
    probabilities = [[float(value) for value in row["probabilities"]] for row in rows]
    targets = [row["options"].index(row["target"]) for row in rows]
    correct = [
        float(max(range(len(probability)), key=probability.__getitem__) == target)
        for probability, target in zip(probabilities, targets, strict=True)
    ]
    confidence = [max(probability) for probability in probabilities]
    nll = [
        -math.log(min(max(probability[target], 1e-12), 1.0))
        for probability, target in zip(probabilities, targets, strict=True)
    ]
    brier = []
    for probability, target in zip(probabilities, targets, strict=True):
        brier.append(
            math.fsum(
                (value - float(index == target)) ** 2 for index, value in enumerate(probability)
            )
        )
    ece = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        selected = [
            offset
            for offset, value in enumerate(confidence)
            if value >= low and (value < high if index < bins - 1 else value <= high)
        ]
        if selected:
            ece += (
                len(selected)
                / len(rows)
                * abs(
                    fmean(correct[offset] for offset in selected)
                    - fmean(confidence[offset] for offset in selected)
                )
            )
    return {
        "count": len(rows),
        "accuracy": fmean(correct),
        "nll": fmean(nll),
        "brier": fmean(brier),
        "ece_15_bins": ece,
    }


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def paired_scene_bootstrap(
    reference: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
    *,
    replicates: int = 10_000,
    seed: int = 17,
) -> dict[str, Any]:
    _validate_pair(reference, candidate)
    scene_to_ids: dict[int, list[str]] = {}
    for sample_id, row in reference.items():
        scene_to_ids.setdefault(int(row["scene_index"]), []).append(sample_id)
    scenes = sorted(scene_to_ids)
    if replicates < 1 or len(scenes) < 2:
        raise ValueError("need positive replicates and at least two independent scenes")
    rng = random.Random(seed)
    metric_names = ("accuracy", "nll", "brier", "ece_15_bins")
    distributions: dict[str, list[float]] = {name: [] for name in metric_names}
    by_type: dict[str, dict[str, list[float]]] = {}
    task_types = sorted({row["question_type"] for row in reference.values()})
    for task in task_types:
        by_type[task] = {name: [] for name in metric_names}
    for _ in range(replicates):
        sampled_scenes = rng.choices(scenes, k=len(scenes))
        sample_ids = [sample_id for scene in sampled_scenes for sample_id in scene_to_ids[scene]]
        ref_rows = [reference[sample_id] for sample_id in sample_ids]
        cand_rows = [candidate[sample_id] for sample_id in sample_ids]
        ref_metrics, cand_metrics = _metrics(ref_rows), _metrics(cand_rows)
        for name in metric_names:
            distributions[name].append(float(cand_metrics[name]) - float(ref_metrics[name]))
        for task in task_types:
            ref_task = [row for row in ref_rows if row["question_type"] == task]
            cand_task = [row for row in cand_rows if row["question_type"] == task]
            ref_task_metrics, cand_task_metrics = _metrics(ref_task), _metrics(cand_task)
            for name in metric_names:
                by_type[task][name].append(
                    float(cand_task_metrics[name]) - float(ref_task_metrics[name])
                )
    intervals = {}
    for name, values in distributions.items():
        intervals[name] = {
            "candidate_minus_reference_mean": fmean(values),
            "percentile_95_ci": [_percentile(values, 0.025), _percentile(values, 0.975)],
        }
    task_intervals = {
        task: {
            name: {
                "candidate_minus_reference_mean": fmean(values),
                "percentile_95_ci": [_percentile(values, 0.025), _percentile(values, 0.975)],
            }
            for name, values in metrics.items()
        }
        for task, metrics in by_type.items()
    }
    return {
        "method": (
            "paired percentile bootstrap resampling independent scene IDs with replacement; "
            "all questions for a sampled scene move together; Python random.Random RNG"
        ),
        "seed": seed,
        "replicates": replicates,
        "independent_scenes": len(scenes),
        "sample_count": len(reference),
        "difference_direction": "candidate minus reference; lower NLL/Brier/ECE is better",
        "overall": intervals,
        "by_question_type": task_intervals,
    }


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = _args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {output}")
    reference = _read_predictions(args.reference.resolve(strict=True))
    candidate = _read_predictions(args.candidate.resolve(strict=True))
    _validate_pair(reference, candidate)
    report = {
        "status": "complete_paired_scene_bootstrap",
        "reference_path": str(args.reference.resolve()),
        "candidate_path": str(args.candidate.resolve()),
        "reference_metrics": _metrics(list(reference.values())),
        "candidate_metrics": _metrics(list(candidate.values())),
        "bootstrap": paired_scene_bootstrap(
            reference, candidate, replicates=args.replicates, seed=args.seed
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
