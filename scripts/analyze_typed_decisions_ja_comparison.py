"""Paired, state-clustered bootstrap for the frozen Typed Decisions comparison."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _read_predictions(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _row_metrics(row: dict[str, Any]) -> dict[str, Any]:
    labels = list(row["gold_probabilities"])
    if set(labels) != set(row["probabilities"]):
        raise ValueError("prediction and target option labels differ")
    target = np.asarray([row["gold_probabilities"][key] for key in labels])
    probabilities = np.asarray([row["probabilities"][key] for key in labels])
    prediction_index = int(probabilities.argmax())
    return {
        "accuracy": float(labels[prediction_index] == row["target"]),
        "nll": float(-(target * np.log(np.maximum(probabilities, 1e-12))).sum()),
        "brier": float(np.square(probabilities - target).sum()),
        "confidence": float(probabilities[prediction_index]),
        "expected_correctness": float(target[prediction_index]),
    }


def _paired_rows(
    first: list[dict[str, Any]], second: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    key_fields = ("sample_id", "workflow", "question_key", "question_type", "target")
    first_by_key = {
        tuple(row[field] for field in key_fields): row for row in first
    }
    second_by_key = {
        tuple(row[field] for field in key_fields): row for row in second
    }
    if len(first_by_key) != len(first) or len(second_by_key) != len(second):
        raise ValueError("duplicate sample/question prediction key")
    if first_by_key.keys() != second_by_key.keys():
        raise ValueError("the runs do not contain identical validation decisions")
    paired = []
    for key, left in first_by_key.items():
        right = second_by_key[key]
        if left["gold_probabilities"] != right["gold_probabilities"]:
            raise ValueError("teacher target distributions differ across runs")
        paired.append(
            {
                "sample_id": left["sample_id"],
                "workflow": left["workflow"],
                "question_type": left["question_type"],
                "first": _row_metrics(left),
                "second": _row_metrics(right),
            }
        )
    return paired


def _cell_metrics(rows: list[dict[str, Any]], side: str) -> dict[str, float]:
    if not rows:
        raise ValueError("bootstrap sample produced an empty workflow/question cell")
    ece = 0.0
    for bin_index in range(15):
        values = [
            row[side]
            for row in rows
            if min(int(row[side]["confidence"] * 15), 14) == bin_index
        ]
        if values:
            ece += len(values) / len(rows) * abs(
                sum(item["confidence"] for item in values) / len(values)
                - sum(item["expected_correctness"] for item in values) / len(values)
            )
    return {
        metric: sum(row[side][metric] for row in rows) / len(rows)
        for metric in ("accuracy", "nll", "brier")
    } | {"ece": ece}


def _macro_metrics(
    rows: list[dict[str, Any]], side: str
) -> dict[str, float]:
    cells: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        cells[(row["workflow"], row["question_type"])].append(row)
    per_cell = [_cell_metrics(values, side) for values in cells.values()]
    return {
        metric: sum(cell[metric] for cell in per_cell) / len(per_cell)
        for metric in ("accuracy", "nll", "brier", "ece")
    }


def compare(
    first: list[dict[str, Any]],
    second: list[dict[str, Any]],
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    paired = _paired_rows(first, second)
    by_workflow_state: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    cell_state_counts: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in paired:
        by_workflow_state[row["workflow"]][row["sample_id"]].append(row)
        cell_state_counts[(row["workflow"], row["question_type"])].add(
            row["sample_id"]
        )

    first_point = _macro_metrics(paired, "first")
    second_point = _macro_metrics(paired, "second")
    point_delta = {key: second_point[key] - first_point[key] for key in first_point}
    rng = np.random.default_rng(seed)
    workflows = sorted(by_workflow_state)
    states_by_workflow = {
        workflow: sorted(by_workflow_state[workflow]) for workflow in workflows
    }
    deltas = {key: [] for key in first_point}
    for _ in range(repetitions):
        sample = []
        for workflow in workflows:
            state_ids = states_by_workflow[workflow]
            selected = rng.choice(state_ids, size=len(state_ids), replace=True)
            for sample_id in selected:
                sample.extend(by_workflow_state[workflow][str(sample_id)])
        first_metrics = _macro_metrics(sample, "first")
        second_metrics = _macro_metrics(sample, "second")
        for metric in deltas:
            deltas[metric].append(second_metrics[metric] - first_metrics[metric])

    intervals = {
        metric: {
            "percentile_95_ci": [
                float(np.quantile(values, 0.025)),
                float(np.quantile(values, 0.975)),
            ],
            "bootstrap_fraction_second_better": float(
                np.mean(np.asarray(values) > 0)
                if metric == "accuracy"
                else np.mean(np.asarray(values) < 0)
            ),
        }
        for metric, values in deltas.items()
    }
    return {
        "paired_decisions": len(paired),
        "paired_state_groups": sum(len(value) for value in by_workflow_state.values()),
        "state_groups_by_workflow": {
            workflow: len(states_by_workflow[workflow]) for workflow in workflows
        },
        "state_groups_by_cell": {
            f"{workflow}/{question_type}": len(state_ids)
            for (workflow, question_type), state_ids in sorted(cell_state_counts.items())
        },
        "first_macro_metrics": first_point,
        "second_macro_metrics": second_point,
        "delta_second_minus_first": point_delta,
        "bootstrap": {
            "method": "paired state-cluster bootstrap, sampled within workflow",
            "repetitions": repetitions,
            "seed": seed,
            "intervals": intervals,
            "caveat": (
                "Both readouts were selected on this validation split; intervals are "
                "descriptive, not blind-test inference."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-predictions", type=Path, required=True)
    parser.add_argument("--second-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261123)
    args = parser.parse_args()
    if args.repetitions < 1000:
        raise ValueError("at least 1000 bootstrap repetitions are required")
    result = compare(
        _read_predictions(args.first_predictions),
        _read_predictions(args.second_predictions),
        repetitions=args.repetitions,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
