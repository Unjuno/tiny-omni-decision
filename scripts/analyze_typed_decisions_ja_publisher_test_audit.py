"""Paired state-cluster bootstrap on the frozen publisher-test predictions."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _load(path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    result = {}
    for row in rows:
        key = (row["sample_id"], row["workflow"], row["question_key"])
        if key in result:
            raise ValueError(f"duplicate prediction key: {key}")
        result[key] = row
    return result


def _example_metrics(row: dict[str, Any]) -> tuple[float, float, float, int, float, float]:
    target = row["gold_probabilities"]
    probs = row["probabilities"]
    if set(target) != set(probs) or abs(sum(target.values()) - 1.0) > 0.02:
        raise ValueError("prediction options do not match valid target distribution")
    prediction = max(probs, key=probs.get)
    nll = -sum(float(target[key]) * np.log(max(float(probs[key]), 1e-12)) for key in target)
    brier = sum((float(probs[key]) - float(target[key])) ** 2 for key in target)
    confidence = float(probs[prediction])
    expected = float(target[prediction])
    bin_index = min(int(confidence * 15), 14)
    return float(prediction == row["target"]), nll, brier, bin_index, confidence, expected


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bert-predictions", type=Path, required=True)
    parser.add_argument("--compact-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20_261_123)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    bert = _load(args.bert_predictions)
    compact = _load(args.compact_predictions)
    if bert.keys() != compact.keys():
        raise ValueError("candidate predictions are not aligned by state/question")

    by_workflow_state_cell: dict[tuple[str, str, str], dict[str, list[Any]]] = defaultdict(
        lambda: {"bert": [], "compact": []}
    )
    workflow_states: dict[str, set[str]] = defaultdict(set)
    for key in sorted(bert):
        state_id, workflow, question_key = key
        left, right = bert[key], compact[key]
        for field in ("target", "gold_probabilities", "question_type"):
            if left[field] != right[field]:
                raise ValueError(f"candidates disagree on {field} for {key}")
        workflow_states[workflow].add(state_id)
        cell = (workflow, left["question_type"], state_id)
        by_workflow_state_cell[cell]["bert"].append(_example_metrics(left))
        by_workflow_state_cell[cell]["compact"].append(_example_metrics(right))

    workflows = sorted(workflow_states)
    if len(workflows) != 4 or any(len(workflow_states[w]) != 100 for w in workflows):
        raise ValueError("expected four workflows with 100 independent test states each")
    cells = sorted(
        {(workflow, question_type) for workflow, question_type, _ in by_workflow_state_cell}
    )
    if len(cells) != 12:
        raise ValueError("expected all 12 workflow/question-type cells")

    # Per state and cell store count, sums for three proper metrics, and ECE
    # bin sufficient statistics. Bootstrap resamples whole states within each
    # workflow, so all five questions for each selected state stay together.
    stats: dict[str, dict[tuple[str, str], dict[str, np.ndarray]]] = {
        name: {} for name in ("bert", "compact")
    }
    for workflow, question_type in cells:
        state_ids = sorted(workflow_states[workflow])
        for candidate in stats:
            sums = np.zeros((len(state_ids), 4), dtype=np.float64)
            bins = np.zeros((len(state_ids), 15, 3), dtype=np.float64)
            for state_index, state_id in enumerate(state_ids):
                rows = by_workflow_state_cell[(workflow, question_type, state_id)][candidate]
                for accuracy, nll, brier, bin_index, confidence, expected in rows:
                    sums[state_index] += (1.0, accuracy, nll, brier)
                    bins[state_index, bin_index] += (1.0, confidence, expected)
            stats[candidate][(workflow, question_type)] = {
                "sums": sums,
                "bins": bins,
            }

    rng = np.random.Generator(np.random.PCG64(args.seed))
    deltas = {
        key: np.empty(args.replicates, dtype=np.float64)
        for key in ("accuracy", "nll", "brier", "ece")
    }
    for replicate in range(args.replicates):
        selected_counts: dict[str, np.ndarray] = {}
        for workflow in workflows:
            n = len(workflow_states[workflow])
            selected = rng.integers(0, n, size=n)
            selected_counts[workflow] = np.bincount(selected, minlength=n)
        metrics = {candidate: {key: [] for key in deltas} for candidate in stats}
        for workflow, question_type in cells:
            state_counts = selected_counts[workflow]
            for candidate in stats:
                sums = stats[candidate][(workflow, question_type)]["sums"]
                weighted = (sums * state_counts[:, None]).sum(axis=0)
                count = weighted[0]
                metrics[candidate]["accuracy"].append(weighted[1] / count)
                metrics[candidate]["nll"].append(weighted[2] / count)
                metrics[candidate]["brier"].append(weighted[3] / count)
                bins = (
                    stats[candidate][(workflow, question_type)]["bins"]
                    * state_counts[:, None, None]
                ).sum(axis=0)
                occupied = bins[:, 0] > 0
                ece = np.sum(
                    bins[occupied, 0]
                    / count
                    * np.abs(
                        bins[occupied, 1] / bins[occupied, 0]
                        - bins[occupied, 2] / bins[occupied, 0]
                    )
                )
                metrics[candidate]["ece"].append(ece)
        for key in deltas:
            left = float(np.mean(metrics["bert"][key]))
            right = float(np.mean(metrics["compact"][key]))
            deltas[key][replicate] = left - right

    result = {
        "status": "complete_posthoc_paired_test_prediction_bootstrap",
        "method": (
            "PCG64 percentile bootstrap; resample 100 state IDs with replacement "
            "independently within each of 4 workflows; retain all questions per "
            "state; compute macro across 12 workflow/question-type cells; BERT minus compact"
        ),
        "seed": args.seed,
        "replicates": args.replicates,
        "candidate_prediction_files": {
            "bert": str(args.bert_predictions),
            "compact": str(args.compact_predictions),
        },
        "differences": {
            key: {
                "mean": float(values.mean()),
                "percentile_95_interval": [
                    float(np.quantile(values, 0.025)),
                    float(np.quantile(values, 0.975)),
                ],
                "fraction_bert_better": float(np.mean(values > 0))
                if key == "accuracy"
                else float(np.mean(values < 0)),
            }
            for key, values in deltas.items()
        },
        "interpretation_limit": (
            "descriptive uncertainty on one already-opened publisher test; not a "
            "blind benchmark and not for further checkpoint or hyperparameter selection"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
