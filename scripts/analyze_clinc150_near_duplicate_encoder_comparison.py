"""Compare two frozen CLINC150 readouts with a paired group bootstrap."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--minilm-run-dir", type=Path, required=True)
    parser.add_argument("--bert-run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261128)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite comparison: {args.output}")
    rows = read_jsonl(args.data_dir / "validation.jsonl")
    minilm_predictions = read_jsonl(args.minilm_run_dir / "validation-predictions.jsonl")
    bert_predictions = read_jsonl(args.bert_run_dir / "validation-predictions.jsonl")
    if not len(rows) == len(minilm_predictions) == len(bert_predictions):
        raise ValueError("Prediction counts do not match validation rows")
    for row, minilm, bert in zip(rows, minilm_predictions, bert_predictions, strict=True):
        if row["sample_id"] != minilm["sample_id"] or row["sample_id"] != bert["sample_id"]:
            raise ValueError("Encoder predictions are not paired by validation sample ID")
        if int(row["label"]) != int(minilm["label"]) or int(row["label"]) != int(bert["label"]):
            raise ValueError("Encoder prediction targets do not match validation")

    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    minilm_probs = np.asarray(
        [row["probabilities"] for row in minilm_predictions], dtype=np.float64
    )
    bert_probs = np.asarray([row["probabilities"] for row in bert_predictions], dtype=np.float64)
    minilm_correct = minilm_probs.argmax(axis=1) == labels
    bert_correct = bert_probs.argmax(axis=1) == labels
    minilm_nll = -np.log(np.clip(minilm_probs[np.arange(len(labels)), labels], 1e-15, 1.0))
    bert_nll = -np.log(np.clip(bert_probs[np.arange(len(labels)), labels], 1e-15, 1.0))
    one_hot = np.eye(minilm_probs.shape[1], dtype=np.float64)[labels]
    minilm_brier = np.square(minilm_probs - one_hot).sum(axis=1)
    bert_brier = np.square(bert_probs - one_hot).sum(axis=1)
    observed = {
        "accuracy": {
            "minilm_l6": float(minilm_correct.mean()),
            "bert_mini": float(bert_correct.mean()),
            "delta_minilm_minus_bert": float(minilm_correct.mean() - bert_correct.mean()),
        },
        "nll": {
            "minilm_l6": float(minilm_nll.mean()),
            "bert_mini": float(bert_nll.mean()),
            "delta_minilm_minus_bert": float(minilm_nll.mean() - bert_nll.mean()),
        },
        "brier": {
            "minilm_l6": float(minilm_brier.mean()),
            "bert_mini": float(bert_brier.mean()),
            "delta_minilm_minus_bert": float(minilm_brier.mean() - bert_brier.mean()),
        },
    }

    group_to_rows: dict[str, list[int]] = defaultdict(list)
    group_labels: dict[str, set[int]] = defaultdict(set)
    class_names = {int(row["label"]): str(row["label_name"]) for row in rows}
    for index, row in enumerate(rows):
        group_id = str(row["near_duplicate_group_id"])
        group_to_rows[group_id].append(index)
        group_labels[group_id].add(int(row["label"]))
    groups = sorted(group_to_rows)
    group_index = {group: index for index, group in enumerate(groups)}
    row_group_index = np.asarray(
        [group_index[str(row["near_duplicate_group_id"])] for row in rows], dtype=np.int64
    )
    rng = np.random.default_rng(args.seed)
    deltas = {metric: np.empty(args.repetitions, dtype=np.float64) for metric in observed}
    better_fraction = {metric: 0 for metric in observed}
    group_count = len(groups)
    for repetition in range(args.repetitions):
        drawn_groups = rng.integers(0, group_count, size=group_count)
        group_weights = np.bincount(drawn_groups, minlength=group_count)
        row_weights = group_weights[row_group_index].astype(np.float64)
        denominator = row_weights.sum()
        for metric, minilm_values, bert_values, smaller_is_better in (
            ("accuracy", minilm_correct, bert_correct, False),
            ("nll", minilm_nll, bert_nll, True),
            ("brier", minilm_brier, bert_brier, True),
        ):
            delta = float(
                np.sum(row_weights * minilm_values)
                / denominator
                - np.sum(row_weights * bert_values)
                / denominator
            )
            deltas[metric][repetition] = delta
            better_fraction[metric] += delta < 0 if smaller_is_better else delta > 0

    confidence = {
        metric: {
            "percentile_95_ci_delta_minilm_minus_bert": [
                float(value) for value in np.quantile(values, [0.025, 0.975])
            ],
            "bootstrap_fraction_minilm_better": better_fraction[metric] / args.repetitions,
        }
        for metric, values in deltas.items()
    }
    per_class: dict[str, dict[str, Any]] = {}
    for label, name in sorted(class_names.items()):
        mask = labels == label
        per_class[name] = {
            "validation_rows": int(mask.sum()),
            "minilm_l6_accuracy": float(minilm_correct[mask].mean()),
            "bert_mini_accuracy": float(bert_correct[mask].mean()),
        }
    worst = {
        model: [
            {"label_name": name, **metrics}
            for name, metrics in sorted(
                per_class.items(), key=lambda item: item[1][f"{model}_accuracy"]
            )[:10]
        ]
        for model in ("minilm_l6", "bert_mini")
    }
    result = {
        "schema": "tiny-omni-decision/clinc150-near-duplicate-encoder-comparison-v1",
        "validation_rows": len(rows),
        "validation_near_duplicate_groups": group_count,
        "cross_label_groups_in_validation": sum(
            len(labels_for_group) > 1 for labels_for_group in group_labels.values()
        ),
        "rows_in_cross_label_groups": sum(
            len(group_to_rows[group])
            for group, labels_for_group in group_labels.items()
            if len(labels_for_group) > 1
        ),
        "observed_metrics": observed,
        "paired_group_bootstrap": {
            "resampling_unit": "near_duplicate_group; all rows in a group retained together",
            "repetitions": args.repetitions,
            "seed": args.seed,
            "confidence_intervals": confidence,
            "caveat": (
                "development-only descriptive intervals; both model checkpoints and early-stop "
                "epochs were selected using this validation split"
            ),
        },
        "worst_ten_intents_by_accuracy": worst,
        "per_class_metrics": per_class,
        "official_test_loaded": False,
        "sealed_audit_loaded": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = {key: value for key, value in result.items() if key != "per_class_metrics"}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
