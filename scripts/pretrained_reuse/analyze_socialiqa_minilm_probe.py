"""Verify saved SocialIQA predictions and compute paired context bootstrap."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.pretrained_reuse.socialiqa_probe_data import read_split


def read_predictions(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def verify_predictions(rows: list[dict[str, Any]], items: list[dict[str, Any]]) -> np.ndarray:
    if len(rows) != len(items):
        raise ValueError(f"prediction count {len(rows)} != item count {len(items)}")
    probabilities = []
    for index, (row, item) in enumerate(zip(rows, items, strict=True)):
        expected_group_hash = hashlib.sha256(item["group"].encode()).hexdigest()
        if row["id"] != item["id"] or row["target"] != item["target"]:
            raise ValueError(f"prediction id/target mismatch at row {index}")
        if row["context_group_sha256"] != expected_group_hash:
            raise ValueError(f"prediction context group mismatch at row {index}")
        p = np.asarray(row["probabilities"], dtype=np.float64)
        if p.shape != (3,) or not np.isfinite(p).all() or np.any(p < 0):
            raise ValueError(f"invalid probability vector at row {index}")
        if abs(float(p.sum()) - 1.0) > 1e-5:
            raise ValueError(f"probabilities do not sum to one at row {index}")
        if int(np.argmax(p)) != int(row["prediction"]):
            raise ValueError(f"prediction does not match probability argmax at row {index}")
        probabilities.append(p)
    return np.stack(probabilities)


def score(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predictions == labels
    nll = -np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1.0)).mean()
    one_hot = np.eye(3)[labels]
    brier = ((probabilities - one_hot) ** 2).sum(axis=1).mean()
    ece = 0.0
    for index in range(10):
        lower, upper = index / 10, (index + 1) / 10
        mask = (confidence >= lower) & (
            (confidence < upper) if index < 9 else (confidence <= upper)
        )
        if mask.any():
            ece += mask.mean() * abs(confidence[mask].mean() - correct[mask].mean())
    return {
        "accuracy": float(correct.mean()),
        "nll": float(nll),
        "brier": float(brier),
        "ece_10": float(ece),
    }


def grouped_bootstrap(
    items: list[dict[str, Any]],
    first: np.ndarray,
    second: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    by_group: dict[str, list[int]] = {}
    for index, item in enumerate(items):
        by_group.setdefault(item["group"], []).append(index)
    group_names = sorted(by_group)
    group_correct = np.asarray(
        [
            [
                sum(int(first[i].argmax() == items[i]["target"]) for i in by_group[group]),
                sum(int(second[i].argmax() == items[i]["target"]) for i in by_group[group]),
                len(by_group[group]),
            ]
            for group in group_names
        ],
        dtype=np.int64,
    )
    rng = random.Random(seed)
    deltas = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sampled = [rng.randrange(len(group_names)) for _ in group_names]
        total_first = group_correct[sampled, 0].sum()
        total_second = group_correct[sampled, 1].sum()
        total_n = group_correct[sampled, 2].sum()
        deltas[replicate] = (total_first - total_second) / total_n
    return {
        "unit": "normalized context group",
        "groups": len(group_names),
        "replicates": replicates,
        "seed": seed,
        "l3_minus_l6_accuracy_delta": float(
            (group_correct[:, 0].sum() - group_correct[:, 1].sum()) / group_correct[:, 2].sum()
        ),
        "percentile_95_interval": [
            float(np.quantile(deltas, 0.025)),
            float(np.quantile(deltas, 0.975)),
        ],
    }


def analyze(data_dir: Path, result_dir: Path, replicates: int, seed: int) -> dict[str, Any]:
    dev_items = read_split(data_dir, "dev")
    labels = np.asarray([item["target"] for item in dev_items], dtype=np.int64)
    l3_rows = read_predictions(result_dir / "minilm_l3" / "publisher-dev-predictions.jsonl")
    l6_rows = read_predictions(result_dir / "minilm_l6" / "publisher-dev-predictions.jsonl")
    l3 = verify_predictions(l3_rows, dev_items)
    l6 = verify_predictions(l6_rows, dev_items)
    train_items = read_split(data_dir, "train")
    train_counts = np.bincount([item["target"] for item in train_items], minlength=3)
    majority = int(train_counts.argmax())
    majority_probs = np.tile(train_counts / train_counts.sum(), (len(labels), 1))
    return {
        "publisher_dev_rows": len(dev_items),
        "publisher_dev_context_groups": len({item["group"] for item in dev_items}),
        "prediction_integrity": (
            "PASS: ordered IDs, labels, context hashes, argmax, finite probabilities, sums"
        ),
        "label_counts": {str(i + 1): int((labels == i).sum()) for i in range(3)},
        "uniform_chance": {"accuracy": 1 / 3, "nll": math.log(3), "brier": 2 / 3, "ece_10": 0.0},
        "train_majority_class_1_based": majority + 1,
        "train_majority_baseline": score(majority_probs, labels),
        "minilm_l3": score(l3, labels),
        "minilm_l6": score(l6, labels),
        "paired_context_bootstrap": grouped_bootstrap(
            dev_items, l3, l6, replicates=replicates, seed=seed
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    report = analyze(args.data_dir, args.result_dir, args.replicates, args.seed)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
