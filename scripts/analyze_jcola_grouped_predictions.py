"""Analyze saved JCoLA predictions without loading models or source rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_predictions(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [str(row["uid"]) for row in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError(f"{path} must contain unique, nonempty prediction rows")
    for row in rows:
        if int(row["target"]) not in (0, 1):
            raise ValueError("prediction has a non-binary target")
        if not 0 <= float(row["p_unacceptable"]) <= 1:
            raise ValueError("invalid unacceptable probability")
        if not 0 <= float(row["p_acceptable"]) <= 1:
            raise ValueError("invalid acceptable probability")
        if abs(float(row["p_unacceptable"]) + float(row["p_acceptable"]) - 1) > 1e-5:
            raise ValueError("option probabilities do not sum to one")
    return rows


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("metrics require at least one prediction")
    tp = tn = fp = fn = 0
    correct = 0
    nll = brier = confidence_sum = ece = 0.0
    bins: list[list[tuple[float, float]]] = [[] for _ in range(15)]
    label_counts = {0: 0, 1: 0}
    for row in rows:
        target = int(row["target"])
        p = [float(row["p_unacceptable"]), float(row["p_acceptable"])]
        prediction = int(row["prediction"])
        if prediction != int(p[1] > p[0]):
            raise ValueError("stored prediction disagrees with option probabilities")
        label_counts[target] += 1
        correct += prediction == target
        if prediction == 1 and target == 1:
            tp += 1
        elif prediction == 0 and target == 0:
            tn += 1
        elif prediction == 1:
            fp += 1
        else:
            fn += 1
        nll -= math.log(max(p[target], 1e-12))
        brier += sum((p[index] - float(index == target)) ** 2 for index in (0, 1))
        confidence = max(p)
        confidence_sum += confidence
        bin_index = min(int(confidence * 15), 14)
        bins[bin_index].append((confidence, float(prediction == target)))
    count = len(rows)
    for values in bins:
        if values:
            ece += len(values) / count * abs(
                sum(conf for conf, _ in values) / len(values)
                - sum(acc for _, acc in values) / len(values)
            )
    recalls = {
        label: sum(int(row["prediction"]) == label for row in rows if int(row["target"]) == label)
        / label_counts[label]
        if label_counts[label]
        else 0.0
        for label in (0, 1)
    }
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "count": count,
        "label_counts": {str(key): value for key, value in label_counts.items()},
        "accuracy": correct / count,
        "macro_accuracy": sum(recalls.values()) / 2,
        "unacceptable_recall": recalls[0],
        "acceptable_recall": recalls[1],
        "nll": nll / count,
        "brier": brier / count,
        "ece_15": ece,
        "mcc": (tp * tn - fp * fn) / denominator if denominator else 0.0,
        "mean_confidence": confidence_sum / count,
    }


def _percentile(values: list[float], fraction: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] * (upper - position) + values[upper] * (position - lower)


def paired_source_bootstrap(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    *,
    repeats: int = 5000,
    seed: int = 20261123,
) -> dict[str, Any]:
    left_by_id = {str(row["uid"]): row for row in left}
    right_by_id = {str(row["uid"]): row for row in right}
    if left_by_id.keys() != right_by_id.keys():
        raise ValueError("paired prediction sets have different UIDs")
    for uid in left_by_id:
        if left_by_id[uid]["source"] != right_by_id[uid]["source"]:
            raise ValueError("paired predictions disagree on source group")
        if left_by_id[uid]["target"] != right_by_id[uid]["target"]:
            raise ValueError("paired predictions disagree on target")
    by_source: dict[str, list[str]] = defaultdict(list)
    for uid, row in left_by_id.items():
        by_source[str(row["source"])].append(uid)
    sources = sorted(by_source)
    if len(sources) < 2:
        raise ValueError("source bootstrap requires multiple independent groups")
    rng = random.Random(seed)
    deltas: dict[str, list[float]] = {key: [] for key in ("accuracy", "nll", "brier")}
    for _ in range(repeats):
        sampled = [rng.choice(sources) for _ in sources]
        left_rows = [left_by_id[uid] for source in sampled for uid in by_source[source]]
        right_rows = [right_by_id[uid] for source in sampled for uid in by_source[source]]
        left_metrics, right_metrics = compute_metrics(left_rows), compute_metrics(right_rows)
        for key in deltas:
            deltas[key].append(float(right_metrics[key]) - float(left_metrics[key]))
    return {
        "unit": "source group",
        "source_groups": len(sources),
        "repeats": repeats,
        "seed": seed,
        "interval": "percentile 95%",
        "direction": "right model minus left model",
        "metrics": {
            key: {
                "mean_delta": sum(values) / len(values),
                "ci95": [_percentile(values, 0.025), _percentile(values, 0.975)],
                "fraction_delta_positive": sum(value > 0 for value in values) / len(values),
            }
            for key, values in deltas.items()
        },
    }


def analyze(run_dir: Path, *, repeats: int = 5000, seed: int = 20261123) -> dict[str, Any]:
    model_names = ("tohoku-bert-base-v3", "line-distilbert-ja-4layer")
    inputs = {
        name: run_dir / name / "out-of-domain-development-predictions.jsonl"
        for name in model_names
    }
    predictions = {name: read_predictions(path) for name, path in inputs.items()}
    per_source = {}
    for name, rows in predictions.items():
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row["source"])].append(row)
        per_source[name] = {
            source: compute_metrics(group_rows)
            for source, group_rows in sorted(grouped.items())
        }
    return {
        "analysis_kind": "saved-prediction-only; no model, tokenizer, or raw corpus loaded",
        "models": {
            name: {
                "overall": compute_metrics(rows),
                "per_source": per_source[name],
                "predictions_sha256": sha256_file(inputs[name]),
            }
            for name, rows in predictions.items()
        },
        "paired_source_cluster_bootstrap": paired_source_bootstrap(
            predictions[model_names[0]], predictions[model_names[1]], repeats=repeats, seed=seed
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--repeats", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261123)
    args = parser.parse_args()
    report = analyze(args.run_dir, repeats=args.repeats, seed=args.seed)
    output = args.output or args.run_dir / "saved-prediction-analysis.json"
    report["analysis_script_sha256"] = sha256_file(Path(__file__))
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(output),
        "models": {name: value["overall"] for name, value in report["models"].items()},
        "paired_source_cluster_bootstrap": report["paired_source_cluster_bootstrap"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
