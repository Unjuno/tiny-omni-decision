"""Measure train-only label predictability from recorded Physion++ factors."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

DEFAULT_TRAIN_SHA256 = "0de1bbbb1cc54000936c1771ea54e375c214fedda36734536f31a44c4f64932b"
FOLD_COUNT = 5
ALPHA = 1.0
_FACTOR = re.compile(r"([A-Za-z_]+)=([^/#-]+)")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def factor_features(row: dict[str, Any]) -> tuple[str, ...]:
    source_id = row.get("source_record_id")
    if not isinstance(source_id, str):
        raise ValueError("Physion++ row lacks source_record_id")
    parts = source_id.split("/")
    if len(parts) < 3 or parts[0] != "readout_data_v1":
        raise ValueError(f"unexpected Physion++ source record ID: {source_id}")
    features = {f"scenario={parts[1]}"}
    features.update(f"{key}={value}" for key, value in _FACTOR.findall(source_id))
    return tuple(sorted(features))


def group_fold(group_id: str, fold_count: int = FOLD_COUNT) -> int:
    if not group_id:
        raise ValueError("source_asset_group_id must be non-empty")
    if fold_count < 2:
        raise ValueError("fold_count must be at least 2")
    digest = hashlib.sha256(group_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % fold_count


def _binary_metrics(probabilities: list[float], targets: list[int]) -> dict[str, Any]:
    if len(probabilities) != len(targets) or not targets:
        raise ValueError("metric inputs must be non-empty and aligned")
    predictions = [int(probability >= 0.5) for probability in probabilities]
    class_counts = Counter(targets)
    if set(class_counts) != {0, 1}:
        raise ValueError("each metric split must contain both labels")
    accuracy = sum(
        pred == target for pred, target in zip(predictions, targets, strict=True)
    ) / len(targets)
    recall_no = sum(
        pred == target == 0 for pred, target in zip(predictions, targets, strict=True)
    ) / class_counts[0]
    recall_yes = sum(
        pred == target == 1 for pred, target in zip(predictions, targets, strict=True)
    ) / class_counts[1]
    nll = sum(
        -math.log(max(1e-12, probability if target else 1.0 - probability))
        for probability, target in zip(probabilities, targets, strict=True)
    ) / len(targets)
    brier = sum(
        (probability - target) ** 2
        for probability, target in zip(probabilities, targets, strict=True)
    ) / len(targets)
    ece = 0.0
    for bin_index in range(10):
        indices = [
            index
            for index, probability in enumerate(probabilities)
            if bin_index / 10 <= probability < (bin_index + 1) / 10
            or (bin_index == 9 and probability == 1.0)
        ]
        if indices:
            mean_probability = sum(probabilities[index] for index in indices) / len(indices)
            observed_yes = sum(targets[index] for index in indices) / len(indices)
            ece += len(indices) / len(targets) * abs(mean_probability - observed_yes)
    return {
        "n": len(targets),
        "class_counts": {"No": class_counts[0], "Yes": class_counts[1]},
        "accuracy": accuracy,
        "balanced_accuracy": (recall_no + recall_yes) / 2,
        "nll": nll,
        "brier": brier,
        "probability_ece10": ece,
        "ece_definition": "10-bin ECE over P(Yes), compared with empirical Yes frequency",
    }


def _fit_fold(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    train_labels = [int(bool(row["source_target"])) for row in train_rows]
    label_counts = Counter(train_labels)
    if set(label_counts) != {0, 1}:
        raise ValueError("training fold must contain both labels")
    prior_yes = (label_counts[1] + ALPHA) / (len(train_rows) + 2 * ALPHA)
    value_counts: dict[tuple[str, str], Counter[int]] = defaultdict(Counter)
    category_counts: dict[str, set[str]] = defaultdict(set)
    for row in train_rows:
        label = int(bool(row["source_target"]))
        for feature in factor_features(row):
            field, value = feature.split("=", 1)
            value_counts[(field, value)][label] += 1
            category_counts[field].add(value)

    predictions: list[dict[str, Any]] = []
    factor_probabilities: list[float] = []
    prior_probabilities: list[float] = []
    targets: list[int] = []
    for row in test_rows:
        target = int(bool(row["source_target"]))
        targets.append(target)
        prior_probabilities.append(prior_yes)
        log_scores: list[float] = []
        for label in (0, 1):
            score = math.log((label_counts[label] + ALPHA) / (len(train_rows) + 2 * ALPHA))
            for feature in factor_features(row):
                field, value = feature.split("=", 1)
                categories = category_counts[field]
                if value in categories:
                    score += math.log(
                        (value_counts[(field, value)][label] + ALPHA)
                        / (label_counts[label] + ALPHA * len(categories))
                    )
            log_scores.append(score)
        maximum = max(log_scores)
        exp_no = math.exp(log_scores[0] - maximum)
        exp_yes = math.exp(log_scores[1] - maximum)
        probability_yes = exp_yes / (exp_no + exp_yes)
        factor_probabilities.append(probability_yes)
        predictions.append(
            {
                "sample_id": row["id"],
                "source_asset_group_id": row["source_asset_group_id"],
                "fold": group_fold(row["source_asset_group_id"]),
                "target": "Yes" if target else "No",
                "prior_probability_yes": prior_yes,
                "factor_probability_yes": probability_yes,
            }
        )
    return (
        predictions,
        _binary_metrics(prior_probabilities, targets),
        _binary_metrics(factor_probabilities, targets),
    )


def run_factor_audit(
    train_jsonl: Path,
    output_dir: Path,
    expected_sha256: str = DEFAULT_TRAIN_SHA256,
) -> dict[str, Any]:
    raw = train_jsonl.read_bytes()
    manifest_sha = sha256_bytes(raw)
    if manifest_sha != expected_sha256.lower():
        raise ValueError(f"train manifest SHA-256 mismatch: {manifest_sha}")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if not rows or any(row.get("split") != "train" for row in rows):
        raise ValueError("factor-only audit accepts only a nonempty train split")
    if any(row.get("modality") != "video" for row in rows):
        raise ValueError("factor-only audit accepts only video rows")
    if len({row.get("id") for row in rows}) != len(rows):
        raise ValueError("train sample IDs must be unique")
    for row in rows:
        if (
            row.get("target") not in {"Yes", "No"}
            or not isinstance(row.get("source_target"), bool)
            or (row["target"] == "Yes") != row["source_target"]
        ):
            raise ValueError(f"invalid or inconsistent train label: {row.get('id')}")
        if not row.get("source_asset_group_id"):
            raise ValueError(f"missing source asset group: {row.get('id')}")
        factor_features(row)

    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite audit output: {output_dir}")
    output_dir.mkdir(parents=True)
    fold_ids = {group_fold(row["source_asset_group_id"]) for row in rows}
    if fold_ids != set(range(FOLD_COUNT)):
        raise ValueError(f"train groups do not populate all fixed folds: {sorted(fold_ids)}")

    predictions: list[dict[str, Any]] = []
    folds: list[dict[str, Any]] = []
    groups_by_fold: dict[int, set[str]] = defaultdict(set)
    for row in rows:
        groups_by_fold[group_fold(row["source_asset_group_id"])].add(
            row["source_asset_group_id"]
        )
    for fold in range(FOLD_COUNT):
        test_rows = [row for row in rows if group_fold(row["source_asset_group_id"]) == fold]
        train_rows_fold = [row for row in rows if group_fold(row["source_asset_group_id"]) != fold]
        train_groups = {row["source_asset_group_id"] for row in train_rows_fold}
        test_groups = groups_by_fold[fold]
        if train_groups & test_groups:
            raise ValueError(f"source asset group crosses fixed folds: {fold}")
        fold_predictions, prior_metrics, factor_metrics = _fit_fold(
            train_rows_fold, test_rows
        )
        predictions.extend(fold_predictions)
        folds.append(
            {
                "fold": fold,
                "train_examples": len(train_rows_fold),
                "test_examples": len(test_rows),
                "train_groups": len(train_groups),
                "test_groups": len(test_groups),
                "prior_baseline": prior_metrics,
                "factor_only": factor_metrics,
            }
        )

    prior_all = [item["prior_probability_yes"] for item in predictions]
    factor_all = [item["factor_probability_yes"] for item in predictions]
    targets_all = [int(item["target"] == "Yes") for item in predictions]
    prior_metrics = _binary_metrics(prior_all, targets_all)
    factor_metrics = _binary_metrics(factor_all, targets_all)

    predictions_path = output_dir / "oof-predictions.jsonl"
    with predictions_path.open("x", encoding="utf-8", newline="\n") as sink:
        for item in predictions:
            sink.write(json.dumps(item, sort_keys=True) + "\n")
    report = {
        "status": "pass",
        "purpose": (
            "train-only recorded-factor shortcut diagnostic; "
            "not a video model-quality score"
        ),
        "train_manifest_path": str(train_jsonl),
        "train_manifest_sha256": manifest_sha,
        "audit_script_sha256": sha256_file(Path(__file__)),
        "train_records": len(rows),
        "unique_source_asset_groups": len({row["source_asset_group_id"] for row in rows}),
        "fold_policy": "five folds assigned by int(SHA256(source_asset_group_id)[:4])/modulo 5",
        "factor_model": (
            "categorical naive Bayes; add-one smoothing; scenario and recorded "
            "filename factors; fixed alpha=1"
        ),
        "folds": folds,
        "oof_prior_baseline": prior_metrics,
        "oof_factor_only": factor_metrics,
        "oof_predictions_sha256": sha256_file(predictions_path),
        "validation_or_sealed_data_opened": False,
        "gpu_used": False,
        "checkpoint_selected": False,
    }
    (output_dir / "audit-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-train-sha256", default=DEFAULT_TRAIN_SHA256)
    args = parser.parse_args()
    print(
        json.dumps(
            run_factor_audit(args.train_jsonl, args.output_dir, args.expected_train_sha256),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
