"""Evaluate Physion++ recorded factors with deterministic scenario holdouts."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from scripts.audit_physionpp_factor_only_baseline import (
        ALPHA,
        DEFAULT_TRAIN_SHA256,
        _binary_metrics,
        factor_features,
        sha256_bytes,
        sha256_file,
    )
except ModuleNotFoundError:
    from audit_physionpp_factor_only_baseline import (
        ALPHA,
        DEFAULT_TRAIN_SHA256,
        _binary_metrics,
        factor_features,
        sha256_bytes,
        sha256_file,
    )


def _scenario(row: dict[str, Any]) -> str:
    parts = row["source_record_id"].split("/")
    if len(parts) < 3 or parts[0] != "readout_data_v1":
        raise ValueError(f"invalid source record ID: {row['source_record_id']}")
    return parts[1]


def _fit_factor_model(
    train_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]]
) -> list[float]:
    labels = [int(bool(row["source_target"])) for row in train_rows]
    label_counts = Counter(labels)
    if set(label_counts) != {0, 1}:
        raise ValueError("training partition must contain both labels")
    category_counts: dict[str, set[str]] = defaultdict(set)
    value_counts: dict[tuple[str, str], Counter[int]] = defaultdict(Counter)
    for row, label in zip(train_rows, labels, strict=True):
        for feature in factor_features(row):
            field, value = feature.split("=", 1)
            category_counts[field].add(value)
            value_counts[(field, value)][label] += 1

    probabilities: list[float] = []
    for row in test_rows:
        log_scores = []
        for label in (0, 1):
            score = math.log(
                (label_counts[label] + ALPHA)
                / (len(train_rows) + 2 * ALPHA)
            )
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
        no_score = math.exp(log_scores[0] - maximum)
        yes_score = math.exp(log_scores[1] - maximum)
        probabilities.append(yes_score / (no_score + yes_score))
    return probabilities


def run_scenario_holdout(
    train_jsonl: Path,
    output_dir: Path,
    *,
    expected_sha256: str = DEFAULT_TRAIN_SHA256,
) -> dict[str, Any]:
    raw = train_jsonl.read_bytes()
    manifest_sha = sha256_bytes(raw)
    if manifest_sha != expected_sha256.lower():
        raise ValueError(f"train manifest SHA-256 mismatch: {manifest_sha}")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line]
    if not rows or any(row.get("split") != "train" for row in rows):
        raise ValueError("scenario-holdout accepts only train rows")
    if any(row.get("modality") != "video" for row in rows):
        raise ValueError("scenario-holdout accepts only video rows")
    if len({row.get("id") for row in rows}) != len(rows):
        raise ValueError("sample IDs must be unique")
    for row in rows:
        if not isinstance(row.get("source_target"), bool):
            raise ValueError(f"invalid source label: {row.get('id')}")
        if (row.get("target") == "Yes") != row["source_target"]:
            raise ValueError(f"inconsistent source label: {row.get('id')}")
        if not row.get("source_asset_group_id"):
            raise ValueError(f"missing source asset group: {row.get('id')}")
        factor_features(row)
        _scenario(row)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {output_dir}")

    scenarios = sorted({_scenario(row) for row in rows})
    by_scenario = {
        scenario: [row for row in rows if _scenario(row) == scenario]
        for scenario in scenarios
    }
    for scenario, test_rows in by_scenario.items():
        if {row["target"] for row in test_rows} != {"No", "Yes"}:
            raise ValueError(f"held-out scenario lacks both labels: {scenario}")

    predictions: list[dict[str, Any]] = []
    fold_reports: list[dict[str, Any]] = []
    for scenario in scenarios:
        test_rows = by_scenario[scenario]
        train_rows = [row for row in rows if _scenario(row) != scenario]
        train_groups = {row["source_asset_group_id"] for row in train_rows}
        test_groups = {row["source_asset_group_id"] for row in test_rows}
        if train_groups & test_groups:
            raise ValueError(f"source asset group crosses scenario split: {scenario}")
        prior_yes = (sum(row["source_target"] for row in train_rows) + 1) / (
            len(train_rows) + 2
        )
        probabilities = _fit_factor_model(train_rows, test_rows)
        targets = [int(row["source_target"]) for row in test_rows]
        fold_reports.append(
            {
                "held_out_scenario": scenario,
                "train_examples": len(train_rows),
                "test_examples": len(test_rows),
                "train_source_asset_groups": len(train_groups),
                "test_source_asset_groups": len(test_groups),
                "prior_baseline": _binary_metrics(
                    [prior_yes] * len(test_rows), targets
                ),
                "factor_only": _binary_metrics(probabilities, targets),
            }
        )
        for row, probability in zip(test_rows, probabilities, strict=True):
            predictions.append(
                {
                    "sample_id": row["id"],
                    "source_asset_group_id": row["source_asset_group_id"],
                    "held_out_scenario": scenario,
                    "target": row["target"],
                    "probability_yes": probability,
                    "prior_probability_yes": prior_yes,
                    "prediction": "Yes" if probability >= 0.5 else "No",
                }
            )

    predictions.sort(key=lambda item: item["sample_id"])
    output_dir.mkdir(parents=True)
    prediction_path = output_dir / "oof-predictions.jsonl"
    prediction_path.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in predictions),
        encoding="utf-8",
    )
    targets_all = [int(item["target"] == "Yes") for item in predictions]
    probabilities_all = [item["probability_yes"] for item in predictions]
    report = {
        "status": "pass",
        "purpose": (
            "train-only recorded-factor generalization diagnostic to unseen "
            "Physion++ scenarios; not video-model quality"
        ),
        "train_manifest_sha256": manifest_sha,
        "train_records": len(rows),
        "scenarios": len(scenarios),
        "scenario_holdout_policy": "leave one entire scenario out; no tuning",
        "factor_model": "categorical naive Bayes, add-one smoothing, fixed alpha=1",
        "folds": fold_reports,
        "pooled_oof_prior": _binary_metrics(
            [item["prior_probability_yes"] for item in predictions], targets_all
        ),
        "pooled_oof_factor_only": _binary_metrics(probabilities_all, targets_all),
        "oof_predictions_sha256": sha256_file(prediction_path),
        "validation_or_sealed_data_opened": False,
        "media_or_model_loaded": False,
        "gpu_used": False,
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
            run_scenario_holdout(
                args.train_jsonl,
                args.output_dir,
                expected_sha256=args.expected_train_sha256,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
