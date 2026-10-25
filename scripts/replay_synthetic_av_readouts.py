"""Independently replay metrics from saved AV probabilities on CPU."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from tiny_omni_decision.synthetic_av_evaluation import replay_prediction_metrics


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay(evaluation_dir: Path, output: Path) -> dict[str, Any]:
    evaluation_dir = evaluation_dir.resolve(strict=True)
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite replay output: {output}")
    evaluation_path = evaluation_dir / "evaluation.json"
    predictions_path = evaluation_dir / "predictions.jsonl"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    if evaluation.get("status") != "PASS_SYNTHETIC_AV_READOUT_VALIDATION":
        raise ValueError("evaluation receipt is incomplete")
    if _sha256(predictions_path) != evaluation.get("prediction_sha256"):
        raise ValueError("saved prediction hash differs from evaluation receipt")
    by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    with predictions_path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            key = (row["architecture"], row["checkpoint"], row["control"])
            by_key.setdefault(key, []).append(row)
    replayed = []
    mismatches = []
    metric_corrections = []
    for result in evaluation["results"]:
        for control, expected in result["controls"].items():
            key = (result["architecture"], result["checkpoint"], control)
            records = by_key.get(key, [])
            if len(records) != evaluation["validation_count"]:
                raise ValueError(f"saved predictions are incomplete for {key}")
            metrics = replay_prediction_metrics(records)
            legacy_records = [
                {name: value for name, value in row.items() if name != "prediction"}
                for row in records
            ]
            probability_argmax_metrics = replay_prediction_metrics(legacy_records)
            tie_prediction_disagreements = sum(
                row["prediction"] != int(row["probabilities"][1] > row["probabilities"][0])
                and abs(row["probabilities"][0] - row["probabilities"][1]) <= 1e-7
                for row in records
            )
            cohort_reference_mismatches = []
            for metric, value in metrics.items():
                reference_value = (
                    probability_argmax_metrics[metric]
                    if metric in {"accuracy", "ece"}
                    else value
                )
                if abs(reference_value - expected[metric]) > 5e-6:
                    cohort_reference_mismatches.append(
                        {
                            "key": key,
                            "metric": metric,
                            "expected": expected[metric],
                            "replayed_from_probability_argmax": (
                                probability_argmax_metrics[metric]
                                if metric in {"accuracy", "ece"}
                                else None
                            ),
                            "replayed_from_saved_prediction": (
                                value if metric not in {"accuracy", "ece"} else None
                            ),
                        }
                    )
            if cohort_reference_mismatches:
                mismatches.extend(cohort_reference_mismatches)
            else:
                for metric, value in metrics.items():
                    if abs(value - expected[metric]) <= 5e-6:
                        continue
                    difference = {
                        "key": key,
                        "metric": metric,
                        "expected": expected[metric],
                        "saved_prediction_replay": value,
                    }
                    if metric in {
                        "accuracy",
                        "ece",
                        "pair_exact_accuracy",
                        "group_macro_accuracy",
                    } and tie_prediction_disagreements:
                        metric_corrections.append(difference)
                    else:
                        mismatches.append(difference)
            replayed.append(
                {
                    "key": key,
                    "example_count": len(records),
                    "tie_prediction_disagreements": tie_prediction_disagreements,
                    "probability_argmax_accuracy": probability_argmax_metrics["accuracy"],
                    "probability_argmax_ece": probability_argmax_metrics["ece"],
                    "metrics": metrics,
                }
            )
    report = {
        "status": (
            "FAIL"
            if mismatches
            else "PASS_SYNTHETIC_AV_PREDICTION_REPLAY_WITH_METRIC_CORRECTIONS"
            if metric_corrections
            else "PASS_SYNTHETIC_AV_PREDICTION_REPLAY"
        ),
        "evaluation_sha256": _sha256(evaluation_path),
        "prediction_sha256": _sha256(predictions_path),
        "metric_absolute_tolerance": 5e-6,
        "probability_tie_tolerance": 1e-7,
        "replayed_cohorts": len(replayed),
        "metrics": replayed,
        "metric_corrections": metric_corrections,
        "mismatches": mismatches,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "replay.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(replay(args.evaluation_dir, args.output), indent=2))


if __name__ == "__main__":
    main()
