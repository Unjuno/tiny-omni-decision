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
    for result in evaluation["results"]:
        for control, expected in result["controls"].items():
            key = (result["architecture"], result["checkpoint"], control)
            records = by_key.get(key, [])
            if len(records) != evaluation["validation_count"]:
                raise ValueError(f"saved predictions are incomplete for {key}")
            metrics = replay_prediction_metrics(records)
            for metric, value in metrics.items():
                if abs(value - expected[metric]) > 1e-6:
                    mismatches.append(
                        {
                            "key": key,
                            "metric": metric,
                            "expected": expected[metric],
                            "actual": value,
                        }
                    )
            replayed.append({"key": key, "example_count": len(records), "metrics": metrics})
    report = {
        "status": "PASS_SYNTHETIC_AV_PREDICTION_REPLAY" if not mismatches else "FAIL",
        "evaluation_sha256": _sha256(evaluation_path),
        "prediction_sha256": _sha256(predictions_path),
        "replayed_cohorts": len(replayed),
        "metrics": replayed,
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
