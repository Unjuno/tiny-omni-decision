"""Correct PhysicalAI full-validation metrics from saved rows without inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scripts.audit_physicalai_saved_prediction_metrics import score, sha256_file
except ModuleNotFoundError as exc:
    if exc.name != "scripts":
        raise
    from audit_physicalai_saved_prediction_metrics import score, sha256_file

ARMS = (
    "proprio_history",
    "video_8frame_mean",
    "video_last_frame",
    "joint_video_proprio",
    "joint_video_missing",
    "joint_proprio_missing",
    "joint_video_episode_misaligned",
    "train_class_prior",
)


def audit(predictions_path: Path, results_path: Path) -> dict[str, Any]:
    predictions_path = predictions_path.resolve(strict=True)
    results_path = results_path.resolve(strict=True)
    rows = json.loads(predictions_path.read_text(encoding="utf-8"))
    results = json.loads(results_path.read_text(encoding="utf-8"))["results"]
    if not rows:
        raise ValueError("saved full-validation predictions are empty")
    targets = np.asarray([row["target_prototype"] for row in rows], dtype=np.int64)
    corrected: dict[str, Any] = {}
    for arm in ARMS:
        probabilities = np.asarray(
            [row[arm]["probabilities"] for row in rows], dtype=np.float64
        )
        if any(
            row[arm]["prediction"] != int(probabilities[index].argmax())
            for index, row in enumerate(rows)
        ):
            raise ValueError(f"saved prediction disagrees with probabilities for {arm}")
        values = score(probabilities, targets)
        previous = results[arm]
        for key in ("accuracy", "nll", "brier"):
            if abs(values[key] - previous[key]) > 5e-6:
                raise ValueError(f"saved {key} differs from historical result for {arm}")
        values["legacy_reported_ece10"] = previous["ece_10"]
        corrected[arm] = values
    code_path = Path(__file__).resolve(strict=True)
    return {
        "schema_version": 1,
        "audit_type": "saved_full_validation_metric_correction",
        "validation_rows": len(rows),
        "arms": corrected,
        "inputs": {
            "audit_code": str(code_path),
            "audit_code_sha256": sha256_file(code_path),
            "predictions": str(predictions_path),
            "predictions_sha256": sha256_file(predictions_path),
            "results": str(results_path),
            "results_sha256": sha256_file(results_path),
        },
        "model_inference": False,
        "feature_extraction": False,
        "optimizer_updates": 0,
        "checkpoint_selection": False,
        "sealed_audit_loaded": False,
        "interpretation": (
            "Corrected ECE includes confidence exactly 1.0 in the final bin. "
            "Accuracy, NLL, and Brier are verified against saved historical metrics."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result = audit(args.predictions, args.results)
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"audited {result['validation_rows']} saved full-validation predictions")


if __name__ == "__main__":
    main()
