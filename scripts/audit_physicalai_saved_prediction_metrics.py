"""Recompute PhysicalAI pilot metrics from saved predictions, without inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scripts.physicalai_pilot_metrics import expected_calibration_error
except ModuleNotFoundError as exc:
    if exc.name != "scripts":
        raise
    from physicalai_pilot_metrics import expected_calibration_error


PREDICTION_ARMS = (
    "proprio_history",
    "video_8frame_mean",
    "video_last_frame",
    "joint_video_proprio",
    "joint_video_episode_misaligned",
    "joint_video_missing",
    "joint_proprio_missing",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def score(probabilities: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    prediction = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    one_hot = np.eye(probabilities.shape[1], dtype=np.float64)[targets]
    exact_one = int(np.count_nonzero(confidence == 1.0))
    return {
        "count": int(targets.size),
        "accuracy": float(np.mean(prediction == targets)),
        "nll": float(
            -np.log(np.clip(probabilities[np.arange(targets.size), targets], 1e-8, 1.0)).mean()
        ),
        "brier": float(((probabilities - one_hot) ** 2).sum(axis=1).mean()),
        "ece10_corrected": expected_calibration_error(probabilities, targets, bins=10),
        "confidence_exactly_one_count": exact_one,
        "legacy_ece_coverage_count": int(targets.size - exact_one),
        "mean_confidence": float(confidence.mean()),
    }


def audit(
    predictions_path: Path, metrics_path: Path, duplicate_predictions_path: Path
) -> dict[str, Any]:
    predictions_path = predictions_path.resolve(strict=True)
    metrics_path = metrics_path.resolve(strict=True)
    duplicate_predictions_path = duplicate_predictions_path.resolve(strict=True)
    prediction_hash = sha256_file(predictions_path)
    duplicate_prediction_hash = sha256_file(duplicate_predictions_path)
    audit_code_path = Path(__file__).resolve(strict=True)
    audit_code_hash = sha256_file(audit_code_path)
    if prediction_hash != duplicate_prediction_hash:
        raise ValueError("original and reproduction prediction files differ")

    rows = json.loads(predictions_path.read_text(encoding="utf-8"))
    reported = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not rows:
        raise ValueError("saved validation predictions are empty")
    targets = np.asarray([row["target_prototype"] for row in rows], dtype=np.int64)
    arm_metrics: dict[str, Any] = {}
    for arm in PREDICTION_ARMS:
        probabilities = np.asarray([row[arm]["probabilities"] for row in rows], dtype=np.float64)
        if probabilities.ndim != 2 or probabilities.shape[0] != len(rows):
            raise ValueError(f"invalid saved probability array for {arm}")
        prediction_mismatch = any(
            row[arm]["prediction"] != int(probabilities[i].argmax())
            for i, row in enumerate(rows)
        )
        if prediction_mismatch:
            raise ValueError(f"saved prediction disagrees with probability argmax for {arm}")
        values = score(probabilities, targets)
        old = reported[arm]
        for key in ("accuracy", "nll", "brier"):
            if abs(values[key] - old[key]) > 5e-6:
                raise ValueError(f"saved {key} differs from original metric for {arm}")
        values["legacy_reported_ece10"] = old["ece"]
        arm_metrics[arm] = values

    prior = np.asarray(reported["train_class_counts"], dtype=np.float64)
    prior /= prior.sum()
    arm_metrics["train_prior_option_only"] = score(np.tile(prior, (len(rows), 1)), targets)
    prior_record = reported["train_prior_option_only"]
    for key in ("accuracy", "nll", "brier"):
        if abs(arm_metrics["train_prior_option_only"][key] - prior_record[key]) > 5e-6:
            raise ValueError(f"reconstructed train prior {key} differs from original metric")
    arm_metrics["train_prior_option_only"]["legacy_reported_ece10"] = prior_record["ece"]

    return {
        "schema_version": 1,
        "audit_type": "saved_prediction_metric_correction",
        "validation_rows": len(rows),
        "arms": arm_metrics,
        "inputs": {
            "audit_code": str(audit_code_path),
            "audit_code_sha256": audit_code_hash,
            "original_predictions": str(predictions_path),
            "original_predictions_sha256": prediction_hash,
            "reproduction_predictions": str(duplicate_predictions_path),
            "reproduction_predictions_sha256": duplicate_prediction_hash,
            "original_metrics": str(metrics_path),
            "original_metrics_sha256": sha256_file(metrics_path),
        },
        "model_inference": False,
        "feature_extraction": False,
        "optimizer_updates": 0,
        "checkpoint_selection": False,
        "sealed_audit_loaded": False,
        "interpretation": (
            "ECE was recomputed with confidence 1.0 included in the final bin. "
            "Accuracy, NLL, and Brier are checked against the original metric file."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--duplicate-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result = audit(args.predictions, args.metrics, args.duplicate_predictions)
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"audited {result['validation_rows']} saved predictions; no model inference")


if __name__ == "__main__":
    main()
