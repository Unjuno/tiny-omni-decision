"""Group-bootstrap paired uncertainty from saved text-free audio predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from tiny_omni_decision.dataset import sha256_file


def _metrics(
    probabilities: np.ndarray, targets: np.ndarray, weights: np.ndarray
) -> dict[str, float]:
    count = float(weights.sum())
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predictions == targets
    eps = np.finfo(np.float64).tiny
    nll = -np.log(np.clip(probabilities[np.arange(len(targets)), targets], eps, 1)).dot(
        weights
    ) / count
    one_hot = np.eye(probabilities.shape[1], dtype=np.float64)[targets]
    brier = (((probabilities - one_hot) ** 2).sum(axis=1) * weights).sum() / count
    ece = 0.0
    for bin_index in range(15):
        low, high = bin_index / 15, (bin_index + 1) / 15
        mask = (confidence >= low) & (
            confidence < high if bin_index < 14 else confidence <= high
        )
        bin_weight = weights[mask].sum()
        if bin_weight:
            accuracy = np.average(correct[mask], weights=weights[mask])
            mean_confidence = np.average(confidence[mask], weights=weights[mask])
            ece += bin_weight / count * abs(accuracy - mean_confidence)
    return {
        "accuracy": float(np.average(correct, weights=weights)),
        "nll": float(nll),
        "brier": float(brier),
        "ece_15_bins": float(ece),
    }


def paired_speaker_bootstrap(
    rows: list[dict[str, Any]], *, seed: int = 17, replicates: int = 10_000
) -> dict[str, Any]:
    if not rows or replicates < 1:
        raise ValueError("rows and a positive replicate count are required")
    speakers = sorted({row["speaker_group_sha256"] for row in rows})
    if len(speakers) < 2:
        raise ValueError("at least two independent speakers are required")
    speaker_index = {speaker: index for index, speaker in enumerate(speakers)}
    row_groups = np.asarray(
        [speaker_index[row["speaker_group_sha256"]] for row in rows], dtype=np.int64
    )
    targets = np.asarray([row["target_index"] for row in rows], dtype=np.int64)
    raw = np.asarray([row["probabilities_uncalibrated"] for row in rows], dtype=np.float64)
    calibrated = np.asarray(
        [row["probabilities_temperature_scaled"] for row in rows], dtype=np.float64
    )
    if raw.ndim != 2 or calibrated.shape != raw.shape or raw.shape[0] != len(rows):
        raise ValueError("prediction probability arrays have inconsistent shapes")
    if targets.min() < 0 or targets.max() >= raw.shape[1]:
        raise ValueError("target index outside probability class range")
    if (
        not np.isfinite(raw).all()
        or not np.isfinite(calibrated).all()
        or (raw < 0).any()
        or (calibrated < 0).any()
        or not np.allclose(raw.sum(axis=1), 1.0, atol=1e-6)
        or not np.allclose(calibrated.sum(axis=1), 1.0, atol=1e-6)
    ):
        raise ValueError("invalid probability vectors")

    all_weights = np.ones(len(rows), dtype=np.float64)
    point_raw = _metrics(raw, targets, all_weights)
    point_calibrated = _metrics(calibrated, targets, all_weights)
    metric_names = ("accuracy", "nll", "brier", "ece_15_bins")
    differences = {name: np.empty(replicates, dtype=np.float64) for name in metric_names}
    rng = np.random.default_rng(seed)
    for replicate in range(replicates):
        sampled = rng.integers(0, len(speakers), size=len(speakers))
        group_counts = np.bincount(sampled, minlength=len(speakers)).astype(np.float64)
        row_weights = group_counts[row_groups]
        before = _metrics(raw, targets, row_weights)
        after = _metrics(calibrated, targets, row_weights)
        for name in metric_names:
            differences[name][replicate] = after[name] - before[name]
    result = {}
    for name, values in differences.items():
        result[name] = {
            "point_difference_calibrated_minus_raw": point_calibrated[name]
            - point_raw[name],
            "percentile_95_ci": [float(x) for x in np.quantile(values, [0.025, 0.975])],
            "fraction_bootstrap_differences_below_zero": float(np.mean(values < 0)),
        }
    return {
        "examples": len(rows),
        "independent_speaker_groups": len(speakers),
        "seed": seed,
        "replicates": replicates,
        "resampling_unit": "speaker_group_sha256; sample groups with replacement",
        "raw_point_metrics": point_raw,
        "temperature_scaled_point_metrics": point_calibrated,
        "paired_differences": result,
        "scope": (
            "retrospective uncertainty on previously inspected development validation; "
            "not blind"
        ),
    }


def run(
    predictions_path: Path,
    run_report_path: Path,
    output_path: Path,
    *,
    replicates: int = 10_000,
) -> dict[str, Any]:
    predictions_path = predictions_path.resolve(strict=True)
    run_report_path = run_report_path.resolve(strict=True)
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite bootstrap report: {output_path}")
    rows = [
        json.loads(line)
        for line in predictions_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    run_report = json.loads(run_report_path.read_text(encoding="utf-8"))
    class_ids = run_report.get("config", {}).get("classes")
    if not isinstance(class_ids, list) or sha256_file(predictions_path) != run_report.get(
        "predictions_sha256"
    ):
        raise ValueError("prediction file does not match the pinned run report")
    for row in rows:
        row["target_index"] = class_ids.index(row["target_class_id"])
        row["class_ids"] = class_ids
        if len(class_ids) != len(row["probabilities_uncalibrated"]):
            raise ValueError("class IDs and probability vectors differ in length")
    report = paired_speaker_bootstrap(rows, replicates=replicates)
    report["source_predictions_sha256"] = sha256_file(predictions_path)
    report["source_run_report_sha256"] = sha256_file(run_report_path)
    report["source_script_sha256"] = sha256_file(Path(__file__).resolve())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--run-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=10_000)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.predictions,
                args.run_report,
                args.output,
                replicates=args.replicates,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
