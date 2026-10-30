"""Recompute a pinned synthetic text validation from frozen features on CPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

from scripts.semantic_state_diagnostic import metrics

EXPECTED_RUN_REPORT_SHA256 = "d9a8aa50d66ff13ea81071416789c9c34b989c78de13c16a30437f0e735642bf"
EXPECTED_FEATURE_CACHE_SHA256 = "6dbe639194d172d4223abeb768baadb6c1a89e0fa631907afcc49c962692d344"
EXPECTED_VALIDATION_SHA256 = "7be1d1624752938553e13e1eecef5363ca2c56dd264da2915b97ddae8d0b60a8"
EXPECTED_READOUT_SHA256 = "12d8278cb0c13c384216ba706c4edd39a30ce6888f6a1900ea7ce26304bf02d0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pad_score_rows(rows: list[np.ndarray]) -> np.ndarray:
    if not rows or any(row.ndim != 1 or len(row) < 2 or not np.isfinite(row).all() for row in rows):
        raise ValueError("finite score vectors with at least two options are required")
    matrix = np.full((len(rows), max(map(len, rows))), -1e30, dtype=np.float64)
    for index, row in enumerate(rows):
        matrix[index, : len(row)] = row
    return matrix


def replay(
    feature_path: Path, readout_path: Path, run_report_path: Path
) -> dict[str, Any]:
    import torch

    from tiny_omni_decision.decision import FrozenFeatureCandidateScorer

    for path in (feature_path, readout_path, run_report_path):
        path.resolve(strict=True)
    hashes = {
        "feature_cache_sha256": sha256_file(feature_path),
        "readout_sha256": sha256_file(readout_path),
        "run_report_sha256": sha256_file(run_report_path),
    }
    run_report = json.loads(run_report_path.read_text(encoding="utf-8"))
    if (
        hashes["feature_cache_sha256"] != EXPECTED_FEATURE_CACHE_SHA256
        or hashes["readout_sha256"] != EXPECTED_READOUT_SHA256
        or hashes["run_report_sha256"] != EXPECTED_RUN_REPORT_SHA256
        or run_report.get("validation_file_sha256") != EXPECTED_VALIDATION_SHA256
        or run_report.get("validation_examples") != 2540
        or run_report.get("best_epoch") != 3
    ):
        raise ValueError("pinned synthetic text validation inputs or selector changed")

    feature_cache = torch.load(feature_path, map_location="cpu", weights_only=True)
    features = feature_cache["validation_features"]
    targets = np.asarray(feature_cache["validation_targets"], dtype=np.int64)
    if len(features) != 2540 or targets.shape != (2540,):
        raise ValueError("unexpected frozen validation feature count")
    if any(
        item.ndim != 2
        or item.shape[1] != 1536
        or item.shape[0] < 2
        or not torch.isfinite(item).all()
        for item in features
    ):
        raise ValueError("invalid frozen option feature row")
    if any(
        target < 0 or target >= len(item)
        for target, item in zip(targets, features, strict=True)
    ):
        raise ValueError("validation target outside supplied option set")

    model = FrozenFeatureCandidateScorer(1536, hidden_size=128)
    state = torch.load(readout_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    start = time.perf_counter()
    with torch.inference_mode():
        rows = [model(item).float().numpy() for item in features]
    inference_seconds = time.perf_counter() - start
    score_matrix = pad_score_rows(rows)
    result_metrics = metrics(score_matrix, targets)
    expected = run_report["best_validation"]
    comparisons = {
        "accuracy_abs_delta": abs(result_metrics["accuracy"] - expected["accuracy"]),
        "nll_abs_delta": abs(result_metrics["nll"] - expected["nll"]),
        "brier_abs_delta": abs(result_metrics["brier"] - expected["brier"]),
        "ece_abs_delta": abs(result_metrics["ece"] - expected["ece_15_bins"]),
    }
    if max(comparisons.values()) > 1e-6:
        raise ValueError(f"CPU replay differs from pinned training report: {comparisons}")
    return {
        "schema_version": 1,
        "status": "PASS_CPU_FROZEN_FEATURE_READOUT_REPLAY",
        "scope": "fixed synthetic Typed Decisions Synth validation; no fit or selection",
        "hashes": hashes,
        "validation_file_sha256": EXPECTED_VALIDATION_SHA256,
        "validation_count": len(features),
        "device": "cpu",
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "training_best_epoch": run_report["best_epoch"],
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "metrics": result_metrics,
        "difference_from_training_report": comparisons,
        "inference_seconds": inference_seconds,
        "seconds_per_decision": inference_seconds / len(features),
        "encoder_executed": False,
        "training_performed": False,
        "sealed_audit_loaded": False,
        "interpretation_limit": (
            "This verifies the saved readout on its pinned cached validation features. "
            "It does not replay the text encoder or establish general text decision quality."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--readout", type=Path, required=True)
    parser.add_argument("--run-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite replay report: {output}")
    report = replay(args.features, args.readout, args.run_report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
