"""Apply the frozen TRAIN-only persistence rule once to HapTile validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from scripts.pretrained_reuse.evaluate_haptile_kinematic_persistence_train import (
    DEADBAND_METERS,
    HORIZON_SECONDS,
    predict_constant_velocity,
    summarize_train_predictions,
)

VALIDATION_CACHE_SHA256 = "ed874a01274e23e6ac8da4645f5bc2a46aca1422df7ca3145c8ddb33c797fa44"
VALIDATION_EXAMPLES = 169
VALIDATION_TASK_GROUPS = 5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_validation_cache(cache_path: Path) -> tuple[dict[str, np.ndarray], str]:
    """Fail closed unless this is the one frozen, exposed development split."""
    if (
        cache_path.name != "decision-features.npz"
        or cache_path.parent.name != "validation-joined-v1"
    ):
        raise ValueError("only the frozen validation-joined-v1 feature cache is allowed")
    metadata_path = cache_path.parent / "result.json"
    if not metadata_path.is_file():
        raise ValueError("validation cache metadata is missing")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    digest = sha256_file(cache_path)
    if (
        metadata.get("split") != "validation"
        or metadata.get("sealed_audit_opened") is not False
        or metadata.get("examples") != VALIDATION_EXAMPLES
        or metadata.get("task_groups") != VALIDATION_TASK_GROUPS
        or metadata.get("feature_file_sha256") != VALIDATION_CACHE_SHA256
        or digest != VALIDATION_CACHE_SHA256
    ):
        raise ValueError("input does not match the frozen isolated validation cache receipt")
    with np.load(cache_path, allow_pickle=False) as archive:
        required = {"state_values", "state_timestamps_ms", "targets", "task_groups", "sample_paths"}
        if not required.issubset(archive.files):
            raise ValueError("validation cache is missing required feature arrays")
        arrays = {name: archive[name] for name in required}
    if (
        arrays["state_values"].shape != (VALIDATION_EXAMPLES, 8, 25)
        or arrays["state_timestamps_ms"].shape != (VALIDATION_EXAMPLES, 8)
        or arrays["targets"].shape != (VALIDATION_EXAMPLES,)
        or arrays["task_groups"].shape != (VALIDATION_EXAMPLES,)
        or arrays["sample_paths"].shape != (VALIDATION_EXAMPLES,)
    ):
        raise ValueError("validation feature array shapes differ from the frozen contract")
    return arrays, digest


def run(cache_path: Path) -> dict[str, Any]:
    arrays, feature_sha256 = load_validation_cache(cache_path)
    sample_ids = [str(item) for item in arrays["sample_paths"].tolist()]
    task_groups = [str(item) for item in arrays["task_groups"].tolist()]
    predictions = [
        predict_constant_velocity(state, timestamps)
        for state, timestamps in zip(
            arrays["state_values"], arrays["state_timestamps_ms"], strict=True
        )
    ]
    summary = summarize_train_predictions(sample_ids, task_groups, arrays["targets"], predictions)
    if summary["task_groups"] != VALIDATION_TASK_GROUPS:
        raise ValueError("validation cache task-group count differs from the frozen contract")
    return {
        "format": "haptile_validation_kinematic_persistence_v1",
        "status": "completed_selection_affected_validation_diagnostic",
        "scope": (
            "previously exposed development validation; fixed rule derived from TRAIN-only "
            "diagnostic; no fitting, checkpoint selection, calibration, or sealed audit"
        ),
        "checkpoint_selection_affected_validation": True,
        "feature_cache_sha256": feature_sha256,
        "rule": {
            "input": "last two aligned XYZ positions from the causal robot-state sequence",
            "horizon_seconds": HORIZON_SECONDS,
            "deadband_meters": DEADBAND_METERS,
            "class_rule": "existing directional_action_label, dominant signed axis, tie x/y/z",
            "parameters_fitted": 0,
        },
        **summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
    started_at = datetime.now(UTC)
    started_clock = time.perf_counter()
    result = run(args.validation_cache)
    result["started_at_utc"] = started_at.isoformat()
    result["finished_at_utc"] = datetime.now(UTC).isoformat()
    result["elapsed_wall_seconds"] = time.perf_counter() - started_clock
    result["runtime"] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "device": "cpu",
        "cuda_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    summary = {key: value for key, value in result.items() if key != "predictions"}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
