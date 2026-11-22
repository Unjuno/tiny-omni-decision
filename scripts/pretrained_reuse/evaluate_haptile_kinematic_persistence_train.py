"""Evaluate a fixed constant-velocity rule on the frozen HapTile TRAIN cache only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from tiny_omni_decision.haptile_action import ACTION_LABELS, directional_action_label

TRAIN_CACHE_SHA256 = "4d0b8bfc4d9d279ec04e76811f6a9fc6a8410ceb3d498a33cca9cc0112ba42c2"
TRAIN_EXAMPLES = 1112
TRAIN_TASK_GROUPS = 24
HORIZON_SECONDS = 0.5
DEADBAND_METERS = 0.005


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def predict_constant_velocity(
    state_values: np.ndarray,
    state_timestamps_ms: np.ndarray,
    *,
    horizon_seconds: float = HORIZON_SECONDS,
    deadband_meters: float = DEADBAND_METERS,
) -> str:
    """Project the last observed XYZ velocity, then apply the frozen label rule."""
    values = np.asarray(state_values, dtype=np.float64)
    timestamps = np.asarray(state_timestamps_ms, dtype=np.float64)
    if values.shape != (8, 25) or timestamps.shape != (8,):
        raise ValueError("state values and timestamps must have shapes [8,25] and [8]")
    if not np.isfinite(values).all() or not np.isfinite(timestamps).all():
        raise ValueError("state values and timestamps must be finite")
    elapsed_seconds = float(timestamps[-1] - timestamps[-2]) / 1000.0
    if not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0:
        raise ValueError("last two state timestamps must be strictly increasing")
    if not math.isfinite(horizon_seconds) or horizon_seconds <= 0:
        raise ValueError("horizon_seconds must be finite and positive")
    displacement = (values[-1, :3] - values[-2, :3]) * (horizon_seconds / elapsed_seconds)
    return directional_action_label((0.0, 0.0, 0.0), displacement, deadband_m=deadband_meters)


def summarize_train_predictions(
    sample_ids: list[str],
    task_groups: list[str],
    targets: np.ndarray,
    predictions: list[str],
) -> dict[str, Any]:
    """Summarize fixed-rule predictions without fitting or checkpoint selection."""
    target_ids = np.asarray(targets, dtype=np.int64)
    if (
        not sample_ids
        or len(sample_ids) != len(task_groups)
        or len(sample_ids) != len(target_ids)
        or len(sample_ids) != len(predictions)
    ):
        raise ValueError("IDs, groups, targets and predictions must have equal nonzero length")
    if (
        len(set(sample_ids)) != len(sample_ids)
        or not set(np.unique(target_ids)).issubset(range(len(ACTION_LABELS)))
        or any(prediction not in ACTION_LABELS for prediction in predictions)
        or any(not group for group in task_groups)
    ):
        raise ValueError("sample identities, groups or class labels are invalid")

    prediction_ids = np.asarray([ACTION_LABELS.index(item) for item in predictions])
    correct = prediction_ids == target_ids
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(task_groups):
        grouped[group].append(index)
    per_task: dict[str, dict[str, float | int]] = {}
    for group, indices in sorted(grouped.items()):
        target_slice = target_ids[indices]
        correct_slice = correct[indices]
        per_task[group] = {
            "examples": len(indices),
            "persistence_correct": int(correct_slice.sum()),
            "persistence_accuracy": float(correct_slice.mean()),
            "always_hold_accuracy": float((target_slice == ACTION_LABELS.index("hold")).mean()),
        }
    hold_correct = target_ids == ACTION_LABELS.index("hold")
    return {
        "examples": len(sample_ids),
        "task_groups": len(grouped),
        "persistence_correct": int(correct.sum()),
        "persistence_accuracy": float(correct.mean()),
        "persistence_macro_task_accuracy": float(
            np.mean([item["persistence_accuracy"] for item in per_task.values()])
        ),
        "always_hold_correct": int(hold_correct.sum()),
        "always_hold_accuracy": float(hold_correct.mean()),
        "always_hold_macro_task_accuracy": float(
            np.mean([item["always_hold_accuracy"] for item in per_task.values()])
        ),
        "tasks_persistence_beats_always_hold": sum(
            item["persistence_accuracy"] > item["always_hold_accuracy"]
            for item in per_task.values()
        ),
        "per_task": per_task,
        "predictions": [
            {
                "sample_id": sample_id,
                "task_group": group,
                "target": ACTION_LABELS[int(target)],
                "prediction": prediction,
            }
            for sample_id, group, target, prediction in zip(
                sample_ids, task_groups, target_ids, predictions, strict=True
            )
        ],
    }


def load_train_cache(cache_path: Path) -> tuple[dict[str, np.ndarray], str]:
    """Fail closed unless the input is exactly the frozen, isolated TRAIN cache."""
    if cache_path.name != "decision-features.npz" or cache_path.parent.name != "train-joined-v1":
        raise ValueError("only the frozen train-joined-v1 decision feature cache is allowed")
    metadata_path = cache_path.parent / "result.json"
    if not metadata_path.is_file():
        raise ValueError("train cache metadata is missing")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    digest = sha256_file(cache_path)
    if (
        metadata.get("split") != "train"
        or metadata.get("sealed_audit_opened") is not False
        or metadata.get("examples") != TRAIN_EXAMPLES
        or metadata.get("task_groups") != TRAIN_TASK_GROUPS
        or metadata.get("feature_file_sha256") != TRAIN_CACHE_SHA256
        or digest != TRAIN_CACHE_SHA256
    ):
        raise ValueError("input does not match the frozen isolated TRAIN cache receipt")
    with np.load(cache_path, allow_pickle=False) as archive:
        required = {"state_values", "state_timestamps_ms", "targets", "task_groups", "sample_paths"}
        if not required.issubset(archive.files):
            raise ValueError("train cache is missing required feature arrays")
        arrays = {name: archive[name] for name in required}
    if (
        arrays["state_values"].shape != (TRAIN_EXAMPLES, 8, 25)
        or arrays["state_timestamps_ms"].shape != (TRAIN_EXAMPLES, 8)
        or arrays["targets"].shape != (TRAIN_EXAMPLES,)
        or arrays["task_groups"].shape != (TRAIN_EXAMPLES,)
        or arrays["sample_paths"].shape != (TRAIN_EXAMPLES,)
    ):
        raise ValueError("train feature array shapes differ from the frozen contract")
    return arrays, digest


def run(cache_path: Path) -> dict[str, Any]:
    arrays, feature_sha256 = load_train_cache(cache_path)
    sample_ids = [str(item) for item in arrays["sample_paths"].tolist()]
    task_groups = [str(item) for item in arrays["task_groups"].tolist()]
    predictions = [
        predict_constant_velocity(state, timestamps)
        for state, timestamps in zip(
            arrays["state_values"], arrays["state_timestamps_ms"], strict=True
        )
    ]
    summary = summarize_train_predictions(sample_ids, task_groups, arrays["targets"], predictions)
    if summary["task_groups"] != TRAIN_TASK_GROUPS:
        raise ValueError("train cache task-group count differs from the frozen contract")
    return {
        "format": "haptile_train_kinematic_persistence_v1",
        "status": "completed_train_only_fixed_rule_diagnostic",
        "scope": "train_only; fixed rule; no fitting; no validation/calibration/sealed audit",
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
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
    started_at = datetime.now(UTC)
    started_clock = time.perf_counter()
    result = run(args.train_cache)
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
    print(
        json.dumps({key: value for key, value in result.items() if key != "predictions"}, indent=2)
    )


if __name__ == "__main__":
    main()
