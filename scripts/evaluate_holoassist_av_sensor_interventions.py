"""Evaluate sensor interventions on a fixed HoloAssist AVS checkpoint (CPU only)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from scripts.analyze_holoassist_paired_bootstrap import (  # noqa: E402
    load_predictions,
    paired_session_bootstrap,
)
from scripts.audit_holoassist_metadata import sha256_file  # noqa: E402
from scripts.train_holoassist_temporal_readout import (  # noqa: E402
    _configured_hidden_size,
    _evaluate,
    attach_imu_features,
    load_feature_split,
)
from tiny_omni_decision.temporal_decision import (  # noqa: E402
    TaskConditionedAVSensorTemporalDecision,
)


def derange_sensor_sessions(data: dict[str, Any]) -> tuple[dict[str, Any], dict[str, int]]:
    """Rotate whole IMU sequences within exact task and session-length groups."""
    result = dict(data)
    result["sensor"] = np.asarray(data["sensor"]).copy()
    groups: dict[tuple[str, int], list[int]] = {}
    for index, indices in enumerate(data["session_indices"]):
        first = int(indices[0])
        key = (str(data["task_type"][first]), len(indices))
        groups.setdefault(key, []).append(index)
    changed = 0
    eligible = 0
    for members in groups.values():
        members.sort(key=lambda item: str(data["session_names"][item]))
        if len(members) < 2:
            continue
        eligible += len(members)
        donor_map = {
            member: members[(position + 1) % len(members)]
            for position, member in enumerate(members)
        }
        for receiver in members:
            receiver_rows = data["session_indices"][receiver]
            donor_rows = data["session_indices"][donor_map[receiver]]
            if np.array_equal(data["sensor"][receiver_rows], data["sensor"][donor_rows]):
                continue
            result["sensor"][receiver_rows] = data["sensor"][donor_rows]
            changed += 1
    return result, {
        "sessions_total": len(data["session_indices"]),
        "sessions_eligible": eligible,
        "sessions_changed": changed,
        "singleton_task_length_groups": sum(
            len(members) for members in groups.values() if len(members) < 2
        ),
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite intervention output: {args.output_dir}")
    if args.device != "cpu":
        raise ValueError("this intervention diagnostic is restricted to CPU")
    run = json.loads(args.run_metadata.read_text(encoding="utf-8"))
    if run.get("status") != "complete" or run.get("sealed_audit_loaded") is not False:
        raise ValueError("a completed development run with no sealed audit is required")
    if run.get("source_commit") != "c80551e9197efb22d3f32be2cc93f97a256e44e2":
        raise ValueError("unexpected source run commit")
    config = run["config"]
    if config.get("seed") != 17 or config.get("max_optimizer_steps") != 2048:
        raise ValueError("unexpected source run configuration")
    arm_dir = args.run_metadata.parent / "avs"
    selected = json.loads((arm_dir / "best-checkpoint.json").read_text(encoding="utf-8"))
    if selected.get("step") != 256:
        raise ValueError("expected the recorded AVS best checkpoint at step 256")
    checkpoint = arm_dir / "best-checkpoint.pt"
    avs_result = next((row for row in run["results"] if row.get("arm") == "avs"), None)
    if avs_result is None or avs_result.get("best_step") != selected["step"]:
        raise ValueError("AVS checkpoint selector metadata is inconsistent")
    expected_hash = avs_result["best_checkpoint_sha256"]
    if sha256_file(checkpoint) != expected_hash:
        raise ValueError("AVS checkpoint hash differs from completed run metadata")
    if sha256_file(args.validation_features) != run["validation_features_sha256"]:
        raise ValueError("validation feature hash differs from the completed run")
    if sha256_file(args.imu_validation_features) != run["imu_validation_features_sha256"]:
        raise ValueError("IMU validation feature hash differs from the completed run")
    if sha256_file(args.normalization) != run["imu_normalization_sha256"]:
        raise ValueError("sensor normalization hash differs from the completed run")

    validation = load_feature_split(args.validation_features, require_text=False)
    receipt = json.loads(args.imu_receipt.read_text(encoding="utf-8"))
    if sha256_file(args.imu_receipt) != run["imu_receipt_sha256"]:
        raise ValueError("IMU receipt hash differs from the completed run")
    validation = attach_imu_features(
        validation,
        args.imu_validation_features,
        split="validation",
        receipt=receipt,
    )
    normalization = json.loads(args.normalization.read_text(encoding="utf-8"))
    mean = np.asarray(normalization["mean"], dtype=np.float32)
    std = np.asarray(normalization["std"], dtype=np.float32)
    if (
        mean.shape != (630,)
        or std.shape != (630,)
        or not np.isfinite(mean).all()
        or not np.isfinite(std).all()
        or np.any(std <= 0)
    ):
        raise ValueError("invalid train-fitted sensor normalization")
    validation["sensor"] = np.clip((validation["sensor"] - mean) / std, -10, 10)

    task_map = run["task_type_vocabulary"]
    model = TaskConditionedAVSensorTemporalDecision(
        task_count=len(task_map),
        hidden_size=_configured_hidden_size(config),
        modality_mode="avs",
    ).cpu()
    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True
    )
    model.eval()

    args.output_dir.mkdir(parents=True)
    results: dict[str, Any] = {
        "status": "running",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_run_metadata_sha256": sha256_file(args.run_metadata),
        "source_checkpoint_sha256": sha256_file(checkpoint),
        "validation_feature_sha256": sha256_file(args.validation_features),
        "imu_validation_feature_sha256": sha256_file(args.imu_validation_features),
        "imu_receipt_sha256": sha256_file(args.imu_receipt),
        "normalization_sha256": sha256_file(args.normalization),
        "device": "cpu",
        "sealed_audit_loaded": False,
        "checkpoint_step": selected["step"],
    }
    _write_atomic(args.output_dir / "run-metadata.json", results)
    try:
        baseline_metrics, baseline_rows = _evaluate(
            model,
            validation,
            torch.device("cpu"),
            config["batch_sessions"],
            config["ece_bins"],
            task_map,
        )
        stored_path = arm_dir / f"predictions-step-{selected['step']:06d}.json"
        stored_rows = json.loads(stored_path.read_text(encoding="utf-8"))
        if len(stored_rows) != len(baseline_rows):
            raise ValueError("baseline replay prediction count differs from stored predictions")
        for expected, observed in zip(stored_rows, baseline_rows, strict=True):
            if any(
                expected[field] != observed[field]
                for field in ("id", "session_id", "target", "prediction")
            ):
                raise ValueError("baseline replay identity or prediction differs")
            delta = np.max(
                np.abs(
                    np.asarray(expected["probabilities"]) - np.asarray(observed["probabilities"])
                )
            )
            if delta > 1e-6:
                raise ValueError(f"baseline replay probability parity failed: {delta}")

        missing = dict(validation)
        missing["sensor"] = np.zeros_like(validation["sensor"])
        shuffled, shuffle_counts = derange_sensor_sessions(validation)
        interventions = {
            "baseline_replay": (validation, baseline_metrics, baseline_rows),
        }
        for name, altered in (
            ("sensor_missing_zero", missing),
            ("sensor_same_task_session_derangement", shuffled),
        ):
            metrics, rows = _evaluate(
                model,
                altered,
                torch.device("cpu"),
                config["batch_sessions"],
                config["ece_bins"],
                task_map,
            )
            interventions[name] = (altered, metrics, rows)
        all_predictions = {}
        for name, (_, metrics, rows) in interventions.items():
            prediction_path = args.output_dir / f"predictions-{name}.json"
            _write_atomic(prediction_path, rows)
            all_predictions[name] = load_predictions(prediction_path)
            results.setdefault("conditions", {})[name] = {
                "metrics": metrics,
                "predictions_sha256": sha256_file(prediction_path),
            }
        results["same_task_derangement"] = shuffle_counts
        results["paired_session_bootstrap"] = {
            name: paired_session_bootstrap(
                all_predictions["baseline_replay"],
                all_predictions[name],
                replicates=10_000,
                seed=1701,
            )
            for name in ("sensor_missing_zero", "sensor_same_task_session_derangement")
        }
        weights = np.asarray([row["modality_weights"] for row in baseline_rows], dtype=np.float64)
        results["baseline_gate_weights"] = {
            "mean": weights.mean(axis=0).tolist(),
            "std": weights.std(axis=0).tolist(),
            "order": ["video", "audio", "sensor"],
        }
        results["status"] = "complete"
        results["finished_at_utc"] = datetime.now(UTC).isoformat()
        _write_atomic(args.output_dir / "run-metadata.json", results)
    except Exception as exc:
        results["status"] = "failed"
        results["failure"] = f"{type(exc).__name__}: {exc}"
        results["finished_at_utc"] = datetime.now(UTC).isoformat()
        _write_atomic(args.output_dir / "run-metadata.json", results)
        raise
    return results


def _write_atomic(path: Path, data: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-metadata", type=Path, required=True)
    parser.add_argument("--validation-features", type=Path, required=True)
    parser.add_argument("--imu-validation-features", type=Path, required=True)
    parser.add_argument("--imu-receipt", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    print(json.dumps(evaluate(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
