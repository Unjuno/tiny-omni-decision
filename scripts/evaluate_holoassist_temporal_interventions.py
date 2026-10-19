"""Measure fixed validation interventions on a completed text-free AV readout."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.audit_holoassist_metadata import sha256_file
from scripts.train_holoassist_temporal_readout import (
    collate_sessions,
    load_feature_split,
    metric_summary,
    validate_feature_receipt,
    validate_split_identity,
)
from tiny_omni_decision.temporal_decision import TaskConditionedTemporalDecision
from tiny_omni_decision.temporal_interventions import (
    INTERVENTIONS,
    apply_feature_intervention,
)


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def evaluate_intervention(
    model: TaskConditionedTemporalDecision,
    data: dict[str, Any],
    task_type_to_id: dict[str, int],
    *,
    device: torch.device,
    batch_sessions: int,
    ece_bins: int,
    intervention: str,
    seed: int = 17,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, int]]:
    altered = apply_feature_intervention(data, intervention, seed=seed)
    view = dict(data, video=altered.video, audio=altered.audio)
    logits_rows: list[torch.Tensor] = []
    gate_rows: list[torch.Tensor] = []
    target_rows: list[int] = []
    task_type_rows: list[str] = []
    gate_rows_by_task_type: dict[str, list[torch.Tensor]] = {}
    predictions: list[dict[str, Any]] = []
    order = sorted(range(len(data["session_names"])), key=lambda i: str(data["session_names"][i]))
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(order), batch_sessions):
            selected = order[start : start + batch_sessions]
            video, audio, _text, lengths, targets, mask = collate_sessions(
                view, selected, device, include_text=False
            )
            task_ids = torch.tensor(
                [
                    task_type_to_id[str(data["task_type"][data["session_indices"][position][0]])]
                    for position in selected
                ],
                dtype=torch.int64,
                device=device,
            )
            video_present = torch.ones(video.shape[:2], dtype=torch.bool, device=device)
            audio_present = torch.ones_like(video_present)
            for row, position in enumerate(selected):
                indices = data["session_indices"][position]
                length = len(indices)
                video_present[row, :length] = torch.from_numpy(altered.video_present[indices]).to(
                    device
                )
                audio_present[row, :length] = torch.from_numpy(altered.audio_present[indices]).to(
                    device
                )
            output = model(
                video,
                audio,
                task_ids,
                lengths=lengths,
                video_present=video_present,
                audio_present=audio_present,
            )
            for row, position in enumerate(selected):
                indices = data["session_indices"][position]
                for step, index in enumerate(indices):
                    if not bool(mask[row, step].item()):
                        continue
                    target = int(targets[row, step].item())
                    task_type = str(data["task_type"][index])
                    logits = output.logits[row, step].detach().float().cpu()
                    gates = output.modality_weights[row, step].detach().float().cpu()
                    probabilities = torch.softmax(logits, dim=-1).tolist()
                    logits_rows.append(logits)
                    gate_rows.append(gates)
                    target_rows.append(target)
                    task_type_rows.append(task_type)
                    gate_rows_by_task_type.setdefault(task_type, []).append(gates)
                    predictions.append(
                        {
                            "id": str(data["sample_id"][index]),
                            "session_id": str(data["session_id"][index]),
                            "task_type": task_type,
                            "window_index": int(data["window_index"][index]),
                            "target": target,
                            "prediction": int(np.argmax(probabilities)),
                            "probabilities": probabilities,
                        }
                    )
    if not logits_rows:
        raise ValueError("intervention produced no labelled validation examples")
    all_metrics = metric_summary(
        torch.stack(logits_rows).to(device),
        torch.tensor(target_rows, dtype=torch.int64, device=device),
        bins=ece_bins,
    )
    by_task_type: dict[str, Any] = {}
    for task_type in sorted(set(task_type_rows)):
        positions = [i for i, value in enumerate(task_type_rows) if value == task_type]
        by_task_type[task_type] = metric_summary(
            torch.stack([logits_rows[i] for i in positions]).to(device),
            torch.tensor([target_rows[i] for i in positions], dtype=torch.int64, device=device),
            bins=ece_bins,
        )
    all_metrics["by_task_type"] = by_task_type
    gate_values = torch.stack(gate_rows)
    all_metrics["mean_gate_weights"] = {
        "video": float(gate_values[:, 0].mean().item()),
        "audio": float(gate_values[:, 1].mean().item()),
    }
    for task_type, values in by_task_type.items():
        task_gates = torch.stack(gate_rows_by_task_type[task_type])
        values["mean_gate_weights"] = {
            "video": float(task_gates[:, 0].mean().item()),
            "audio": float(task_gates[:, 1].mean().item()),
        }
    all_metrics["task_type_macro"] = {
        metric: float(np.mean([row[metric] for row in by_task_type.values()]))
        for metric in ("accuracy", "nll", "brier", "ece")
    }
    intervention_info = {
        "eligible_session_count": altered.eligible_session_count,
        "shuffle_group_count": altered.shuffle_group_count,
    }
    if intervention == "audio_shuffle_session" and not altered.eligible_session_count:
        raise ValueError("validation has no session groups eligible for audio shuffling")
    return all_metrics, predictions, intervention_info


def _validate_prediction_parity(
    saved_path: Path, replay: list[dict[str, Any]], *, tolerance: float = 1e-6
) -> float:
    saved = json.loads(saved_path.read_text(encoding="utf-8"))
    if len(saved) != len(replay):
        raise ValueError("baseline replay changed the saved validation count")
    max_delta = 0.0
    identity_fields = ("id", "session_id", "task_type", "window_index", "target", "prediction")
    for before, after in zip(saved, replay, strict=True):
        if any(before[field] != after[field] for field in identity_fields):
            raise ValueError("baseline replay changed validation decisions or ordering")
        max_delta = max(
            max_delta,
            float(
                np.max(
                    np.abs(
                        np.asarray(before["probabilities"], dtype=np.float64)
                        - np.asarray(after["probabilities"], dtype=np.float64)
                    )
                )
            ),
        )
    if not math.isfinite(max_delta) or max_delta > tolerance:
        raise ValueError(f"baseline replay probability delta {max_delta} exceeds tolerance")
    return max_delta


def run_interventions(
    *,
    run_dir: Path,
    train_features: Path,
    validation_features: Path,
    feature_receipt_path: Path,
    output_dir: Path,
    device_name: str = "cuda",
    seed: int = 17,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite intervention output: {output_dir}")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    run_meta_path = run_dir / "run-metadata.json"
    run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
    if run_meta.get("status") != "complete" or run_meta.get("sealed_audit_loaded") is not False:
        raise ValueError("a completed development run with sealed audit excluded is required")
    config = run_meta.get("config", {})
    if config.get("conditioning") != "task_type_id":
        raise ValueError("intervention runner requires the text-free task-type condition")
    expected_arms = set(config.get("arms", []))
    completed_arms = {str(result.get("arm")) for result in run_meta.get("results", [])}
    if not expected_arms or completed_arms != expected_arms:
        raise ValueError("all configured readout arms must finish before interventions")
    if run_meta.get("train_features_sha256") != sha256_file(train_features):
        raise ValueError("training feature hash differs from the completed run")
    if run_meta.get("validation_features_sha256") != sha256_file(validation_features):
        raise ValueError("validation feature hash differs from the completed run")
    if run_meta.get("feature_receipt_sha256") != sha256_file(feature_receipt_path):
        raise ValueError("feature receipt hash differs from the completed run")
    train = load_feature_split(train_features, require_text=False)
    validation = load_feature_split(validation_features, require_text=False)
    validate_split_identity(train, validation)
    expected_encoders = {name: config["encoders"][name] for name in ("video", "audio")}
    feature_receipt = json.loads(feature_receipt_path.read_text(encoding="utf-8"))
    validate_feature_receipt(
        feature_receipt,
        train_path=train_features,
        validation_path=validation_features,
        train=train,
        validation=validation,
        expected_encoders=expected_encoders,
    )
    task_type_to_id = {
        str(task): int(index) for task, index in run_meta.get("task_type_vocabulary", {}).items()
    }
    if not task_type_to_id or not set(map(str, validation["task_type"])) <= set(task_type_to_id):
        raise ValueError("validation task types do not match the training vocabulary")
    run_result = json.loads((run_dir / "joint" / "result.json").read_text(encoding="utf-8"))
    best_info = json.loads((run_dir / "joint" / "best-checkpoint.json").read_text(encoding="utf-8"))
    checkpoint_path = run_dir / "joint" / "best-checkpoint.pt"
    if (
        run_result.get("arm") != "joint"
        or best_info.get("step") != run_result.get("best_step")
        or best_info.get("validation_nll") != run_result.get("best_validation_nll")
        or run_result.get("best_checkpoint_sha256") != sha256_file(checkpoint_path)
    ):
        raise ValueError("joint best checkpoint metadata/hash does not reconcile")
    model = TaskConditionedTemporalDecision(
        task_count=len(task_type_to_id),
        hidden_size=int(config["hidden_size"]),
        modality_mode="joint",
    ).to(device_name)
    state = torch.load(checkpoint_path, map_location=device_name, weights_only=True)
    model.load_state_dict(state, strict=True)
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    output_dir.mkdir(parents=True)
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    metadata = {
        "status": "running",
        "experiment_id": "holoassist-temporal-av-interventions-v1",
        "source_commit": source_commit,
        "base_run_dir": str(run_dir.resolve()),
        "base_run_metadata_sha256": sha256_file(run_meta_path),
        "base_checkpoint": str(checkpoint_path.resolve()),
        "base_checkpoint_sha256": sha256_file(checkpoint_path),
        "base_best_step": int(run_result["best_step"]),
        "seed": seed,
        "device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
        "torch_version": torch.__version__,
        "validation_feature_sha256": sha256_file(validation_features),
        "validation_receipt_sha256": sha256_file(feature_receipt_path),
        "validation_sample_ids_sha256": hashlib.sha256(
            "\n".join(map(str, validation["sample_id"])).encode()
        ).hexdigest(),
        "validation_session_count": len(validation["session_names"]),
        "validation_window_count": len(validation["sample_id"]),
        "trainable_parameters": model.trainable_parameter_count,
        "best_checkpoint_bytes": checkpoint_path.stat().st_size,
        "interventions": list(INTERVENTIONS),
        "test_data_loaded": False,
        "sealed_audit_loaded": False,
    }
    _write_json(output_dir / "run-metadata.json", metadata)
    started = time.perf_counter()
    results = {}
    batch_sessions = int(config["batch_sessions"])
    ece_bins = int(config["ece_bins"])
    for intervention in INTERVENTIONS:
        intervention_started = time.perf_counter()
        metrics, predictions, intervention_info = evaluate_intervention(
            model,
            validation,
            task_type_to_id,
            device=device,
            batch_sessions=batch_sessions,
            ece_bins=ece_bins,
            intervention=intervention,
            seed=seed,
        )
        predictions_path = output_dir / f"predictions-{intervention}.json"
        _write_json(predictions_path, predictions)
        results[intervention] = {
            "metrics": metrics,
            "evaluation_seconds": time.perf_counter() - intervention_started,
            **intervention_info,
            "predictions_sha256": sha256_file(predictions_path),
            "prediction_count": len(predictions),
        }
        if device.type == "cuda":
            results[intervention]["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(
                device
            )
            results[intervention]["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved(
                device
            )
        if intervention == "baseline":
            best_step = int(run_result["best_step"])
            saved_predictions = run_dir / "joint" / f"predictions-step-{best_step:06d}.json"
            results[intervention]["saved_checkpoint_replay_max_probability_delta"] = (
                _validate_prediction_parity(saved_predictions, predictions)
            )
        _write_json(output_dir / "run-metadata.json", {**metadata, "results": results})
    result = {
        "status": "complete_fixed_validation_interventions",
        "elapsed_seconds": time.perf_counter() - started,
        "results": results,
    }
    _write_json(output_dir / "result.json", result)
    metadata.update(status="complete", finished_at_utc=datetime.now(UTC).isoformat())
    _write_json(output_dir / "run-metadata.json", {**metadata, "results": results})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--validation-features", type=Path, required=True)
    parser.add_argument("--feature-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    result = run_interventions(
        run_dir=args.run_dir,
        train_features=args.train_features,
        validation_features=args.validation_features,
        feature_receipt_path=args.feature_receipt,
        output_dir=args.output_dir,
        device_name=args.device,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
