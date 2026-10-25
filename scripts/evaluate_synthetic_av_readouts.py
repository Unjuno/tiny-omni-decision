"""Evaluate synthetic AV readout checkpoints on one immutable validation cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.train_synthetic_av_readouts import (
    _construct_model,
    _model_inputs,
    load_training_data,
    sha256_file,
)
from tiny_omni_decision.synthetic_av_evaluation import (
    group_accuracy,
    paired_group_bootstrap_difference,
)


def _pair_exact_accuracy(
    predictions: np.ndarray, targets: np.ndarray, sample_ids: list[str], manifest: dict[str, Any]
) -> float:
    pred_by_id = dict(zip(sample_ids, predictions.tolist(), strict=True))
    target_by_id = dict(zip(sample_ids, targets.tolist(), strict=True))
    rows = {row["sample_id"]: row for row in manifest["examples"]}
    pairs: dict[str, list[str]] = {}
    for sample_id in sample_ids:
        pairs.setdefault(rows[sample_id]["pair_id"], []).append(sample_id)
    if any(len(pair) != 2 for pair in pairs.values()):
        raise ValueError("validation cohort must contain both query-polarity rows per pair")
    return float(
        np.mean(
            [
                all(pred_by_id[sample_id] == target_by_id[sample_id] for sample_id in pair)
                for pair in pairs.values()
            ]
        )
    )


def _audio_shuffle_indices(
    arrays: dict[str, np.ndarray], manifest: dict[str, Any], validation_indices: np.ndarray
) -> np.ndarray:
    ids = arrays["example_ids"].astype(str)
    rows = {row["sample_id"]: row for row in manifest["examples"]}
    group_polarity: dict[tuple[str, str], list[int]] = {}
    for index in validation_indices:
        row = rows[ids[index]]
        group_polarity.setdefault((row["group_id"], row["query_polarity"]), []).append(
            int(index)
        )
    source_by_index: dict[int, int] = {}
    for indices in group_polarity.values():
        ordered = sorted(indices, key=lambda index: rows[ids[index]]["pairing"])
        if len(ordered) != 4:
            raise ValueError("each validation group/query must have all four pairings")
        for offset, index in enumerate(ordered):
            source_by_index[index] = ordered[(offset + 1) % len(ordered)]
    return np.asarray([source_by_index[int(index)] for index in validation_indices])


def _evaluate_controls(
    model: torch.nn.Module,
    all_inputs: dict[str, torch.Tensor],
    arrays: dict[str, np.ndarray],
    manifest: dict[str, Any],
    validation_indices: np.ndarray,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    index = torch.as_tensor(validation_indices, dtype=torch.long, device=device)
    base = {key: value[index] for key, value in all_inputs.items()}
    shuffled_source = torch.as_tensor(
        _audio_shuffle_indices(arrays, manifest, validation_indices),
        dtype=torch.long,
        device=device,
    )
    controls = {
        "full": (base["audio"], base["video"]),
        "audio_only": (base["audio"], torch.zeros_like(base["video"])),
        "video_only": (torch.zeros_like(base["audio"]), base["video"]),
        "missing_both": (torch.zeros_like(base["audio"]), torch.zeros_like(base["video"])),
        "shuffled_audio_pairing": (
            all_inputs["audio"][shuffled_source],
            base["video"],
        ),
        "reverse_audio_time": (base["audio"].flip(dims=(1,)), base["video"]),
        "reverse_both_time": (
            base["audio"].flip(dims=(1,)),
            base["video"].flip(dims=(1,)),
        ),
    }
    rows = {row["sample_id"]: row for row in manifest["examples"]}
    sample_ids = arrays["example_ids"][validation_indices].astype(str).tolist()
    group_ids = arrays["group_ids"][validation_indices].astype(str)
    predictions = []
    report: dict[str, Any] = {}
    for name, (audio, video) in controls.items():
        model.eval()
        with torch.inference_mode():
            logits = model(audio, video, base["query"])
        targets = base["targets"]
        metrics = _evaluate_logits(logits, targets)
        pred = logits.argmax(dim=-1).detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()
        probabilities = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        correct = pred == targets_np
        group_metrics = group_accuracy(correct.astype(np.float64), group_ids)
        metrics["pair_exact_accuracy"] = _pair_exact_accuracy(
            pred, targets_np, sample_ids, manifest
        )
        metrics["group_macro_accuracy"] = float(np.mean(list(group_metrics.values())))
        metrics["group_count"] = len(group_metrics)
        report[name] = {**metrics, "group_accuracy": group_metrics}
        for offset, sample_id in enumerate(sample_ids):
            predictions.append(
                {
                    "sample_id": sample_id,
                    "group_id": rows[sample_id]["group_id"],
                    "pair_id": rows[sample_id]["pair_id"],
                    "control": name,
                    "target": int(targets_np[offset]),
                    "prediction": int(pred[offset]),
                    "probabilities": [float(value) for value in probabilities[offset]],
                }
            )
    return report, predictions


def _evaluate_logits(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    from tiny_omni_decision.synthetic_av_training import classification_metrics

    return classification_metrics(logits, targets, ece_bins=15)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.device != "cpu" and not os.environ.get("TINY_OMNI_GPU_GATE_RESERVED_MIB"):
        raise RuntimeError("CUDA readout evaluation requires the HotUGPU reservation")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    run_path = args.training_run.resolve(strict=True)
    run = json.loads(run_path.read_text(encoding="utf-8"))
    if run.get("status") != "PASS_SYNTHETIC_AV_READOUT_TRAINING":
        raise ValueError("training run receipt is incomplete")
    if run.get("smoke") and not args.allow_smoke:
        raise ValueError("validation evaluation refuses a smoke training run")
    training_runs = [(run_path, run)]
    if args.reference_training_run is not None:
        reference_path = args.reference_training_run.resolve(strict=True)
        reference_run = json.loads(reference_path.read_text(encoding="utf-8"))
        if reference_run.get("status") != "PASS_SYNTHETIC_AV_READOUT_TRAINING":
            raise ValueError("reference training run receipt is incomplete")
        if reference_run.get("smoke") and not args.allow_smoke:
            raise ValueError("validation evaluation refuses a smoke reference run")
        training_runs.append((reference_path, reference_run))
    data = load_training_data(
        args.render_dir.resolve(strict=True),
        args.feature_dir.resolve(strict=True),
        allow_smoke=args.allow_smoke,
    )
    for source_path, source_run in training_runs:
        if source_run.get("manifest_sha256") != data["manifest_sha256"] or source_run.get(
            "feature_cache_sha256"
        ) != data["feature_cache_sha256"]:
            raise ValueError(
                f"training run {source_path} and evaluation inputs do not share frozen data"
            )
        if source_run.get("validation_example_ids") != data["validation_ids"]:
            raise ValueError(
                f"validation example IDs/order differ from training run {source_path}"
            )
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite evaluation output: {output}")
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    inputs = _model_inputs(data["arrays"], device)
    validation_indices = data["validation_indices"]
    result: dict[str, Any] = {
        "status": "RUNNING",
        "training_run_sha256": sha256_file(run_path),
        "reference_training_runs": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path, _ in training_runs[1:]
        ],
        "manifest_sha256": data["manifest_sha256"],
        "feature_cache_sha256": data["feature_cache_sha256"],
        "validation_ids_sha256": hashlib.sha256(
            json.dumps(data["validation_ids"], separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "validation_count": len(validation_indices),
        "device": torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu",
        "results": [],
    }
    all_predictions = []
    architecture_best_group_accuracy: dict[str, dict[str, float]] = {}
    started = time.perf_counter()
    try:
        for source_path, source_run in training_runs:
            for training_result in source_run["results"]:
                architecture = training_result["architecture"]
                model_config = source_run["effective_config"]["models"][architecture]
                for checkpoint_name in ("best", "terminal"):
                    checkpoint_path = source_path.parent / architecture / f"{checkpoint_name}.pt"
                    checkpoint_hash = sha256_file(checkpoint_path)
                    expected_hash = training_result[f"{checkpoint_name}_checkpoint_sha256"]
                    if checkpoint_hash != expected_hash:
                        raise ValueError(
                            f"{architecture} {checkpoint_name} checkpoint hash "
                            "differs from run receipt"
                        )
                    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
                    expected_step = (
                        training_result["best_step"]
                        if checkpoint_name == "best"
                        else training_result["optimizer_updates"]
                    )
                    if payload.get("step") != expected_step:
                        raise ValueError(
                            f"{architecture} {checkpoint_name} checkpoint step "
                            "differs from run receipt"
                        )
                    model = _construct_model(architecture, model_config).to(device)
                    model.load_state_dict(payload["state_dict"], strict=True)
                    controls, predictions = _evaluate_controls(
                        model,
                        inputs,
                        data["arrays"],
                        data["manifest"],
                        validation_indices,
                        device,
                    )
                    all_predictions.extend(
                        {
                            "architecture": architecture,
                            "checkpoint": checkpoint_name,
                            "source_training_run_sha256": sha256_file(source_path),
                            **row,
                        }
                        for row in predictions
                    )
                    record = {
                        "architecture": architecture,
                        "checkpoint": checkpoint_name,
                        "source_training_run_sha256": sha256_file(source_path),
                        "checkpoint_step": payload["step"],
                        "checkpoint_sha256": checkpoint_hash,
                        "controls": controls,
                    }
                    result["results"].append(record)
                    if checkpoint_name == "best":
                        architecture_best_group_accuracy[architecture] = controls["full"][
                            "group_accuracy"
                        ]
        reference_name = "temporal_lstm"
        if reference_name in architecture_best_group_accuracy:
            result["paired_group_bootstrap_full_best"] = {
                name: paired_group_bootstrap_difference(
                    architecture_best_group_accuracy[reference_name],
                    architecture_best_group_accuracy[name],
                    replicates=int(
                        run["effective_config"]["validation"].get(
                            "grouped_bootstrap_replicates", 10_000
                        )
                    ),
                    seed=int(run["effective_config"]["seed"]),
                )
                for name in architecture_best_group_accuracy
                if name != reference_name
            }
        else:
            result["paired_group_bootstrap_full_best"] = None
        predictions_path = output / "predictions.jsonl"
        with predictions_path.open("w", encoding="utf-8") as stream:
            for row in all_predictions:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        result.update(
            {
                "status": "PASS_SYNTHETIC_AV_READOUT_VALIDATION",
                "prediction_sha256": sha256_file(predictions_path),
                "prediction_count": len(all_predictions),
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        (output / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    except Exception as error:
        result.update(
            {
                "status": "FAILED",
                "finished_elapsed_seconds": time.perf_counter() - started,
                "error": f"{type(error).__name__}: {error}",
            }
        )
        (output / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--reference-training-run", type=Path)
    parser.add_argument("--render-dir", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--allow-smoke", action="store_true")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
