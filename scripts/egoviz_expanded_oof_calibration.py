"""Fit an expanded-train-only cross-fitted scalar and apply it once to saved fresh-group logits."""

from __future__ import annotations

import argparse
import json
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from scripts.egoviz_temperature_math import fit_alpha
from scripts.evaluate_egoviz_temperature import bootstrap
from scripts.evaluate_egoviz_train_oof_temperature import probabilities


def training_folds(groups: np.ndarray, *, seed: int = 17):
    """Return five deterministic fit/inner/held masks over distinct groups."""
    groups = np.asarray(groups)
    if groups.ndim != 1 or groups.size == 0 or any(str(value) == "" for value in groups):
        raise ValueError("groups must be a non-empty one-dimensional array of IDs")
    unique = np.unique(groups)
    if len(unique) < 5:
        raise ValueError("five non-empty group folds are required")
    partitions = np.array_split(np.random.default_rng(seed).permutation(unique), 5)
    if any(len(partition) == 0 for partition in partitions):
        raise ValueError("group partition is empty")
    result = []
    for index in range(5):
        held = np.isin(groups, partitions[index])
        inner = np.isin(groups, partitions[(index + 1) % 5])
        fit = ~(held | inner)
        if np.any(held & inner) or not fit.any() or not inner.any() or not held.any():
            raise ValueError("invalid group fold")
        result.append((fit, inner, held))
    return result


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _verify_metrics(actual: dict[str, Any], expected: dict[str, Any], *, tolerance: float = 1e-6):
    for key in ("accuracy", "nll", "brier", "ece", "mean_confidence", "count"):
        if key not in expected or abs(float(actual[key]) - float(expected[key])) > tolerance:
            raise ValueError(f"stored metric mismatch for {key}")


def run(corpus: Path, train_run: Path, fresh_eval: Path, output: Path) -> dict[str, Any]:
    import torch

    from scripts.train_egoviz_vs_sync_readout import (
        JointTemporalScorer,
        _fit,
        _load_samples,
        _tensor_data,
        metrics,
        sha256_file,
    )

    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    merge_path = corpus / "merge-report.json"
    merge = json.loads(merge_path.read_text(encoding="utf-8"))
    inputs = {
        "train-samples.jsonl": corpus / "train-samples.jsonl",
        "development-samples.jsonl": corpus / "development-samples.jsonl",
        "imu-windows.npy": corpus / "imu-windows.npy",
        "video-features.npz": corpus / "video-features.npz",
        "sensor-features.npz": corpus / "sensor-features.npz",
    }
    actual_hashes = {name: sha256_file(path) for name, path in inputs.items()}
    if actual_hashes != merge["sha256"]:
        raise ValueError("expanded training corpus hashes differ from merge report")
    train_summary = json.loads((train_run / "results.json").read_text(encoding="utf-8"))
    if train_summary["input_sha256"] != actual_hashes:
        raise ValueError("expanded training run did not use this exact corpus")
    fresh_summary_path = fresh_eval / "results.json"
    fresh_summary = json.loads(fresh_summary_path.read_text(encoding="utf-8"))
    fresh_path = fresh_eval / "predictions.jsonl"
    if sha256_file(fresh_path) != fresh_summary["predictions_sha256"]:
        raise ValueError("stored fresh-group prediction hash mismatch")
    if fresh_summary["sealed_audit_accessed"] is not False:
        raise ValueError("fresh evaluation does not attest to audit isolation")
    if fresh_summary["checkpoint_selected_from_fresh_development"] is not False:
        raise ValueError("fresh groups were used for checkpoint selection")
    checkpoint = train_run / "best.pt"
    checkpoint_hash = sha256_file(checkpoint)
    model_record = fresh_summary["models"]["linear_selected_step_616"]
    if checkpoint_hash != model_record["checkpoint_sha256"]:
        raise ValueError("training checkpoint does not match stored fresh-group predictions")
    if train_summary["best_step"] != 616:
        raise ValueError("training reference checkpoint is not the fixed step-616 checkpoint")

    rows, video, sensor = _load_samples(
        corpus, inputs["video-features.npz"], inputs["sensor-features.npz"]
    )
    train = _tensor_data(rows, video, sensor, "train")
    groups = train[3]
    targets = train[2].numpy().astype(np.int64)
    unique_groups = np.unique(groups)
    expected_train_groups = merge["old_train_sessions"] + merge["new_train_sessions"]
    if len(train[4]) != merge["train_examples"] or len(unique_groups) != expected_train_groups:
        raise ValueError("expanded train counts differ from merge receipt")
    folds = training_folds(groups, seed=17)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    started = time.perf_counter()
    oof = np.full((len(targets), 4), np.nan, dtype=np.float64)
    coverage = np.zeros(len(targets), dtype=np.int32)
    fold_records = []

    for fold_index, (fit_mask, inner_mask, held_mask) in enumerate(folds):
        random.seed(17)
        np.random.seed(17)
        torch.random.default_generator.manual_seed(17)
        model = JointTemporalScorer().cpu()
        fit_data = tuple(value[fit_mask] for value in train[:3])
        inner_data = tuple(value[inner_mask] for value in train[:3])
        result = _fit(model, fit_data, inner_data, joint=True)
        model.eval()
        with torch.inference_mode():
            logits = model(train[0][held_mask], train[1][held_mask]).numpy().astype(np.float64)
        oof[held_mask] = logits
        coverage[held_mask] += 1
        state_path = output / f"fold-{fold_index}-state.pt"
        torch.save(result["best_state"], state_path)
        fold_records.append(
            {
                "fold": fold_index,
                "fit_groups": sorted(np.unique(groups[fit_mask]).tolist()),
                "inner_validation_groups": sorted(np.unique(groups[inner_mask]).tolist()),
                "heldout_groups": sorted(np.unique(groups[held_mask]).tolist()),
                "fit_examples": int(fit_mask.sum()),
                "inner_examples": int(inner_mask.sum()),
                "heldout_examples": int(held_mask.sum()),
                "best_epoch": result["best_epoch"],
                "stopped_epoch": result["stopped_epoch"],
                "training_seconds": result["training_seconds"],
                "parameter_count": result["parameter_count"],
                "state_file": state_path.name,
                "state_sha256": sha256_file(state_path),
                "history": result["history"],
            }
        )

    if not np.isfinite(oof).all() or not np.all(coverage == 1):
        raise ValueError("training OOF coverage is incomplete")
    alpha = fit_alpha(oof, targets)
    # Fresh-group targets/logits are first read only after the scalar is frozen.
    fresh_rows = _read_jsonl(fresh_path)
    fresh_logits = np.asarray(
        [row["predictions"]["linear_selected_step_616"]["logits"] for row in fresh_rows],
        dtype=np.float64,
    )
    fresh_targets = np.asarray([row["target"] for row in fresh_rows], dtype=np.int64)
    fresh_groups = np.asarray([row["group_id"] for row in fresh_rows])
    fresh_ids = [row["sample_id"] for row in fresh_rows]
    if (
        fresh_logits.shape != (fresh_summary["input_count"], 4)
        or not np.isfinite(fresh_logits).all()
    ):
        raise ValueError("fresh-group logits are invalid")
    if (
        len(fresh_groups) != fresh_summary["input_count"]
        or len(np.unique(fresh_groups)) != fresh_summary["unique_group_count"]
    ):
        raise ValueError("fresh-group rows or group count differ from stored evaluation")
    if not np.array_equal(fresh_logits.argmax(axis=1), (alpha * fresh_logits).argmax(axis=1)):
        raise ValueError("positive temperature changed candidate ranking")
    base_metrics = metrics(fresh_logits, fresh_targets)
    _verify_metrics(base_metrics, fresh_summary["metrics"]["linear_selected_step_616"])
    scaled_logits = alpha * fresh_logits
    scaled_metrics = metrics(scaled_logits, fresh_targets)
    paired = {
        name: bootstrap(
            fresh_logits,
            scaled_logits,
            fresh_targets,
            fresh_groups,
            name,
            repetitions=2000,
        )
        for name in ("nll", "brier", "ece")
    }
    elapsed = time.perf_counter() - started
    report = {
        "status": "POSTHOC_TRAIN_ONLY_CALIBRATION_TRANSFER_ON_EXPOSED_FRESH_GROUP_DEV",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "runner_sha256": sha256_file(Path(__file__)),
        "expanded_merge_report_sha256": sha256_file(merge_path),
        "expanded_input_sha256": actual_hashes,
        "expanded_train_examples": len(targets),
        "expanded_train_groups": len(unique_groups),
        "train_run_results_sha256": sha256_file(train_run / "results.json"),
        "fixed_checkpoint_sha256": checkpoint_hash,
        "fresh_evaluation_results_sha256": sha256_file(fresh_summary_path),
        "fresh_prediction_sha256": sha256_file(fresh_path),
        "fresh_groups": int(len(np.unique(fresh_groups))),
        "fresh_examples": len(fresh_targets),
        "fold_count": len(fold_records),
        "seed": 17,
        "train_oof_inverse_temperature": alpha,
        "temperature": 1.0 / alpha,
        "train_oof_metrics": metrics(oof, targets),
        "train_oof_scaled_in_sample_for_scalar": metrics(alpha * oof, targets),
        "fresh_baseline_metrics": base_metrics,
        "fresh_scaled_metrics": scaled_metrics,
        "paired_group_bootstrap_raw_minus_scaled": paired,
        "bootstrap_method": (
            "2,000 paired group resamples with replacement; seed 17; scalar held fixed"
        ),
        "folds": fold_records,
        "training_seconds": sum(row["training_seconds"] for row in fold_records),
        "elapsed_seconds": elapsed,
        "device": "cpu",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "numpy": np.__version__,
        "fresh_labels_used_for_calibration_or_selection": False,
        "fresh_set_previously_reported_unscaled": True,
        "checkpoint_selected_from_fresh_set": False,
        "sealed_audit_accessed": False,
        "model_weights_changed": False,
    }
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (output / "training-oof.jsonl").open("w", encoding="utf-8") as stream:
        for index, sample_id in enumerate(train[4]):
            stream.write(
                json.dumps(
                    {
                        "sample_id": sample_id,
                        "group_id": str(groups[index]),
                        "target": int(targets[index]),
                        "logits": oof[index].tolist(),
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
    with (output / "fresh-predictions.jsonl").open("w", encoding="utf-8") as stream:
        baseline_probs = probabilities(fresh_logits)
        scaled_probs = probabilities(scaled_logits)
        for index, sample_id in enumerate(fresh_ids):
            stream.write(
                json.dumps(
                    {
                        "sample_id": sample_id,
                        "group_id": str(fresh_groups[index]),
                        "target": int(fresh_targets[index]),
                        "baseline_probabilities": baseline_probs[index].tolist(),
                        "scaled_probabilities": scaled_probs[index].tolist(),
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.egoviz_expanded_oof_calibration",
            "--verify-fold-reloads",
            "--corpus",
            str(corpus),
            "--train-run",
            str(train_run),
            "--fresh-eval",
            str(fresh_eval),
            "--output",
            str(output),
        ],
        check=True,
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
    )
    receipt_path = output / "fold-reload-receipt.json"
    report["fold_reload_receipt_sha256"] = sha256_file(receipt_path)
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def verify_fold_reloads(corpus: Path, train_run: Path, output: Path) -> dict[str, Any]:
    """Reload each fold state in a fresh process and replay its held-out logits."""
    import os

    import torch

    from scripts.train_egoviz_vs_sync_readout import (
        JointTemporalScorer,
        _load_samples,
        _tensor_data,
        sha256_file,
    )

    merge = json.loads((corpus / "merge-report.json").read_text(encoding="utf-8"))
    inputs = {
        "train-samples.jsonl": corpus / "train-samples.jsonl",
        "development-samples.jsonl": corpus / "development-samples.jsonl",
        "imu-windows.npy": corpus / "imu-windows.npy",
        "video-features.npz": corpus / "video-features.npz",
        "sensor-features.npz": corpus / "sensor-features.npz",
    }
    if {name: sha256_file(path) for name, path in inputs.items()} != merge["sha256"]:
        raise ValueError("reload process found changed corpus inputs")
    rows, video, sensor = _load_samples(
        corpus, inputs["video-features.npz"], inputs["sensor-features.npz"]
    )
    train = _tensor_data(rows, video, sensor, "train")
    groups, targets = train[3], train[2].numpy().astype(np.int64)
    folds = training_folds(groups, seed=17)
    saved_oof = _read_jsonl(output / "training-oof.jsonl")
    if [row["sample_id"] for row in saved_oof] != train[4]:
        raise ValueError("reload process found changed OOF sample order")
    if [row["target"] for row in saved_oof] != targets.tolist():
        raise ValueError("reload process found changed OOF targets")
    if [row["group_id"] for row in saved_oof] != groups.astype(str).tolist():
        raise ValueError("reload process found changed OOF groups")
    expected_logits = np.asarray([row["logits"] for row in saved_oof], dtype=np.float64)
    records = []
    for fold_index, (_, _, held_mask) in enumerate(folds):
        state_path = output / f"fold-{fold_index}-state.pt"
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        model = JointTemporalScorer().cpu().eval()
        model.load_state_dict(state, strict=True)
        with torch.inference_mode():
            actual = model(train[0][held_mask], train[1][held_mask]).numpy()
        max_abs_error = float(np.max(np.abs(actual - expected_logits[held_mask])))
        if max_abs_error > 1e-7:
            raise ValueError(f"fold {fold_index} reload logits differ: {max_abs_error}")
        records.append(
            {
                "fold": fold_index,
                "state_sha256": sha256_file(state_path),
                "held_examples": int(held_mask.sum()),
                "max_abs_logit_error": max_abs_error,
                "strict_state_load": True,
            }
        )
    receipt = {
        "status": "ALL_FOLD_STATES_RELOADED_AND_HELDOUT_LOGITS_REPLAYED",
        "process_id": os.getpid(),
        "train_results_sha256": sha256_file(train_run / "results.json"),
        "corpus_train_sha256": merge["sha256"]["train-samples.jsonl"],
        "oof_predictions_sha256": sha256_file(output / "training-oof.jsonl"),
        "folds": records,
        "sealed_audit_accessed": False,
    }
    receipt_path = output / "fold-reload-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--train-run", type=Path, required=True)
    parser.add_argument("--fresh-eval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-fold-reloads", action="store_true")
    args = parser.parse_args()
    if args.verify_fold_reloads:
        result = verify_fold_reloads(args.corpus, args.train_run, args.output)
    else:
        result = run(args.corpus, args.train_run, args.fresh_eval, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
