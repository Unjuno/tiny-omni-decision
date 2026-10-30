"""Fit a scalar from training-session out-of-fold scorer predictions."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
from pathlib import Path

import numpy as np

from scripts.egoviz_temperature_math import fit_alpha
from scripts.egoviz_train_oof_folds import training_folds
from scripts.evaluate_egoviz_temperature import bootstrap


def probabilities(logits):
    values = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return values / values.sum(axis=1, keepdims=True)


def run(corpus: Path, selected: Path, output: Path):
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
        raise FileExistsError(output)
    base_report_path = selected / "readout-results.json"
    base = json.loads(base_report_path.read_text(encoding="utf-8"))
    inputs = {"train_samples": corpus / "train-samples.jsonl",
              "development_samples": corpus / "development-samples.jsonl",
              "video_features": corpus / "video-features.npz",
              "sensor_features": corpus / "sensor-features.npz"}
    if base["sealed_audit_accessed"] is not False:
        raise ValueError("baseline does not attest to audit isolation")
    if {key: sha256_file(path) for key, path in inputs.items()} != base["input_sha256"]:
        raise ValueError("input hash mismatch")
    checkpoint = selected / "joint-temporal-state.pt"
    if sha256_file(checkpoint) != base["joint_temporal"]["state_sha256"]:
        raise ValueError("checkpoint hash mismatch")
    rows, video, sensor = _load_samples(corpus, inputs["video_features"], inputs["sensor_features"])
    train = _tensor_data(rows, video, sensor, "train")
    groups, targets = train[3], train[2].numpy()
    folds = training_folds(groups)
    oof = np.full((len(targets), 4), np.nan, dtype=np.float64)
    covered = np.zeros(len(targets), dtype=np.int32)
    output.mkdir(parents=True)
    fold_results = []
    for index, (fit_mask, inner_mask, held_mask) in enumerate(folds):
        random.seed(17)
        np.random.seed(17)
        torch.manual_seed(17)
        model = JointTemporalScorer()
        fit_data = tuple(value[fit_mask] for value in train[:3])
        inner_data = tuple(value[inner_mask] for value in train[:3])
        result = _fit(model, fit_data, inner_data, joint=True)
        model.eval()
        with torch.inference_mode():
            oof[held_mask] = model(train[0][held_mask], train[1][held_mask]).numpy()
        covered[held_mask] += 1
        state_path = output / f"fold-{index}-state.pt"
        torch.save(result["best_state"], state_path)
        fold_results.append({
            "fold": index, "seed": 17,
            "fit_groups": sorted(np.unique(groups[fit_mask]).tolist()),
            "inner_validation_groups": sorted(np.unique(groups[inner_mask]).tolist()),
            "heldout_groups": sorted(np.unique(groups[held_mask]).tolist()),
            "fit_examples": int(fit_mask.sum()), "inner_examples": int(inner_mask.sum()),
            "heldout_examples": int(held_mask.sum()), "best_epoch": result["best_epoch"],
            "stopped_epoch": result["stopped_epoch"], "history": result["history"],
            "training_seconds": result["training_seconds"],
            "state_file": state_path.name, "state_sha256": sha256_file(state_path),
        })
    if not np.isfinite(oof).all() or not np.all(covered == 1):
        raise ValueError("training out-of-fold coverage is incomplete")
    alpha = fit_alpha(oof, targets)
    # The development split is loaded only after all training-only fold predictions
    # and the scalar have been fixed.
    dev = _tensor_data(rows, video, sensor, "development")
    if set(groups) & set(dev[3]):
        raise ValueError("train/development group overlap")
    original = JointTemporalScorer().cpu().eval()
    original.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    with torch.inference_mode():
        dev_logits = original(dev[0], dev[1]).numpy().astype(np.float64)
    dev_y = dev[2].numpy()
    prior_predictions = [json.loads(line) for line in
                         (selected / "development-predictions.jsonl").read_text().splitlines()]
    if ([row["sample_id"] for row in prior_predictions] != dev[4]
            or [row["target"] for row in prior_predictions] != dev_y.tolist()):
        raise ValueError("development IDs/targets differ")
    parity = float(np.max(np.abs(probabilities(dev_logits) - np.asarray(
        [row["joint_probabilities"] for row in prior_predictions]))))
    if parity > 1e-6:
        raise ValueError("baseline probability parity failed")
    calibrated = alpha * dev_logits
    if not np.array_equal(calibrated.argmax(axis=1), dev_logits.argmax(axis=1)):
        raise ValueError("ranking changed")
    if sha256_file(checkpoint) != base["joint_temporal"]["state_sha256"]:
        raise ValueError("original checkpoint changed")
    report = {
        "status": "EXPLORATORY_TRAIN_GROUP_OOF_TEMPERATURE_NOT_BLIND",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "runner_sha256": sha256_file(Path(__file__)), "input_sha256": base["input_sha256"],
        "checkpoint_sha256": base["joint_temporal"]["state_sha256"],
        "train_oof_inverse_temperature": alpha, "temperature": 1 / alpha,
        "train_oof_baseline": metrics(oof, targets),
        "train_oof_scaled_in_sample_for_scalar": metrics(alpha * oof, targets),
        "development_baseline": metrics(dev_logits, dev_y),
        "development_scaled": metrics(calibrated, dev_y),
        "paired_group_bootstrap": {
            metric: bootstrap(dev_logits, calibrated, dev_y, dev[3], metric)
            for metric in ("nll", "brier")},
        "folds": fold_results, "probability_parity_max_abs": parity,
        "training_oof_examples": len(targets), "development_examples": len(dev_y),
        "calibration_uses_development_labels": False,
        "checkpoint_selected_on_development": True, "sealed_audit_accessed": False,
        "device": "cpu", "torch": torch.__version__, "numpy": np.__version__,
    }
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (output / "training-oof.jsonl").open("w", encoding="utf-8") as stream:
        for index, sample_id in enumerate(train[4]):
            row = {"sample_id": sample_id, "group_id": str(groups[index]),
                   "target": int(targets[index]), "logits": oof[index].tolist()}
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    with (output / "development-predictions.jsonl").open("w", encoding="utf-8") as stream:
        original_probs, scaled_probs = probabilities(dev_logits), probabilities(calibrated)
        for index, sample_id in enumerate(dev[4]):
            row = {"sample_id": sample_id, "group_id": str(dev[3][index]),
                   "target": int(dev_y[index]), "baseline": original_probs[index].tolist(),
                   "scaled": scaled_probs[index].tolist()}
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.corpus, args.selected, args.output), indent=2))


if __name__ == "__main__":
    main()
