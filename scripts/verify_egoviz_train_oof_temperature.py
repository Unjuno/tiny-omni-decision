"""Reload all cross-fit weights and verify training and development predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.egoviz_temperature_math import fit_alpha
from scripts.evaluate_egoviz_train_oof_temperature import probabilities


def verify(corpus: Path, selected: Path, result_dir: Path, output: Path):
    import torch

    from scripts.train_egoviz_vs_sync_readout import (
        JointTemporalScorer,
        _load_samples,
        _tensor_data,
        sha256_file,
    )

    if output.exists():
        raise FileExistsError(output)
    report = json.loads((result_dir / "results.json").read_text(encoding="utf-8"))
    files = {"train_samples": corpus / "train-samples.jsonl",
             "development_samples": corpus / "development-samples.jsonl",
             "video_features": corpus / "video-features.npz",
             "sensor_features": corpus / "sensor-features.npz"}
    if {key: sha256_file(value) for key, value in files.items()} != report["input_sha256"]:
        raise ValueError("input hash mismatch")
    original_path = selected / "joint-temporal-state.pt"
    if sha256_file(original_path) != report["checkpoint_sha256"]:
        raise ValueError("original checkpoint hash mismatch")
    rows, video, sensor = _load_samples(corpus, files["video_features"], files["sensor_features"])
    train = _tensor_data(rows, video, sensor, "train")
    dev = _tensor_data(rows, video, sensor, "development")
    training_saved = [json.loads(line) for line in
                      (result_dir / "training-oof.jsonl").read_text().splitlines()]
    development_saved = [json.loads(line) for line in
                         (result_dir / "development-predictions.jsonl").read_text().splitlines()]
    if ([row["sample_id"] for row in training_saved] != train[4]
            or [row["target"] for row in training_saved] != train[2].tolist()
            or [row["sample_id"] for row in development_saved] != dev[4]
            or [row["target"] for row in development_saved] != dev[2].tolist()):
        raise ValueError("saved IDs or targets differ")
    groups = train[3]
    logits = np.full((len(groups), 4), np.nan)
    coverage = np.zeros(len(groups), dtype=int)
    for fold in report["folds"]:
        fit, inner, held = (set(fold[name]) for name in
                            ("fit_groups", "inner_validation_groups", "heldout_groups"))
        if fit & inner or fit & held or inner & held or fit | inner | held != set(groups):
            raise ValueError("fold groups overlap or omit a session")
        state = result_dir / fold["state_file"]
        if sha256_file(state) != fold["state_sha256"]:
            raise ValueError("fold checkpoint hash mismatch")
        model = JointTemporalScorer().eval()
        model.load_state_dict(torch.load(state, map_location="cpu", weights_only=True))
        mask = np.isin(groups, list(held))
        with torch.inference_mode():
            logits[mask] = model(train[0][mask], train[1][mask]).numpy()
        coverage[mask] += 1
    if not np.all(coverage == 1):
        raise ValueError("training groups are not held out exactly once")
    expected = np.asarray([row["logits"] for row in training_saved])
    max_oof_delta = float(np.max(np.abs(logits - expected)))
    alpha = fit_alpha(logits, train[2].numpy())
    if max_oof_delta > 1e-6 or abs(alpha - report["train_oof_inverse_temperature"]) > 1e-6:
        raise ValueError("reloaded out-of-fold scores or scalar differ")
    model = JointTemporalScorer().eval()
    model.load_state_dict(torch.load(original_path, map_location="cpu", weights_only=True))
    with torch.inference_mode():
        dev_logits = model(dev[0], dev[1]).numpy().astype(np.float64)
    expected_dev = np.asarray([row["scaled"] for row in development_saved])
    max_dev_delta = float(np.max(np.abs(probabilities(alpha * dev_logits) - expected_dev)))
    if max_dev_delta > 1e-6:
        raise ValueError("reloaded development probabilities differ")
    receipt = {"status": "PASS_FRESH_PROCESS_FIVE_FOLD_AND_SCALAR_RELOAD",
               "training_oof_count": len(groups), "development_count": len(dev[4]),
               "fold_count": len(report["folds"]), "max_oof_logit_difference": max_oof_delta,
               "alpha_difference": abs(alpha - report["train_oof_inverse_temperature"]),
               "max_development_probability_difference": max_dev_delta,
               "results_sha256": sha256_file(result_dir / "results.json"),
               "runner_sha256": sha256_file(Path(__file__)),
               "training": False, "sealed_audit_accessed": False}
    with output.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.corpus, args.selected, args.result_dir, args.output), indent=2))


if __name__ == "__main__":
    main()
