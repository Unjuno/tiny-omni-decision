"""Evaluate a frozen readout and scalar on the previously observed pilot groups."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from scripts.egoviz_transfer_integrity import validate_isolation
from scripts.evaluate_egoviz_temperature import bootstrap
from scripts.evaluate_egoviz_train_oof_temperature import probabilities

CALIBRATION_SHA256 = "179cae04c90acf59a2dac0146538b2d8f48a69907136a650b75c773361a8ede8"


def run(pilot: Path, current: Path, selected: Path, calibration_path: Path, output: Path):
    import torch

    from scripts.train_egoviz_vs_sync_readout import (
        JointTemporalScorer,
        _load_samples,
        _tensor_data,
        metrics,
        sha256_file,
    )

    if output.exists():
        raise FileExistsError(output)
    old_selected = pilot / "readout-results-cpu-03"
    old_report_path = old_selected / "readout-results.json"
    old_report = json.loads(old_report_path.read_text(encoding="utf-8"))
    new_report_path = selected / "readout-results.json"
    new_report = json.loads(new_report_path.read_text(encoding="utf-8"))
    if sha256_file(calibration_path) != CALIBRATION_SHA256:
        raise ValueError("calibration metadata hash mismatch")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if (old_report["sealed_audit_accessed"] is not False
            or new_report["sealed_audit_accessed"] is not False
            or calibration["sealed_audit_accessed"] is not False):
        raise ValueError("sealed audit isolation not attested")
    pilot_corpus = pilot / "window-corpus-v2"
    pilot_inputs = {"train_samples": pilot_corpus / "train-samples.jsonl",
                    "development_samples": pilot_corpus / "development-samples.jsonl",
                    "video_features": pilot / "videoprism-features-cpu-01/video-features.npz",
                    "sensor_features": pilot / "moment-features-gpu-01/sensor-features.npz"}
    current_inputs = {"train_samples": current / "train-samples.jsonl",
                      "development_samples": current / "development-samples.jsonl",
                      "video_features": current / "video-features.npz",
                      "sensor_features": current / "sensor-features.npz"}
    pilot_hashes = {name: sha256_file(path) for name, path in pilot_inputs.items()}
    if pilot_hashes != old_report["input_sha256"]:
        raise ValueError("pilot input hash mismatch")
    current_hashes = {name: sha256_file(path) for name, path in current_inputs.items()}
    if current_hashes != new_report["input_sha256"]:
        raise ValueError("new model input hash mismatch")
    if calibration["input_sha256"] != new_report["input_sha256"]:
        raise ValueError("scalar was fitted on a different corpus")
    old_weights = old_selected / "joint-temporal-state.pt"
    new_weights = selected / "joint-temporal-state.pt"
    if (sha256_file(old_weights) != old_report["joint_temporal"]["state_sha256"]
            or sha256_file(new_weights) != new_report["joint_temporal"]["state_sha256"]
            or calibration["checkpoint_sha256"] != sha256_file(new_weights)):
        raise ValueError("checkpoint hash mismatch")
    sensor_meta = json.loads((pilot / "moment-features-gpu-01/feature-report.json").read_text())
    video_meta = json.loads((pilot / "videoprism-features-cpu-01/feature-report.json").read_text())
    aggregate = json.loads((current / "aggregate-report.json").read_text())
    if (sensor_meta["checkpoint_sha256"] != aggregate["sensor_checkpoint_sha256"]
            or sensor_meta["moment_source_commit"] != aggregate["sensor_source_commit"]
            or video_meta["source_checkpoint_sha256"] != aggregate["video_checkpoint_sha256"]
            or video_meta["upstream_commit"] != aggregate["video_source_commit"]):
        raise ValueError("encoder checkpoint or source revision mismatch")
    pilot_rows, video, sensor = _load_samples(
        pilot_corpus, pilot_inputs["video_features"], pilot_inputs["sensor_features"])
    current_rows = [json.loads(line) for split in ("train", "development")
                    for line in current_inputs[f"{split}_samples"].read_text().splitlines()]
    isolation = validate_isolation(pilot_rows, current_rows)
    alpha = calibration["train_oof_inverse_temperature"]
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("invalid calibration scalar")
    new_model = JointTemporalScorer().cpu().eval()
    old_model = JointTemporalScorer().cpu().eval()
    new_model.load_state_dict(torch.load(new_weights, map_location="cpu", weights_only=True))
    old_model.load_state_dict(torch.load(old_weights, map_location="cpu", weights_only=True))
    output_rows = []
    results = {}
    for split in ("train", "development"):
        data = _tensor_data(pilot_rows, video, sensor, split)
        with torch.inference_mode():
            new_logits = new_model(data[0], data[1]).numpy().astype(np.float64)
        scaled = alpha * new_logits
        if not np.array_equal(new_logits.argmax(axis=1), scaled.argmax(axis=1)):
            raise ValueError("scaling changed ranking")
        labels = data[2].numpy()
        block = {"new_model_unscaled": metrics(new_logits, labels),
                 "new_model_scaled": metrics(scaled, labels),
                 "scaling_paired_bootstrap": {
                     metric: bootstrap(new_logits, scaled, labels, data[3], metric)
                     for metric in ("nll", "brier")}}
        for interval in block["scaling_paired_bootstrap"].values():
            interval["unit"] = f"pilot {split} session"
        old_logits = None
        if split == "development":
            with torch.inference_mode():
                old_logits = old_model(data[0], data[1]).numpy().astype(np.float64)
            old_metrics = metrics(old_logits, labels)
            if any(abs(old_metrics[key] - old_report["joint_temporal"]["development_metrics"][key])
                   > 1e-6 for key in ("accuracy", "nll", "brier", "ece")):
                raise ValueError("prior model development metrics do not reproduce")
            block["prior_selected_model"] = old_metrics
            block["new_minus_prior_paired_bootstrap"] = {
                metric: bootstrap(new_logits, old_logits, labels, data[3], metric)
                for metric in ("accuracy", "nll")}
        results[split] = block
        unscaled_probs = probabilities(new_logits)
        scaled_probs = probabilities(scaled)
        old_probs = probabilities(old_logits) if old_logits is not None else None
        for index, sample_id in enumerate(data[4]):
            row = {"split": split, "sample_id": sample_id, "group_id": str(data[3][index]),
                   "target": int(labels[index]), "new_logits": new_logits[index].tolist(),
                   "new_unscaled_probabilities": unscaled_probs[index].tolist(),
                   "new_scaled_probabilities": scaled_probs[index].tolist()}
            if old_probs is not None:
                row["prior_model_probabilities"] = old_probs[index].tolist()
            output_rows.append(row)
    if (sha256_file(new_weights) != calibration["checkpoint_sha256"]
            or sha256_file(old_weights) != old_report["joint_temporal"]["state_sha256"]):
        raise ValueError("weights changed during transfer")
    report = {
        "status": "OBSERVED_PILOT_UNSEEN_SESSION_TRANSFER_NOT_BLIND",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "runner_sha256": sha256_file(Path(__file__)), "isolation": isolation,
        "pilot_inputs_sha256": old_report["input_sha256"],
        "current_inputs_sha256": new_report["input_sha256"],
        "pilot_report_sha256": sha256_file(old_report_path),
        "new_report_sha256": sha256_file(new_report_path),
        "calibration_sha256": sha256_file(calibration_path),
        "new_checkpoint_sha256": sha256_file(new_weights),
        "prior_checkpoint_sha256": sha256_file(old_weights),
        "inverse_temperature": alpha, "metrics": results,
        "pilot_moment_backend": sensor_meta["device"], "current_moment_backend": "cpu",
        "calibration_fitted_during_transfer": False, "weights_trained_during_transfer": False,
        "sealed_audit_accessed": False, "device": "cpu",
        "torch": torch.__version__, "numpy": np.__version__,
    }
    output.mkdir(parents=True)
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output / "predictions.jsonl").write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in output_rows),
        encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.pilot, args.current, args.selected, args.calibration, args.output),
                     indent=2))


if __name__ == "__main__":
    main()
