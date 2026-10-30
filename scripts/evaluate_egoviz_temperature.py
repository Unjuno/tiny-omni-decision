"""Evaluate a train-fitted scalar and grouped diagnostic on frozen EgoViz scores."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np

from scripts.egoviz_temperature_math import cross_fit, fit_alpha


def bootstrap(base, candidate, labels, groups, metric, repetitions=2000):
    from scripts.train_egoviz_vs_sync_readout import metrics

    rng = np.random.default_rng(17)
    unique = np.unique(groups)
    values = []
    for _ in range(repetitions):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(groups == group) for group in sampled])
        values.append(metrics(base[indices], labels[indices])[metric]
                      - metrics(candidate[indices], labels[indices])[metric])
    return {"baseline_minus_candidate": metrics(base, labels)[metric]
            - metrics(candidate, labels)[metric],
            "percentile_95_ci": np.quantile(values, [0.025, 0.975]).tolist(),
            "unit": "development session", "seed": 17, "repetitions": repetitions}


def run(corpus: Path, selected: Path, output: Path):
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
    report_path = selected / "readout-results.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    inputs = {
        "train_samples": corpus / "train-samples.jsonl",
        "development_samples": corpus / "development-samples.jsonl",
        "video_features": corpus / "video-features.npz",
        "sensor_features": corpus / "sensor-features.npz",
    }
    if report["sealed_audit_accessed"] is not False:
        raise ValueError("baseline does not attest to audit isolation")
    if {key: sha256_file(path) for key, path in inputs.items()} != report["input_sha256"]:
        raise ValueError("frozen input hash mismatch")
    checkpoint = selected / "joint-temporal-state.pt"
    checkpoint_hash = sha256_file(checkpoint)
    if checkpoint_hash != report["joint_temporal"]["state_sha256"]:
        raise ValueError("checkpoint hash mismatch")
    rows, video, sensor = _load_samples(corpus, inputs["video_features"], inputs["sensor_features"])
    train = _tensor_data(rows, video, sensor, "train")
    development = _tensor_data(rows, video, sensor, "development")
    if set(train[3]) & set(development[3]):
        raise ValueError("train/development group overlap")
    model = JointTemporalScorer().cpu().eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    with torch.inference_mode():
        train_logits = model(train[0], train[1]).numpy().astype(np.float64)
        dev_logits = model(development[0], development[1]).numpy().astype(np.float64)
    train_y = train[2].numpy()
    dev_y = development[2].numpy()
    saved = [json.loads(line) for line in
             (selected / "development-predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    if ([row["sample_id"] for row in saved] != development[4]
            or [row["target"] for row in saved] != dev_y.tolist()):
        raise ValueError("saved development identity mismatch")
    def softmax(values):
        weights = np.exp(values - values.max(axis=1, keepdims=True))
        return weights / weights.sum(axis=1, keepdims=True)

    parity = float(np.max(np.abs(softmax(dev_logits)
                                 - np.array([row["joint_probabilities"] for row in saved]))))
    if parity > 1e-6:
        raise ValueError("baseline probabilities differ from saved predictions")
    train_nll = metrics(train_logits, train_y)["nll"]
    if abs(train_nll - report["joint_temporal"]["train_metrics"]["nll"]) > 1e-6:
        raise ValueError("baseline train metrics differ")
    alpha = fit_alpha(train_logits, train_y)
    calibrated = alpha * dev_logits
    oof, folds = cross_fit(dev_logits, dev_y, development[3])
    for logits in (calibrated, oof):
        if not np.array_equal(logits.argmax(axis=1), dev_logits.argmax(axis=1)):
            raise ValueError("positive scaling changed candidate ranking")
    if sha256_file(checkpoint) != checkpoint_hash:
        raise ValueError("checkpoint changed during evaluation")
    outcomes = {"baseline": metrics(dev_logits, dev_y),
                "train_only_scalar": metrics(calibrated, dev_y),
                "development_group_oof_diagnostic": metrics(oof, dev_y)}
    confidence_intervals = {
        name: {metric: bootstrap(dev_logits, logits, dev_y, development[3], metric)
               for metric in ("nll", "brier")}
        for name, logits in (("train_only_scalar", calibrated), ("development_oof", oof))
    }
    result = {
        "status": "EXPLORATORY_DEVELOPMENT_CALIBRATION_NOT_BLIND",
        "train_only_alpha": alpha, "temperature": 1 / alpha,
        "train_metrics_baseline": metrics(train_logits, train_y),
        "train_metrics_scaled": metrics(alpha * train_logits, train_y),
        "development_metrics": outcomes, "folds": folds,
        "paired_group_bootstrap": confidence_intervals,
        "saved_probability_parity_max_abs": parity,
        "input_sha256": report["input_sha256"], "checkpoint_sha256": checkpoint_hash,
        "baseline_result_sha256": sha256_file(report_path),
        "runner_sha256": sha256_file(Path(__file__)),
        "python": platform.python_version(), "torch": torch.__version__,
        "numpy": np.__version__, "device": "cpu", "sealed_audit_accessed": False,
        "checkpoint_selected_on_development": True,
        "oof_uses_development_labels": True,
    }
    output.mkdir(parents=True)
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for index, sample_id in enumerate(development[4]):
            row = {"sample_id": sample_id, "group_id": str(development[3][index]),
                   "target": int(dev_y[index]),
                   "baseline": softmax(dev_logits[index:index+1])[0].tolist(),
                   "train_only_scalar": softmax(calibrated[index:index+1])[0].tolist(),
                   "development_oof": softmax(oof[index:index+1])[0].tolist()}
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.corpus, args.selected, args.output), indent=2))


if __name__ == "__main__":
    main()
