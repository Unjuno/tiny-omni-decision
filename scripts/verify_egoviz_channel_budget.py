"""Reload best and final checkpoints and compare fixed development predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.evaluate_egoviz_temperature import bootstrap


def read_predictions(path: Path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate sample IDs")
    return rows


def check_predictions(rows, sample_ids, groups, targets):
    for key, expected in (("sample_id", sample_ids), ("group_id", groups), ("target", targets)):
        if [row[key] for row in rows] != list(expected):
            raise ValueError(f"saved {key} differs from hashed corpus")
    logits = np.asarray([row["logits"] for row in rows], dtype=np.float64)
    if logits.shape != (len(rows), 4) or not np.isfinite(logits).all():
        raise ValueError("invalid or nonfinite saved logits")
    return logits


def assert_logits_match(actual, expected):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if (
        actual.shape != expected.shape
        or not np.isfinite(actual).all()
        or not np.isfinite(expected).all()
    ):
        raise ValueError("invalid shape or nonfinite reload logits")
    delta = float(np.max(np.abs(actual - expected)))
    if delta > 1e-6:
        raise ValueError("reloaded logits differ")
    return delta


def compare_predictions(baseline, candidate):
    for key in ("sample_id", "group_id", "target"):
        if [row[key] for row in baseline] != [row[key] for row in candidate]:
            raise ValueError(f"development {key} mismatch")
    base = np.asarray([row["logits"] for row in baseline])
    trial = np.asarray([row["logits"] for row in candidate])
    targets = np.asarray([row["target"] for row in baseline])
    groups = np.asarray([row["group_id"] for row in baseline])
    if len(np.unique(groups)) != 20:
        raise ValueError("expected 20 development sessions")
    accuracy = bootstrap(trial, base, targets, groups, "accuracy")
    accuracy["candidate_minus_baseline"] = accuracy.pop("baseline_minus_candidate")
    return {
        "candidate_minus_baseline_accuracy": accuracy,
        "baseline_minus_candidate_nll": bootstrap(base, trial, targets, groups, "nll"),
        "baseline_minus_candidate_brier": bootstrap(base, trial, targets, groups, "brier"),
    }


def verify(corpus: Path, run_dir: Path, output: Path):
    import torch

    from scripts.egoviz_channel_scorer import ChannelWeightedScorer
    from scripts.train_egoviz_vs_sync_readout import (
        JointTemporalScorer,
        _load_samples,
        _tensor_data,
        sha256_file,
    )

    if output.exists():
        raise FileExistsError(output)
    report_path = run_dir / "results.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    if status["status"] != "COMPLETE" or report["sealed_audit_accessed"] is not False:
        raise ValueError("run incomplete or audit isolation not attested")
    files = {
        "train_samples": corpus / "train-samples.jsonl",
        "development_samples": corpus / "development-samples.jsonl",
        "video_features": corpus / "video-features.npz",
        "sensor_features": corpus / "sensor-features.npz",
    }
    if {key: sha256_file(path) for key, path in files.items()} != report["input_sha256"]:
        raise ValueError("input hashes differ")
    if (
        report["results"]["baseline"]["sample_order_sha256"]
        != report["results"]["channel_weighted"]["sample_order_sha256"]
    ):
        raise ValueError("sample orders differ")
    torch.set_num_threads(4)
    rows, video, sensor = _load_samples(corpus, files["video_features"], files["sensor_features"])
    dev = _tensor_data(rows, video, sensor, "development")
    reloads = {}
    for name, factory in (
        ("baseline", JointTemporalScorer),
        ("channel_weighted", ChannelWeightedScorer),
    ):
        reloads[name] = {}
        checkpoints = [("best", "best"), ("final", "epoch-512")]
        checkpoints.extend(
            (f"epoch-{epoch}", f"epoch-{epoch}") for epoch in (32, 64, 128, 256, 384)
        )
        for label, file_label in checkpoints:
            state = run_dir / f"{name}-{file_label}.pt"
            recorded_hash = report["results"][name].get(f"{label}_checkpoint_sha256")
            if recorded_hash is not None and sha256_file(state) != recorded_hash:
                raise ValueError(f"{name} {label} checkpoint hash mismatch")
            saved = read_predictions(run_dir / f"{name}-{file_label}-predictions.jsonl")
            expected = check_predictions(saved, dev[4], dev[3].tolist(), dev[2].tolist())
            model = factory().cpu().eval()
            model.load_state_dict(torch.load(state, map_location="cpu", weights_only=True))
            with torch.inference_mode():
                logits = model(dev[0], dev[1]).numpy()
            delta = assert_logits_match(logits, expected)
            reloads[name][label] = {
                "max_logit_difference": delta,
                "checkpoint_sha256": sha256_file(state),
            }
    comparisons = {}
    for label in ("epoch-32", "epoch-128", "epoch-256", "epoch-512", "best"):
        baseline = read_predictions(run_dir / f"baseline-{label}-predictions.jsonl")
        candidate = read_predictions(run_dir / f"channel_weighted-{label}-predictions.jsonl")
        comparisons[label] = compare_predictions(baseline, candidate)
    receipt = {
        "status": "PASS_RELOAD_AND_PAIRED_COMPARISON",
        "reloads": reloads,
        "comparisons": comparisons,
        "run_results_sha256": sha256_file(report_path),
        "runner_sha256": sha256_file(Path(__file__)),
        "sealed_audit_accessed": False,
        "best_checkpoint_intervals_account_for_selection": False,
        "session_bootstrap_resamples": 2000,
        "session_bootstrap_seed": 17,
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.corpus, args.run_dir, args.output), indent=2))


if __name__ == "__main__":
    main()
