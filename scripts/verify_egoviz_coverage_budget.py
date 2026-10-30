"""Reload fixed coverage checkpoints and compare with the original baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.verify_egoviz_channel_budget import (
    assert_logits_match,
    check_predictions,
    compare_predictions,
    read_predictions,
)


def check_probabilities(rows, logits):
    probabilities = np.asarray([row["probabilities"] for row in rows], dtype=np.float64)
    if probabilities.shape != logits.shape or not np.isfinite(probabilities).all():
        raise ValueError("invalid saved probabilities")
    shifted = logits - logits.max(axis=1, keepdims=True)
    expected = np.exp(shifted)
    expected /= expected.sum(axis=1, keepdims=True)
    if not np.allclose(probabilities, expected, atol=1e-6, rtol=0):
        raise ValueError("saved probabilities do not match logits")


def verify(corpus: Path, run_dir: Path, baseline_dir: Path, output: Path):
    import torch

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
    baseline_path = baseline_dir / "results.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if (
        report["status"] != "FIXED_COVERAGE_BUDGET_COMPLETE_EXPLORATORY"
        or baseline["status"] != "FIXED_BUDGET_COMPARISON_COMPLETE_EXPLORATORY"
        or report["sealed_audit_accessed"] is not False
        or baseline["sealed_audit_accessed"] is not False
    ):
        raise ValueError("incomplete run or audit isolation not attested")
    for name, expected in report["input_sha256"].items():
        if sha256_file(corpus / name) != expected:
            raise ValueError(f"merged corpus hash mismatch: {name}")
    if (
        report["input_sha256"]["development-samples.jsonl"]
        != baseline["input_sha256"]["development_samples"]
    ):
        raise ValueError("baseline and expanded validation data differ")
    torch.set_num_threads(4)
    rows, video, sensor = _load_samples(
        corpus, corpus / "video-features.npz", corpus / "sensor-features.npz"
    )
    development = _tensor_data(rows, video, sensor, "development")
    comparisons = {}
    reloads = {}
    points = [
        (step, f"epoch-{step // 11}", f"step-{step}") for step in (352, 704, 1408, 2816, 4224, 5632)
    ]
    points.append(("selected", "best", "best"))
    for label, baseline_label, expanded_label in points:
        old_file = baseline_dir / f"baseline-{baseline_label}.pt"
        new_file = run_dir / f"{expanded_label}.pt"
        if label == "selected":
            old_hash = baseline["results"]["baseline"]["best_checkpoint_sha256"]
            new_hash = report["best_checkpoint_sha256"]
        elif label == 5632:
            old_hash = baseline["results"]["baseline"]["final_checkpoint_sha256"]
            new_hash = report["final_checkpoint_sha256"]
        else:
            old_hash = None
            new_hash = None
        for path, expected in ((old_file, old_hash), (new_file, new_hash)):
            if expected is not None and sha256_file(path) != expected:
                raise ValueError("recorded checkpoint hash differs")
        old_saved = read_predictions(baseline_dir / f"baseline-{baseline_label}-predictions.jsonl")
        new_saved = read_predictions(run_dir / f"{expanded_label}-predictions.jsonl")
        old_expected = check_predictions(
            old_saved, development[4], development[3].tolist(), development[2].tolist()
        )
        new_expected = check_predictions(
            new_saved, development[4], development[3].tolist(), development[2].tolist()
        )
        check_probabilities(old_saved, old_expected)
        check_probabilities(new_saved, new_expected)
        deltas = []
        for path, expected in ((old_file, old_expected), (new_file, new_expected)):
            model = JointTemporalScorer().cpu().eval()
            model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            with torch.inference_mode():
                logits = model(development[0], development[1]).numpy()
            deltas.append(assert_logits_match(logits, expected))
        comparisons[str(label)] = compare_predictions(old_saved, new_saved)
        reloads[str(label)] = {
            "baseline_max_logit_difference": deltas[0],
            "expanded_max_logit_difference": deltas[1],
            "baseline_checkpoint_sha256": sha256_file(old_file),
            "expanded_checkpoint_sha256": sha256_file(new_file),
        }
    receipt = {
        "status": "PASS_RELOAD_AND_MATCHED_UPDATE_COMPARISON",
        "development_examples": len(development[4]),
        "comparisons": comparisons,
        "reloads": reloads,
        "runner_sha256": sha256_file(Path(__file__)),
        "expanded_result_sha256": sha256_file(report_path),
        "baseline_result_sha256": sha256_file(baseline_path),
        "selected_interval_accounts_for_selection": False,
        "bootstrap_unit": "development session",
        "bootstrap_seed": 17,
        "bootstrap_repetitions": 2000,
        "sealed_audit_accessed": False,
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.corpus, args.run_dir, args.baseline_dir, args.output), indent=2))


if __name__ == "__main__":
    main()
