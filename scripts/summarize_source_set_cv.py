"""Summarize a complete fixed source set-decision run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.source_set_metrics import binary_metrics, score_record
from scripts.summarize_source_temporal_long_curve import paired_bootstrap
from scripts.train_source_unimodal_cv import digest


def summarize(rows):
    if not rows or len({r["record_id"] for r in rows}) != len(rows):
        raise ValueError("empty or duplicate records")
    candidate_p, candidate_y, none_p, none_y, checked = [], [], [], [], []
    for r in rows:
        ids, targets = r["candidate_ids"], r["target_candidate_ids"]
        none = r["none_probability"]
        conditional = r["conditional_probabilities"]
        scores = score_record(ids, targets, r["selected"], conditional, none)
        for k, value in scores.items():
            if value != r[k]:
                raise ValueError("saved metric mismatch")
        checked.append(scores)
        candidate_p.extend((1 - none) * p for p in conditional)
        candidate_y.extend(int(k in targets) for k in ids)
        none_p.append(none)
        none_y.append(int(not targets))
    return {
        "rows": len(rows),
        "video_groups": len({r["video_id"] for r in rows}),
        "exact_set_accuracy": float(np.mean([r["exact"] for r in checked])),
        "none_decision_accuracy": float(np.mean([r["none_correct"] for r in checked])),
        "macro_record_candidate_nll": float(np.mean([r["membership_nll"] for r in checked])),
        "macro_record_candidate_brier": float(np.mean([r["membership_brier"] for r in checked])),
        "candidate_binary_micro": binary_metrics(candidate_p, candidate_y),
        "none_binary": binary_metrics(none_p, none_y),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    receipt_path = args.run_dir / "run.json"
    receipt = json.loads(receipt_path.read_bytes())
    if receipt["status"] != "PASS_FIXED_TRAIN_SOURCE_CV" or len(receipt["results"]) != 9:
        raise ValueError("incomplete run")
    lookup = {(r["fold"], r["mode"]): r for r in receipt["results"]}
    if len(lookup) != 9:
        raise ValueError("duplicate arms")
    results = {}
    predictions = {}
    for mode in ("audio", "video", "joint"):
        rows = []
        for fold in range(3):
            stem = f"fold-{fold}-{mode}"
            prediction_path = args.run_dir / f"{stem}-predictions.jsonl"
            if digest(prediction_path) != lookup[(fold, mode)]["prediction_sha256"]:
                raise ValueError("prediction hash mismatch")
            if digest(args.run_dir / f"{stem}.pt") != lookup[(fold, mode)]["checkpoint_sha256"]:
                raise ValueError("checkpoint hash mismatch")
            rows.extend(map(json.loads, prediction_path.read_bytes().splitlines()))
        if len(rows) != 2593:
            raise ValueError("unexpected row count")
        results[mode] = summarize(rows)
        predictions[mode] = {r["record_id"]: r for r in rows}
    comparisons = {}
    for mode in ("audio", "video"):
        if predictions[mode].keys() != predictions["joint"].keys():
            raise ValueError("comparison record mismatch")
        pairs = []
        for key, old in predictions[mode].items():
            new = predictions["joint"][key]
            if any(old[k] != new[k] for k in ("video_id", "candidate_ids", "target_candidate_ids")):
                raise ValueError("comparison target mismatch")
            pairs.append((dict(old, correct=old["exact"]), dict(new, correct=new["exact"])))
        comparisons[f"joint_minus_{mode}"] = paired_bootstrap(pairs, "correct")
    result = {
        "run_sha256": digest(receipt_path),
        "script_sha256": digest(Path(__file__)),
        "scope": "nonblind train-source OOF; not a product or audit test",
        "modes": results,
        "paired_exact_bootstrap": comparisons,
        "bootstrap_seed": 17,
        "bootstrap_draws": 10000,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2)
        out.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
