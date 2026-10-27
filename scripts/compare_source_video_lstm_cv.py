"""Compare fixed visual-LSTM and pooled-feature OOF candidate predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.summarize_source_unimodal_cv import paired_interval
from scripts.train_source_unimodal_cv import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--temporal-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("output exists")
    baseline = json.loads((args.baseline_dir / "run.json").read_bytes())
    temporal = json.loads((args.temporal_dir / "run.json").read_bytes())
    if (
        baseline["status"] != "PASS_TRAIN_ONLY_GROUP_CV"
        or temporal["status"] != "PASS_FIXED_TEMPORAL_TRAIN_ONLY_CV"
    ):
        raise ValueError("incomplete source run")
    if digest(args.baseline_dir / "run.json") != temporal["baseline_run_sha256"]:
        raise ValueError("baseline receipt mismatch")
    pairs = []
    for fold in range(3):
        old_entry = next(
            x for x in baseline["results"] if x["fold"] == fold and x["mode"] == "joint"
        )
        new_entry = next(x for x in temporal["results"] if x["fold"] == fold)
        old_path = args.baseline_dir / f"fold-{fold}-joint-predictions.jsonl"
        new_path = args.temporal_dir / f"fold-{fold}-predictions.jsonl"
        if (
            digest(old_path) != old_entry["prediction_sha256"]
            or digest(new_path) != new_entry["prediction_sha256"]
        ):
            raise ValueError("prediction hash mismatch")
        old_rows = {r["record_id"]: r for r in map(json.loads, old_path.read_bytes().splitlines())}
        new_rows = {r["record_id"]: r for r in map(json.loads, new_path.read_bytes().splitlines())}
        if len(old_rows) != old_entry["validation_rows"] or old_rows.keys() != new_rows.keys():
            raise ValueError("row identities differ")
        for key, old in old_rows.items():
            new = new_rows[key]
            if any(
                old[f] != new[f]
                for f in ("video_id", "none_visible", "candidate_ids", "target_candidate_ids")
            ):
                raise ValueError("paired targets differ")
            pairs.append((old, new))
    if len(pairs) != 2593 or len({a["record_id"] for a, _ in pairs}) != 2593:
        raise ValueError("incomplete or repeated held-out records")
    metrics = {}
    for name, index in [("baseline", 0), ("temporal", 1)]:
        rows = [pair[index] for pair in pairs]
        confidence = np.asarray([row["confidence"] for row in rows])
        correct = np.asarray([row["correct"] for row in rows])
        ece = 0.0
        for i in range(10):
            selected = (confidence >= i / 10) & (
                confidence <= 1 if i == 9 else confidence < (i + 1) / 10
            )
            if selected.any():
                ece += float(
                    selected.mean() * abs(confidence[selected].mean() - correct[selected].mean())
                )
        metrics[name] = {
            "accuracy": float(correct.mean()),
            "nll": float(np.mean([r["nll"] for r in rows])),
            "brier": float(np.mean([r["brier"] for r in rows])),
            "ece_10": ece,
            "visible_accuracy": float(
                np.mean([r["correct"] for r in rows if not r["none_visible"]])
            ),
            "none_visible_accuracy": float(
                np.mean([r["correct"] for r in rows if r["none_visible"]])
            ),
        }
    paired = {}
    for field in ("correct", "nll", "brier"):
        paired[field] = paired_interval(
            [
                {"video_id": old["video_id"], "delta": float(new[field]) - float(old[field])}
                for old, new in pairs
            ]
        )
    result = {
        "status": "PASS_PAIRED_TRAIN_ONLY_OOF_COMPARISON",
        "baseline_run_sha256": digest(args.baseline_dir / "run.json"),
        "temporal_run_sha256": digest(args.temporal_dir / "run.json"),
        "script_sha256": digest(Path(__file__)),
        "records": len(pairs),
        "metrics": metrics,
        "temporal_minus_baseline": paired,
        "seed": 17,
        "bootstrap_repetitions": 10000,
        "sealed_audit_opened": False,
        "scope": "previously observed train-source OOF; not blind or product quality",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as target:
        json.dump(result, target, indent=2)
        target.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
