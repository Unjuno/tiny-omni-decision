"""Summarize fixed train-only out-of-fold source-candidate predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def paired_interval(rows, seed=17, repetitions=10000):
    groups = {}
    for row in rows:
        groups.setdefault(row["video_id"], []).append(row["delta"])
    if not groups:
        raise ValueError("empty paired comparison")
    group_values = list(groups.values())
    rng = np.random.default_rng(seed)
    bootstrap = []
    for _ in range(repetitions):
        indices = rng.integers(0, len(group_values), len(group_values))
        bootstrap.append(float(np.mean([value for i in indices for value in group_values[int(i)]])))
    return {
        "delta": float(np.mean([row["delta"] for row in rows])),
        "interval_95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        "video_groups": len(groups),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("output exists")
    run_raw = (args.run_dir / "run.json").read_bytes()
    run = json.loads(run_raw)
    if run["status"] != "PASS_TRAIN_ONLY_GROUP_CV" or len(run["results"]) != 9:
        raise ValueError("incomplete training run")
    arms = {mode: {} for mode in ("audio", "video", "joint")}
    for entry in run["results"]:
        mode, fold = entry["mode"], entry["fold"]
        raw = (args.run_dir / f"fold-{fold}-{mode}-predictions.jsonl").read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry["prediction_sha256"]:
            raise ValueError("prediction hash mismatch")
        for line in raw.splitlines():
            row = json.loads(line)
            if row["record_id"] in arms[mode]:
                raise ValueError("record duplicated across folds")
            arms[mode][row["record_id"]] = row
    if any(set(rows) != set(arms["joint"]) for rows in arms.values()) or len(arms["joint"]) != 2593:
        raise ValueError("out-of-fold record mismatch")
    by_mode, comparisons = {}, {}
    for mode, rows in arms.items():
        values = list(rows.values())
        confidences = np.asarray([row["confidence"] for row in values])
        correct = np.asarray([row["correct"] for row in values])
        ece = 0.0
        for index in range(10):
            selected = (confidences >= index / 10) & (
                confidences <= 1 if index == 9 else confidences < (index + 1) / 10
            )
            if selected.any():
                ece += float(
                    selected.mean() * abs(confidences[selected].mean() - correct[selected].mean())
                )
        by_mode[mode] = {
            "accuracy": float(correct.mean()),
            "nll": float(np.mean([x["nll"] for x in values])),
            "brier": float(np.mean([x["brier"] for x in values])),
            "ece_10": ece,
            "visible_accuracy": float(
                np.mean([x["correct"] for x in values if not x["none_visible"]])
            ),
            "none_visible_accuracy": float(
                np.mean([x["correct"] for x in values if x["none_visible"]])
            ),
        }
    for other in ("audio", "video"):
        paired = []
        for key, joint in arms["joint"].items():
            compared = arms[other][key]
            if any(
                joint[field] != compared[field]
                for field in ("video_id", "none_visible", "candidate_ids", "target_candidate_ids")
            ):
                raise ValueError("mismatched paired targets")
            paired.append(
                {
                    "video_id": joint["video_id"],
                    "delta": int(joint["correct"]) - int(compared["correct"]),
                }
            )
        comparisons[f"joint_minus_{other}"] = paired_interval(paired)
    result = {
        "status": "PASS_TRAIN_ONLY_OOF_SUMMARY",
        "run_sha256": hashlib.sha256(run_raw).hexdigest(),
        "rows": 2593,
        "metrics": by_mode,
        "comparisons": comparisons,
        "seed": 17,
        "bootstrap_repetitions": 10000,
        "sealed_audit_opened": False,
        "scope": "internal training-source group cross-validation, not blind product evaluation",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as target:
        json.dump(result, target, indent=2)
        target.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
