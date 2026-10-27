"""Aggregate and pair fixed-epoch train-only temporal CV predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.train_source_temporal_long_curve import EVALUATIONS
from scripts.train_source_unimodal_cv import digest


def paired_bootstrap(pairs, field, seed=17, repetitions=10000):
    groups = {}
    for old, new in pairs:
        group = groups.setdefault(old["video_id"], [0.0, 0])
        group[0] += float(new[field]) - float(old[field])
        group[1] += 1
    if not groups:
        raise ValueError("empty comparison")
    values = np.asarray(list(groups.values()), dtype=np.float64)
    indices = np.random.default_rng(seed).integers(0, len(values), (repetitions, len(values)))
    selected = values[indices]
    estimates = selected[:, :, 0].sum(axis=1) / selected[:, :, 1].sum(axis=1)
    return {
        "delta": float(values[:, 0].sum() / values[:, 1].sum()),
        "interval_95": np.quantile(estimates, [0.025, 0.975]).tolist(),
        "video_groups": len(groups),
    }


def metrics(rows):
    confidence = np.asarray([row["confidence"] for row in rows])
    correct = np.asarray([row["correct"] for row in rows])
    ece = 0.0
    for index in range(10):
        selected = (confidence >= index / 10) & (
            confidence <= 1 if index == 9 else confidence < (index + 1) / 10
        )
        if selected.any():
            ece += float(
                selected.mean() * abs(confidence[selected].mean() - correct[selected].mean())
            )
    return {
        "accuracy": float(correct.mean()),
        "nll": float(np.mean([r["nll"] for r in rows])),
        "brier": float(np.mean([r["brier"] for r in rows])),
        "ece_10": ece,
        "visible_accuracy": float(np.mean([r["correct"] for r in rows if not r["none_visible"]])),
        "none_visible_accuracy": float(np.mean([r["correct"] for r in rows if r["none_visible"]])),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("output exists")
    run_path = args.run_dir / "run.json"
    run = json.loads(run_path.read_bytes())
    history_path = args.run_dir / "history.jsonl"
    if (
        run["status"] != "PASS_FIXED_32_EPOCH_CURVES"
        or digest(history_path) != run["history_sha256"]
    ):
        raise ValueError("incomplete run or history hash mismatch")
    history = [json.loads(line) for line in history_path.read_bytes().splitlines()]
    lookup = {(r["fold"], r["mode"], r["epoch"]): r for r in history}
    if len(lookup) != 192:
        raise ValueError("unexpected history coverage")
    curves = []
    for epoch in EVALUATIONS:
        data = {"baseline": {}, "temporal": {}}
        for fold in range(3):
            for mode in data:
                path = args.run_dir / f"fold-{fold}-{mode}-epoch-{epoch}-predictions.jsonl"
                if digest(path) != lookup[(fold, mode, epoch)]["prediction_sha256"]:
                    raise ValueError("prediction hash mismatch")
                for row in map(json.loads, path.read_bytes().splitlines()):
                    if row["record_id"] in data[mode]:
                        raise ValueError("duplicate held-out record")
                    data[mode][row["record_id"]] = row
        if len(data["baseline"]) != 2593 or data["baseline"].keys() != data["temporal"].keys():
            raise ValueError("prediction record mismatch")
        pairs = []
        for key, old in data["baseline"].items():
            new = data["temporal"][key]
            if any(
                old[field] != new[field]
                for field in ("video_id", "none_visible", "candidate_ids", "target_candidate_ids")
            ):
                raise ValueError("paired target mismatch")
            pairs.append((old, new))
        curves.append(
            {
                "epoch": epoch,
                "baseline": metrics(list(data["baseline"].values())),
                "temporal": metrics(list(data["temporal"].values())),
                "temporal_minus_baseline": {
                    field: paired_bootstrap(pairs, field) for field in ("correct", "nll", "brier")
                },
            }
        )
    result = {
        "status": "PASS_FIXED_EPOCH_CURVE_SUMMARY",
        "run_sha256": digest(run_path),
        "script_sha256": digest(Path(__file__)),
        "curves": curves,
        "seed": 17,
        "bootstrap_repetitions": 10000,
        "sealed_audit_opened": False,
        "scope": "repeated train-source OOF development; no model selection or product claim",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as target:
        json.dump(result, target, indent=2)
        target.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
