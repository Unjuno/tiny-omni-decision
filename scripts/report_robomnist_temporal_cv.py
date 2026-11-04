"""Summarize saved TRAIN-fold predictions; does not load models or data splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def rows(path, wanted_arm, steps):
    out = {s: {} for s in steps}
    for line in path.open(encoding="utf-8"):
        item = json.loads(line)
        if item.get("arm", "baseline") != wanted_arm or item["step"] not in out:
            continue
        key = (item["pair_unit_id"], item["audio_group"], item["video_group"], item["target"])
        out[item["step"]][key] = np.asarray(item["probabilities"], dtype=float)
    if any(len(v) != 1560 for v in out.values()):
        raise ValueError("expected 1,560 saved OOF predictions at every point")
    return out


def metric(pairs):
    keys = sorted(pairs)
    y = np.asarray([k[3] for k in keys])
    p = np.stack([pairs[k] for k in keys])
    ok = (p.argmax(-1) == y).astype(float)
    nll = -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1))
    brier = ((p - np.eye(2)[y]) ** 2).sum(-1)
    conf = p.max(-1)
    ece = 0.0
    for i in range(15):
        mask = (conf >= i / 15) & ((conf < (i + 1) / 15) if i < 14 else (conf <= 1))
        if mask.any():
            ece += mask.mean() * abs(ok[mask].mean() - conf[mask].mean())
    return {
        "accuracy": float(ok.mean()),
        "nll": float(nll.mean()),
        "brier": float(brier.mean()),
        "ece_15": float(ece),
    }, dict(zip(keys, zip(ok, nll, strict=True), strict=True))


def summarize(temporal, baseline, output):
    steps = (1, 25, 50, 100, 200, 400)
    ordered = rows(temporal / "oof-predictions.jsonl", "ordered", steps)
    shuffled = rows(temporal / "oof-predictions.jsonl", "shuffled", steps)
    pooled = rows(baseline / "oof-predictions.jsonl", "baseline", steps)
    ordered_metrics = {s: metric(ordered[s])[0] for s in steps}
    shuffled_metrics = {s: metric(shuffled[s])[0] for s in steps}
    pooled_metrics = {s: metric(pooled[s])[0] for s in steps}
    group_ids = sorted({k[0] for k in ordered[50]})
    if len(group_ids) != 390:
        raise ValueError("unexpected pair-unit count")
    by_group = {g: [k for k in ordered[50] if k[0] == g] for g in group_ids}
    rng = np.random.default_rng(17)
    intervals = {}
    for step in (50, 400):
        _, a = metric(ordered[step])
        _, b = metric(shuffled[step])
        acc, nll = [], []
        for _ in range(2000):
            sample = rng.choice(group_ids, size=390, replace=True)
            keys = [k for g in sample for k in by_group[g]]
            acc.append(np.mean([a[k][0] - b[k][0] for k in keys]))
            nll.append(np.mean([b[k][1] - a[k][1] for k in keys]))
        intervals[step] = {
            "ordered_minus_shuffled_accuracy": ordered_metrics[step]["accuracy"]
            - shuffled_metrics[step]["accuracy"],
            "accuracy_delta_95pct": np.quantile(acc, [0.025, 0.975]).tolist(),
            "shuffled_minus_ordered_nll": shuffled_metrics[step]["nll"]
            - ordered_metrics[step]["nll"],
            "nll_delta_95pct": np.quantile(nll, [0.025, 0.975]).tolist(),
            "groups": 390,
            "replicates": 2000,
            "seed": 17,
        }
    result = {
        "scope": "TRAIN-only OOF development; not independent evaluation",
        "ordered": ordered_metrics,
        "shuffled": shuffled_metrics,
        "pooled_baseline": pooled_metrics,
        "paired_group_bootstrap": intervals,
        "limitations": [
            "OOF folds share fitting data",
            "prior project development exposure",
            "AV same-repetition correspondence only",
            "independent stream shuffles are not raw-media reversals",
        ],
    }
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--temporal", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(summarize(a.temporal, a.baseline, a.output), indent=2))
