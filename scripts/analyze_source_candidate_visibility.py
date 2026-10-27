"""Stratify already saved Perception candidate predictions by visibility."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

BASE = Path(
    "C:/CodexArtifacts/pretrained-reuse-r2/perception-source-candidate-validation-transfer-20261026-attempt-07"
)
INPUTS = {
    "joint": (
        "diagnostic-controls/predictions-joint.jsonl",
        "d9909c9a0c15bceaeb542802ef6aef3d81fdb5c5a3817bc1bc403fc05f7f91be",
    ),
    "zero_audio": (
        "diagnostic-controls/predictions-video_only_zero_audio.jsonl",
        "a2ba7a1966f41dd7401b9515bfcc1af94c51547425d97025bf60e2e0da11d23c",
    ),
    "mismatched_audio": (
        "diagnostic-controls/predictions-mismatched_audio.jsonl",
        "c050188d85d32171c159c6933b713e31f74c2d0fc09d380c9ff7590bd175371c",
    ),
    "option_prior": (
        "option-only-baseline/predictions.jsonl",
        "44a8d045953d4dd218860362b280d007313d7abeffe69f4eb49afffe4a6da1be",
    ),
}


def matched_rows(reference, other):
    ref = {row["record_id"]: row for row in reference}
    comp = {row["record_id"]: row for row in other}
    if len(ref) != len(reference) or len(comp) != len(other) or ref.keys() != comp.keys():
        raise ValueError("duplicate or mismatched record IDs")
    for key in ref:
        a, b = ref[key], comp[key]
        if any(
            a[field] != b[field] for field in ("video_id", "none_visible", "target_candidate_ids")
        ):
            raise ValueError("inconsistent grouping or targets")
    return [(ref[key], comp[key]) for key in sorted(ref)]


def summarize(pairs, none_visible, seed=17, repetitions=10000):
    rows = [(a, b) for a, b in pairs if a["none_visible"] == none_visible]
    groups = {}
    for a, b in rows:
        delta = int(a["top1_correct"]) - int(b["top1_correct"])
        groups.setdefault(a["video_id"], []).append(delta)
    if not groups:
        raise ValueError("empty stratum")
    group_ids = sorted(groups)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(repetitions):
        choices = rng.integers(0, len(group_ids), size=len(group_ids))
        deltas = [d for i in choices for d in groups[group_ids[int(i)]]]
        samples.append(float(np.mean(deltas)))
    joint = sum(bool(a["top1_correct"]) for a, _ in rows)
    comparison = sum(bool(b["top1_correct"]) for _, b in rows)
    return {
        "rows": len(rows),
        "video_groups": len(groups),
        "joint_correct": joint,
        "comparison_correct": comparison,
        "joint_minus_comparison_accuracy": (joint - comparison) / len(rows),
        "bootstrap_percentile_95": np.quantile(samples, [0.025, 0.975]).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("output exists")
    rows = {}
    for name, (relative, expected) in INPUTS.items():
        data = (BASE / relative).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("input prediction hash mismatch")
        rows[name] = [json.loads(line) for line in data.splitlines()]
    comparisons = {}
    for name in ("zero_audio", "mismatched_audio", "option_prior"):
        pairs = matched_rows(rows["joint"], rows[name])
        comparisons[name] = {
            stratum: summarize(pairs, flag)
            for stratum, flag in [("visible", False), ("none_visible", True)]
        }
    result = {
        "status": "POSTHOC_NONBLIND_DIAGNOSTIC",
        "input_sha256": {name: value[1] for name, value in INPUTS.items()},
        "comparisons": comparisons,
        "seed": 17,
        "bootstrap_repetitions": 10000,
        "method": "paired video-group bootstrap, separately within each stratum",
        "training": False,
        "model_run": False,
        "sealed_audit_opened": False,
        "limitations": "exploratory subgroup analysis of previously observed validation; "
        "zeroing audio is not an independently trained unimodal model",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as target:
        json.dump(result, target, indent=2)
        target.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
