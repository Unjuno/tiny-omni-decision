"""Validate and summarize the fixed train-only joint video/text diagnostic."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from tiny_omni_decision.decision_math import (
    brier_score,
    expected_calibration_error,
    negative_log_likelihood,
)
from tiny_omni_decision.paired_video_intervals import video_cluster_intervals

PAIR_HASH = "8869bb903591af3a0995eeca0eeae1b50b990c8b08752a6e5d4d549f8a356fd7"


def align(rows, pairs):
    expected = {
        (pair["video_id"], question["condition"], mode): question["target"]
        for pair in pairs
        for question in pair["questions"]
        for mode in ("normal", "black_pixels")
    }
    indexed = {}
    for row in rows:
        key = (row["video_id"], row["condition"], row["mode"])
        if key not in expected or key in indexed or row["target"] != expected[key]:
            raise ValueError("unexpected, duplicated or mislabeled prediction")
        p = np.asarray(row["probabilities"], dtype=float)
        if p.shape != (3,) or not np.isfinite(p).all() or (p < 0).any():
            raise ValueError("invalid probabilities")
        if not np.isclose(p.sum(), 1, atol=1e-6, rtol=0):
            raise ValueError("probabilities do not sum to one")
        indexed[key] = p
    if set(indexed) != set(expected):
        raise ValueError("incomplete diagnostic")
    targets = np.array([[q["target"] for q in p["questions"]] for p in pairs])
    values = {
        mode: np.array(
            [
                [indexed[(p["video_id"], q["condition"], mode)] for q in p["questions"]]
                for p in pairs
            ]
        )
        for mode in ("normal", "black_pixels")
    }
    return values, targets


def metrics(values, targets):
    correct = values.argmax(-1) == targets
    p, t = values.reshape(-1, 3).tolist(), targets.ravel().tolist()
    return dict(
        accuracy=float(correct.mean()),
        nll=negative_log_likelihood(p, t),
        brier=brier_score(p, t),
        ece=expected_calibration_error(p, t),
        initial_accuracy=float(correct[:, 0].mean()),
        final_accuracy=float(correct[:, 1].mean()),
        both_correct=float(correct.all(-1).mean()),
        predicted_condition_changes=int((values.argmax(-1)[:, 0] != values.argmax(-1)[:, 1]).sum()),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("fresh output required")
    if hashlib.sha256(args.pairs.read_bytes()).hexdigest() != PAIR_HASH:
        raise ValueError("pair corpus changed")
    completion = json.loads((args.run / "completion.json").read_text())
    path = args.run / "predictions.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if completion["rows"] != 396 or len(rows) != completion["rows"]:
        raise ValueError("complete 99-video diagnostic required")
    pairs = json.loads(args.pairs.read_text())["pairs"]
    values, targets = align(rows, pairs)
    result = dict(
        scope="train-only frozen diagnostic; not held-out generalization",
        predictions_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        completion=completion,
        metrics={name: metrics(p, targets) for name, p in values.items()},
        uncertainty=video_cluster_intervals(
            values["normal"], targets, reference=values["black_pixels"], draws=10000, seed=17
        ),
    )
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
