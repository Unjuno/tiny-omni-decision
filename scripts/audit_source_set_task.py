"""Audit whether a single-choice readout fulfills the select-every task."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from scripts.train_source_unimodal_cv import LABELS, PINNED, digest

RUN_SHA = "619f9ea6c704477c8c2ddcb6ea3030e52aac470e5205dd550cb0ae0dba14d906"
EPOCHS = (4, 8, 32)
INSTRUCTION = (
    "Select every visual candidate that caused the audio event; "
    "if none is visible, select none_visible."
)


def score_row(prediction, annotation):
    """Check the observed argmax and return hit and exact-set correctness."""
    candidates = prediction["candidate_ids"]
    targets = prediction["target_candidate_ids"]
    if (
        annotation["request"]["instruction"] != INSTRUCTION
        or set(annotation["supervision"]["target_candidate_ids"]) != set(targets)
        or annotation["supervision"]["none_visible"] != prediction["none_visible"]
    ):
        raise ValueError("instruction or supervision mismatch")
    probabilities = prediction["probabilities"]
    if len(probabilities) != len(candidates) + 1:
        raise ValueError("probability width mismatch")
    if len(set(candidates)) != len(candidates) or not set(targets) <= set(candidates):
        raise ValueError("candidate identity mismatch")
    if prediction["none_visible"] != (len(targets) == 0):
        raise ValueError("visibility mismatch")
    selected = max(range(len(probabilities)), key=probabilities.__getitem__)
    chosen = "none_visible" if selected == len(candidates) else candidates[selected]
    expected = set(targets) if targets else {"none_visible"}
    hit = chosen in expected
    if hit != prediction["correct"]:
        raise ValueError("recorded hit differs from argmax")
    return {"hit": hit, "exact": {chosen} == expected, "bound": len(expected) == 1}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if digest(LABELS) != PINNED[LABELS] or digest(args.run_dir / "run.json") != RUN_SHA:
        raise ValueError("source hash mismatch")
    run = json.loads((args.run_dir / "run.json").read_bytes())
    history_file = args.run_dir / "history.jsonl"
    if digest(history_file) != run["history_sha256"]:
        raise ValueError("history hash mismatch")
    history = {
        (r["fold"], r["mode"], r["epoch"]): r
        for r in map(json.loads, history_file.read_bytes().splitlines())
    }
    labels = {}
    for row in map(json.loads, LABELS.read_bytes().splitlines()):
        if row["record_id"] in labels or row["split"] != "train":
            raise ValueError("duplicate ID or non-train label")
        labels[row["record_id"]] = row
    result = {}
    for epoch in EPOCHS:
        seen = set()
        counts = collections.Counter()
        for fold in range(3):
            path = args.run_dir / f"fold-{fold}-baseline-epoch-{epoch}-predictions.jsonl"
            if digest(path) != history[(fold, "baseline", epoch)]["prediction_sha256"]:
                raise ValueError("prediction hash mismatch")
            for row in map(json.loads, path.read_bytes().splitlines()):
                key = row["record_id"]
                if key in seen or key not in labels:
                    raise ValueError("duplicate or missing ID")
                seen.add(key)
                value = score_row(row, labels[key])
                counts.update({k: int(v) for k, v in value.items()})
        if seen != labels.keys() or len(seen) != 2593:
            raise ValueError("record set mismatch")
        result[str(epoch)] = {"n": len(seen), **counts}
    result = {
        "run_sha256": RUN_SHA,
        "labels_sha256": PINNED[LABELS],
        "script_sha256": digest(Path(__file__)),
        "epochs": result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2)
        out.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
