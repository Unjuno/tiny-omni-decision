"""Describe set-decision errors and paired transitions on fixed CV outputs."""

import argparse
import collections
import json
from pathlib import Path

from scripts.compare_source_set_single import NEW_SHA, OLD_SHA
from scripts.train_source_unimodal_cv import digest


def error_kind(row):
    target = set(row["target_candidate_ids"])
    predicted = set(row["selected"])
    candidates = set(row["candidate_ids"])
    if not target <= candidates or not predicted <= candidates | {"none_visible"}:
        raise ValueError("unknown candidate")
    if not predicted or ("none_visible" in predicted and len(predicted) != 1):
        raise ValueError("invalid predicted set")
    if not target:
        return "none", "exact" if predicted == {"none_visible"} else "false_visible"
    bucket = "one" if len(target) == 1 else "multi"
    if predicted == {"none_visible"}:
        return bucket, "miss_all_none"
    fp, fn = bool(predicted - target), bool(target - predicted)
    kind = "fp_fn" if fp and fn else "fp_only" if fp else "fn_only" if fn else "exact"
    return bucket, kind


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if digest(args.old / "run.json") != OLD_SHA or digest(args.new / "run.json") != NEW_SHA:
        raise ValueError("run hash mismatch")
    old_run = json.loads((args.old / "run.json").read_bytes())
    new_run = json.loads((args.new / "run.json").read_bytes())
    history_path = args.old / "history.jsonl"
    if digest(history_path) != old_run["history_sha256"]:
        raise ValueError("history mismatch")
    history = {
        (r["fold"], r["mode"], r["epoch"]): r
        for r in map(json.loads, history_path.read_bytes().splitlines())
    }
    receipts = {(r["fold"], r["mode"]): r for r in new_run["results"]}
    errors = collections.Counter()
    transitions = collections.Counter()
    seen = set()
    for fold in range(3):
        p = args.old / f"fold-{fold}-baseline-epoch-7-predictions.jsonl"
        q = args.new / f"fold-{fold}-joint-predictions.jsonl"
        if (
            digest(p) != history[(fold, "baseline", 7)]["prediction_sha256"]
            or digest(q) != receipts[(fold, "joint")]["prediction_sha256"]
        ):
            raise ValueError("prediction hash mismatch")
        x = {r["record_id"]: r for r in map(json.loads, p.read_bytes().splitlines())}
        y = {r["record_id"]: r for r in map(json.loads, q.read_bytes().splitlines())}
        if x.keys() != y.keys() or seen & x.keys():
            raise ValueError("record mismatch")
        seen.update(x)
        for key, before in x.items():
            after = y[key]
            if any(
                before[k] != after[k] for k in ("video_id", "candidate_ids", "target_candidate_ids")
            ):
                raise ValueError("target mismatch")
            bucket, kind = error_kind(after)
            new_correct = kind == "exact"
            if new_correct != after["exact"]:
                raise ValueError("saved exact mismatch")
            old_correct = bool(before["correct"] and len(before["target_candidate_ids"]) <= 1)
            transition = (
                "both"
                if old_correct and new_correct
                else "new_only"
                if new_correct
                else "old_only"
                if old_correct
                else "neither"
            )
            errors[f"{bucket}:{kind}"] += 1
            transitions[f"{bucket}:{transition}"] += 1
    if len(seen) != 2593:
        raise ValueError("unexpected record count")
    result = {
        "n": len(seen),
        "old_sha256": OLD_SHA,
        "new_sha256": NEW_SHA,
        "script_sha256": digest(Path(__file__)),
        "error_counts": dict(errors),
        "paired_counts": dict(transitions),
        "scope": "nonblind descriptive development audit",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
