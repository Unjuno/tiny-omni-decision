"""Compare saved single-choice and set-valued exact-set predictions."""

import argparse
import json
from pathlib import Path

from scripts.summarize_source_temporal_long_curve import paired_bootstrap
from scripts.train_source_unimodal_cv import digest

OLD_SHA = "619f9ea6c704477c8c2ddcb6ea3030e52aac470e5205dd550cb0ae0dba14d906"
NEW_SHA = "5df64f799220dbe95c1234556266c92e4aff332c524bc79c048e1a485fe2218b"


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
    old = json.loads((args.old / "run.json").read_bytes())
    new = json.loads((args.new / "run.json").read_bytes())
    history_path = args.old / "history.jsonl"
    if (
        digest(history_path) != old["history_sha256"]
        or new["status"] != "PASS_FIXED_TRAIN_SOURCE_CV"
    ):
        raise ValueError("incomplete run")
    history = {
        (r["fold"], r["mode"], r["epoch"]): r
        for r in map(json.loads, history_path.read_bytes().splitlines())
    }
    receipts = {(r["fold"], r["mode"]): r for r in new["results"]}
    pairs = []
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
        if x.keys() != y.keys():
            raise ValueError("record ID mismatch")
        for key, prior in x.items():
            candidate = y[key]
            if any(
                prior[k] != candidate[k]
                for k in ("video_id", "candidate_ids", "target_candidate_ids")
            ):
                raise ValueError("target mismatch")
            pairs.append(
                (
                    {
                        "video_id": prior["video_id"],
                        "correct": bool(
                            prior["correct"] and len(prior["target_candidate_ids"]) <= 1
                        ),
                    },
                    {"correct": candidate["exact"]},
                )
            )
    if len(pairs) != 2593:
        raise ValueError("unexpected sample count")
    result = {
        "old_run_sha256": OLD_SHA,
        "new_run_sha256": NEW_SHA,
        "script_sha256": digest(Path(__file__)),
        "n": len(pairs),
        "old_exact": sum(x["correct"] for x, _ in pairs),
        "new_exact": sum(y["correct"] for _, y in pairs),
        "set_minus_single": paired_bootstrap(pairs, "correct"),
        "bootstrap_seed": 17,
        "bootstrap_draws": 10000,
        "scope": "nonblind train-source development; different objectives and decision rules",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
