"""Paired descriptive audit of balanced versus ordinary set objective."""

import argparse
import collections
import json
from pathlib import Path

from scripts.summarize_source_temporal_long_curve import paired_bootstrap
from scripts.train_source_unimodal_cv import digest

OLD = "5df64f799220dbe95c1234556266c92e4aff332c524bc79c048e1a485fe2218b"
NEW = "ee04092410d024692b8123f953bc65c772fffb8f930db03f5a72e633d216267c"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if digest(args.old / "run.json") != OLD or digest(args.new / "run.json") != NEW:
        raise ValueError("run hash mismatch")
    a = json.loads((args.old / "run.json").read_bytes())
    b = json.loads((args.new / "run.json").read_bytes())
    pairs, counts, seen = [], collections.Counter(), set()
    for fold in range(3):
        pa = args.old / f"fold-{fold}-joint-predictions.jsonl"
        pb = args.new / f"fold-{fold}-joint-predictions.jsonl"
        expected_a = next(
            x["prediction_sha256"]
            for x in a["results"]
            if x["fold"] == fold and x["mode"] == "joint"
        )
        expected_b = next(
            x["prediction_sha256"]
            for x in b["results"]
            if x["fold"] == fold and x["mode"] == "joint"
        )
        if digest(pa) != expected_a or digest(pb) != expected_b:
            raise ValueError("prediction hash mismatch")
        x = {r["record_id"]: r for r in map(json.loads, pa.read_bytes().splitlines())}
        y = {r["record_id"]: r for r in map(json.loads, pb.read_bytes().splitlines())}
        if x.keys() != y.keys() or seen & x.keys():
            raise ValueError("record mismatch")
        seen.update(x)
        for key, prior in x.items():
            new = y[key]
            if any(
                prior[k] != new[k] for k in ("video_id", "candidate_ids", "target_candidate_ids")
            ):
                raise ValueError("target mismatch")
            n = len(prior["target_candidate_ids"])
            s = "none" if n == 0 else "one" if n == 1 else "multi"
            pairs.append(
                (
                    {"video_id": prior["video_id"], "correct": prior["exact"], "stratum": s},
                    {"correct": new["exact"]},
                )
            )
            for name, row in (("old", prior), ("balanced", new)):
                truth = set(row["target_candidate_ids"])
                selected = set(row["selected"]) - {"none_visible"}
                counts[f"{s}:{name}:tp"] += len(truth & selected)
                counts[f"{s}:{name}:fn"] += len(truth - selected)
                counts[f"{s}:{name}:fp"] += len(selected - truth)
    if len(seen) != 2593:
        raise ValueError("unexpected coverage")
    results = {}
    for group in ("all", "none", "one", "multi"):
        subset = [pair for pair in pairs if group == "all" or pair[0]["stratum"] == group]
        results[group] = {
            "n": len(subset),
            "old_correct": sum(x["correct"] for x, _ in subset),
            "new_correct": sum(y["correct"] for _, y in subset),
            "paired_delta": paired_bootstrap(subset, "correct"),
        }
    result = {
        "old_sha256": OLD,
        "new_sha256": NEW,
        "script_sha256": digest(Path(__file__)),
        "results": results,
        "candidate_counts": dict(counts),
        "bootstrap_seed": 17,
        "bootstrap_draws": 10000,
        "scope": "nonblind train-source development, video-group uncertainty only",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
