"""Audit confidence and multi-answer semantics in fixed OOF predictions."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from scripts.train_source_unimodal_cv import digest

RUN_SHA = "619f9ea6c704477c8c2ddcb6ea3030e52aac470e5205dd550cb0ae0dba14d906"
EPOCHS = (4, 8, 32)


def audit(data):
    """Summarize errors and matched prediction transitions."""
    if any(data[e].keys() != data[EPOCHS[0]].keys() for e in EPOCHS):
        raise ValueError("record IDs differ between epochs")
    reports = {}
    for epoch in EPOCHS:
        rows = list(data[epoch].values())
        if not rows:
            raise ValueError("empty predictions")
        wrong = [r for r in rows if not r["correct"]]
        total_nll = sum(r["nll"] for r in rows)
        reports[epoch] = {
            "rows": len(rows),
            "target_count": dict(collections.Counter(len(r["target_candidate_ids"]) for r in rows)),
            "errors": len(wrong),
            "errors_confidence_gt_0_9": sum(r["confidence"] > 0.9 for r in wrong),
            "mean_error_confidence": (
                sum(r["confidence"] for r in wrong) / len(wrong) if wrong else None
            ),
            "mean_error_nll": sum(r["nll"] for r in wrong) / len(wrong) if wrong else None,
            "error_nll_fraction": sum(r["nll"] for r in wrong) / total_nll if total_nll else None,
        }
    transitions = {}
    for a, b in ((4, 8), (4, 32), (8, 32)):
        pairs = [(data[a][k], data[b][k]) for k in data[a]]
        for x, y in pairs:
            if any(
                x[k] != y[k]
                for k in ("video_id", "target_candidate_ids", "candidate_ids", "none_visible")
            ):
                raise ValueError("paired target mismatch")
        transitions[f"{a}-{b}"] = {
            "both_wrong": sum(not x["correct"] and not y["correct"] for x, y in pairs),
            "fixed": sum(not x["correct"] and y["correct"] for x, y in pairs),
            "new_errors": sum(x["correct"] and not y["correct"] for x, y in pairs),
        }
    return {"epochs": reports, "transitions": transitions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    receipt = args.run_dir / "run.json"
    if digest(receipt) != RUN_SHA:
        raise ValueError("run receipt mismatch")
    run = json.loads(receipt.read_bytes())
    history_path = args.run_dir / "history.jsonl"
    if digest(history_path) != run["history_sha256"]:
        raise ValueError("history mismatch")
    history = {
        (r["fold"], r["mode"], r["epoch"]): r
        for r in map(json.loads, history_path.read_bytes().splitlines())
    }
    data = {e: {} for e in EPOCHS}
    for epoch in EPOCHS:
        for fold in range(3):
            path = args.run_dir / f"fold-{fold}-baseline-epoch-{epoch}-predictions.jsonl"
            if digest(path) != history[(fold, "baseline", epoch)]["prediction_sha256"]:
                raise ValueError("prediction hash mismatch")
            for row in map(json.loads, path.read_bytes().splitlines()):
                if row["record_id"] in data[epoch]:
                    raise ValueError("duplicate record")
                data[epoch][row["record_id"]] = row
    if any(len(data[e]) != 2593 for e in EPOCHS):
        raise ValueError("unexpected record count")
    result = {"run_sha256": RUN_SHA, "script_sha256": digest(Path(__file__)), **audit(data)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
