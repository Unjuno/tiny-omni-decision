"""Evaluate only a completed, rule-selected vocabulary candidate; no tuning."""

import argparse
import json
import math
from pathlib import Path

from scripts.build_vocabulary_transfer import VARIANTS
from scripts.range_generalization import evaluate, sha, write


def require_selection(result, history, config):
    if result["status"] != "PASS" or result["steps"] != 2048 or result.get("smoke"):
        raise ValueError("full training must have completed")
    if [r["step"] for r in history] != list(range(0, 2049, 128)):
        raise ValueError("incomplete history")
    if any(not math.isfinite(r["selector"]) for r in history):
        raise ValueError("nonfinite selector")
    if min(history, key=lambda r: r["selector"])["step"] != result["best_step"]:
        raise ValueError("checkpoint violates earliest minimum selector")
    if (
        config["curriculum_training"] != "comparison_vocabulary"
        or config["initial_checkpoint"] != "encoder_coverage"
        or config["seed"] != 17
    ):
        raise ValueError("wrong experiment")
    return result["best_step"]


def opposing_pairs(rows, predictions):
    pairs = {}
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("duplicate input IDs")
    for row, pred in zip(rows, predictions, strict=True):
        if any(row[k] != pred[k] for k in ("id", "options", "target")):
            raise ValueError("prediction mismatch")
        probs = pred["probabilities"]
        if (
            len(probs) != len(row["options"])
            or any(not math.isfinite(x) or x < 0 or x > 1 for x in probs)
            or abs(sum(probs) - 1) > 1e-5
        ):
            raise ValueError("invalid probabilities")
        if row["task"] not in ("extremum", "temporal_order"):
            continue
        o = row["oracle"]
        key = (row["group"], o["field"])
        values = pairs.setdefault(key, {})
        if o["direction"] in values:
            raise ValueError("duplicate opposing condition")
        chosen = row["options"][max(range(len(probs)), key=probs.__getitem__)]
        values[o["direction"]] = (chosen, row["target"])
    summary = {}
    for (_, field), values in pairs.items():
        if set(values) != {"minimum", "maximum"}:
            raise ValueError("missing opposing condition")
        a, b = values["minimum"], values["maximum"]
        counts = summary.setdefault(field, dict(pairs=0, both_correct=0, prediction_changed=0))
        counts["pairs"] += 1
        counts["both_correct"] += int(a[0] == a[1] and b[0] == b[1])
        counts["prediction_changed"] += int(a[0] != b[0])
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "model", "data", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    selection_file = args.output.parent / (args.output.name + "-selection.json")
    if args.output.exists() or selection_file.exists():
        raise FileExistsError("output already exists")
    run = args.root / "conditional-repair-comparison-vocabulary-01"

    def read(name):
        return json.loads((run / name).read_text(encoding="utf-8"))

    result, history, config = read("result.json"), read("history.json"), read("config.json")
    step = require_selection(result, history, config)
    if sha(run / "best-delta.pt") != result["best_sha256"]:
        raise ValueError("selected weight mismatch")
    for path, digest in config["paths"].items():
        if sha(Path(path)) != digest:
            raise ValueError("training input hash mismatch")
    checkpoints = {
        "parent": (
            "conditional-repair-lexical-range-01",
            "fccee21e0e5ffbaffdee899a65fe6538f52f3b788752a56b6b5b6388cad15abb",
        ),
        "candidate": (run.name, result["best_sha256"]),
    }
    write(
        selection_file,
        dict(
            step=step,
            checkpoints=checkpoints,
            result_sha256=sha(run / "result.json"),
            history_sha256=sha(run / "history.json"),
            config_sha256=sha(run / "config.json"),
            wrapper_sha256=sha(Path(__file__)),
            selection_rule="existing earliest validation mean NLL, frozen before inference",
        ),
    )
    evaluate(
        args.root,
        args.model,
        args.data,
        "1a825b413e32f319dead47ccbef07ef3b36f2c391ca2c197a694aaeb4ed54dd2",
        args.output,
        checkpoints,
        VARIANTS,
        10240,
    )
    rows = [
        json.loads(line)
        for line in (args.data / "inputs.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    pairs = {}
    for model in checkpoints:
        pairs[model] = {}
        for variant in VARIANTS:
            predictions = json.loads(
                (args.output / f"{model}-{variant}-predictions.json").read_text(encoding="utf-8")
            )
            pairs[model][variant] = opposing_pairs(
                [r for r in rows if r["variant"] == variant], predictions
            )
    write(args.output / "opposing-pairs.json", pairs)


if __name__ == "__main__":
    main()
