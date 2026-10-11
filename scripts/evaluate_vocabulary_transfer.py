"""Evaluate only a completed, rule-selected vocabulary candidate; no tuning."""

import argparse
import json
from pathlib import Path

from scripts.build_vocabulary_transfer import VARIANTS
from scripts.range_generalization import evaluate, sha, write


def require_selection(result, history, config):
    if result["status"] != "PASS" or result["steps"] != 2048 or result.get("smoke"):
        raise ValueError("full training must have completed")
    if [r["step"] for r in history] != list(range(0, 2049, 128)):
        raise ValueError("incomplete history")
    if min(history, key=lambda r: r["selector"])["step"] != result["best_step"]:
        raise ValueError("checkpoint violates earliest minimum selector")
    if (
        config["curriculum_training"] != "comparison_vocabulary"
        or config["initial_checkpoint"] != "encoder_coverage"
        or config["seed"] != 17
    ):
        raise ValueError("wrong experiment")
    return result["best_step"]


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


if __name__ == "__main__":
    main()
