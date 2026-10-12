"""Recompute selected predictions and paired development uncertainty."""

import argparse
import collections
import json
from pathlib import Path

import numpy as np

from scripts.audit_correspondence_validation import sha, write
from scripts.train_occlusion_readout import PINS, ROOT, summarize
from tiny_omni_decision.paired_video_intervals import video_cluster_intervals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("fresh output required")
    run = json.loads((args.run / "result.json").read_text())
    config = json.loads((args.run / "config.json").read_text())
    if set(run) != {"mean", "attention"} or config["epochs"] != 50 or config["seed"] != 17:
        raise ValueError("complete fixed comparison required")
    pairs = {}
    for split, pin in PINS.items():
        path = ROOT / f"perception-occlusion-pairs-01/{split}.json"
        if sha(path) != pin:
            raise ValueError("pairs changed")
        pairs[split] = json.loads(path.read_text())["pairs"]
    counts = collections.defaultdict(collections.Counter)
    for pair in pairs["train"]:
        for question in pair["questions"]:
            counts[(tuple(pair["options"]), question["condition"])][question["target"]] += 1
    prior = []
    for pair in pairs["valid"]:
        for question in pair["questions"]:
            count = counts[(tuple(pair["options"]), question["condition"])]
            prior.append([(count[i] + 1) / (sum(count.values()) + 3) for i in range(3)])
    prior = np.array(prior).reshape(-1, 2, 3)
    targets = np.array([[q["target"] for q in p["questions"]] for p in pairs["valid"]])
    output = dict(
        scope="development validation, checkpoint-selected; not blind generalization",
        run_sha256=sha(args.run / "result.json"),
        config_sha256=sha(args.run / "config.json"),
        prior=summarize(prior.reshape(-1, 3), targets.ravel()),
        models={},
    )
    predictions = {}
    for mode in ("mean", "attention"):
        for name, info in run[mode]["artifacts"].items():
            if sha(args.run / mode / name) != info["sha256"]:
                raise ValueError("run artifact changed")
        saved = json.loads((args.run / mode / "predictions.json").read_text())
        if saved["pairs"] != pairs["valid"] or saved["targets"] != targets.ravel().tolist():
            raise ValueError("prediction identity mismatch")
        normal = np.array(saved["probabilities"]["normal"]).reshape(-1, 2, 3)
        predictions[mode] = normal
        values = {}
        for control, probabilities in saved["probabilities"].items():
            p = np.array(probabilities).reshape(-1, 2, 3)
            score = summarize(p.reshape(-1, 3), targets.ravel())
            if score != run[mode]["controls"][control]:
                raise ValueError("metric reproduction mismatch")
            values[control] = dict(
                metrics=score,
                uncertainty=video_cluster_intervals(
                    p, targets, reference=prior if control == "normal" else normal
                ),
            )
        history = json.loads((args.run / mode / "history.json").read_text())
        selected = min(history, key=lambda r: r["valid"]["nll"])
        if len(history) != 50 or selected["epoch"] != run[mode]["best_epoch"]:
            raise ValueError("selector/history mismatch")
        output["models"][mode] = dict(
            selected_epoch=selected["epoch"],
            controls=values,
            selected_train=selected["train"],
            final_train=history[-1]["train"],
            final_valid=history[-1]["valid"],
        )
    output["attention_minus_mean"] = video_cluster_intervals(
        predictions["attention"], targets, reference=predictions["mean"]
    )
    write(args.output, output)
    print("PASS_METRIC_SELECTOR_HASH_REPRODUCTION_AND_PAIRED_INTERVALS")


if __name__ == "__main__":
    main()
