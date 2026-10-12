"""Recompute selected predictions and paired development uncertainty."""

import argparse
import collections
import json
from pathlib import Path

import numpy as np

from scripts.audit_correspondence_validation import sha, write
from scripts.train_trajectory_readout import PINS, ROOT, summarize
from tiny_omni_decision.paired_video_intervals import video_cluster_intervals


def aligned_reference(saved, pairs, targets):
    if saved["pairs"] != pairs or saved["targets"] != list(targets):
        raise ValueError("reference prediction identity mismatch")
    values = np.asarray(saved["probabilities"]["normal"])
    return values.reshape(len(pairs), 2, -1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("fresh output required")
    run = json.loads((args.run / "result.json").read_text())
    config = json.loads((args.run / "config.json").read_text())
    if set(run) != {"attention"} or config["epochs"] != 50 or config["seed"] != 17:
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
    for mode in ("attention",):
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
    if args.reference is not None:
        reference_result = json.loads((args.reference / "result.json").read_text())
        reference_config = json.loads((args.reference / "config.json").read_text())
        for key in ("seed", "epochs", "learning_rate", "weight_decay", "selector", "pair_hashes"):
            if reference_config[key] != config[key]:
                raise ValueError(f"reference training policy differs: {key}")
        output["reference_result_sha256"] = sha(args.reference / "result.json")
        output["candidate_minus_reference"] = {}
        for mode in ("mean", "attention"):
            path = args.reference / mode / "predictions.json"
            if sha(path) != reference_result[mode]["artifacts"]["predictions.json"]["sha256"]:
                raise ValueError("reference predictions changed")
            reference = aligned_reference(
                json.loads(path.read_text()), pairs["valid"], targets.ravel().tolist()
            )
            output["candidate_minus_reference"][mode] = video_cluster_intervals(
                predictions["attention"], targets, reference=reference
            )
    write(args.output, output)
    print("PASS_METRIC_SELECTOR_HASH_REPRODUCTION_AND_PAIRED_INTERVALS")


if __name__ == "__main__":
    main()
