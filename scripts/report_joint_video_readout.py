"""Verify fixed readout artifacts and compare paired development predictions."""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.train_joint_video_readout import sha, write
from scripts.train_trajectory_readout import PINS, ROOT, summarize
from tiny_omni_decision.paired_video_intervals import video_cluster_intervals


def check_comparison_configs(current, reference):
    for key in (
        "seed",
        "epochs",
        "lr",
        "weight_decay",
        "selector",
        "pair_hashes",
        "checkpoint_sha256",
    ):
        if current[key] != reference[key]:
            raise ValueError(f"comparison setting mismatch: {key}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("fresh report output required")
    path = ROOT / "perception-occlusion-pairs-01/valid.json"
    if sha(path) != PINS["valid"]:
        raise ValueError("validation corpus changed")
    pairs = json.loads(path.read_text())["pairs"]
    targets = np.array([[q["target"] for q in p["questions"]] for p in pairs])
    result_path = args.run / "result.json"
    results = json.loads(result_path.read_text())
    if set(results) != {"normal", "black_pixels"}:
        raise ValueError("two fixed comparison conditions required")
    values = {}
    report = dict(
        scope="observed development validation, selected checkpoints; not blind",
        result_sha256=sha(result_path),
        models={},
    )
    for mode, result in results.items():
        for name, artifact in result["artifacts"].items():
            if sha(args.run / mode / name) != artifact["sha256"]:
                raise ValueError("saved artifact hash mismatch")
        history = json.loads((args.run / mode / "history.json").read_text())
        if [x["epoch"] for x in history] != list(range(51)):
            raise ValueError("incomplete learning curve")
        selected = min(history, key=lambda row: row["valid"]["nll"])
        if selected["epoch"] != result["best_epoch"]:
            raise ValueError("checkpoint selection mismatch")
        saved = json.loads((args.run / mode / "predictions.json").read_text())
        if saved["targets"] != targets.ravel().tolist():
            raise ValueError("prediction target alignment mismatch")
        p = np.asarray(saved["probabilities"], dtype=np.float32)
        measured = summarize(p, targets.ravel())
        if measured != selected["valid"]:
            raise ValueError("selected metric reproduction mismatch")
        values[mode] = p.reshape(len(pairs), 2, 3)
        report["models"][mode] = dict(
            initial=history[0],
            selected=selected,
            final=history[-1],
            seconds=result["seconds"],
            exact_reload=result["exact_reload"],
            trainable_parameters=result["trainable_parameters"],
        )
    report["normal_minus_black"] = video_cluster_intervals(
        values["normal"], targets, reference=values["black_pixels"], draws=10000, seed=17
    )
    if args.reference_run:
        reference = args.reference_run
        check_comparison_configs(
            json.loads((args.run / "config.json").read_text()),
            json.loads((reference / "config.json").read_text()),
        )
        reference_result = json.loads((reference / "result.json").read_text())
        reference_values = {}
        for mode in ("normal", "black_pixels"):
            prediction_path = reference / mode / "predictions.json"
            expected = reference_result[mode]["artifacts"]["predictions.json"]["sha256"]
            if sha(prediction_path) != expected:
                raise ValueError("reference prediction hash mismatch")
            saved = json.loads(prediction_path.read_text())
            if saved["targets"] != targets.ravel().tolist():
                raise ValueError("reference target alignment mismatch")
            reference_values[mode] = np.asarray(saved["probabilities"], dtype=np.float32).reshape(
                len(pairs), 2, 3
            )
        if not np.array_equal(values["black_pixels"], reference_values["black_pixels"]):
            raise ValueError("unchanged black-control prediction mismatch")
        report["black_reference_exact"] = True
        report["reference_result_sha256"] = sha(reference / "result.json")
        report["normal_minus_reference"] = video_cluster_intervals(
            values["normal"], targets, reference=reference_values["normal"], draws=10000, seed=17
        )
    write(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
