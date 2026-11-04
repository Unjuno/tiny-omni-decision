"""Read-only routing/condition diagnostic on saved development predictions."""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.audit_correspondence_validation import sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("new output required")
    run = json.loads((args.run / "result.json").read_text())
    result = dict(run_sha256=sha(args.run / "result.json"), models={})
    for mode in ("mean", "attention"):
        path = args.run / mode / "predictions.json"
        if sha(path) != run[mode]["artifacts"]["predictions.json"]["sha256"]:
            raise ValueError("prediction hash changed")
        saved = json.loads(path.read_text())
        y = np.asarray(saved["targets"]).reshape(-1, 2)
        routing = [q["target"] for p in saved["pairs"] for q in p["questions"]]
        if routing != saved["targets"]:
            raise ValueError("target routing mismatch")
        controls = {}
        normal = np.asarray(saved["probabilities"]["normal"]).reshape(-1, 2, 3)
        for name, values in saved["probabilities"].items():
            p = np.asarray(values).reshape(-1, 2, 3)
            predicted = p.argmax(-1)
            confusion = np.zeros((3, 3), dtype=int)
            np.add.at(confusion, (y.ravel(), predicted.ravel()), 1)
            controls[name] = dict(
                confusion_target_rows=confusion.tolist(),
                predicted_option_counts=np.bincount(predicted.ravel(), minlength=3).tolist(),
                condition_switches=int((predicted[:, 0] != predicted[:, 1]).sum()),
                changed_target_videos=int((y[:, 0] != y[:, 1]).sum()),
                prediction_changes_from_normal=int((predicted != normal.argmax(-1)).sum()),
                mean_condition_probability_l1=float(np.abs(p[:, 0] - p[:, 1]).sum(-1).mean()),
                max_probability_change_from_normal=float(np.abs(p - normal).max()),
            )
        result["models"][mode] = dict(predictions_sha256=sha(path), controls=controls)
    write(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
