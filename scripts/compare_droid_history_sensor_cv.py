"""Compare fixed previous-command and sensor-ablation heads on train folds."""

import argparse
import json
import platform
import runpy
import time
from pathlib import Path

import numpy as np

from scripts.audit_droid_train_temporal import digest
from scripts.compare_droid_train_temporal import temporal_features

ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")
HISTORY = ROOT / "droid-train-temporal-baselines-20261027-attempt-01/train-predictions.jsonl"
CV = ROOT / "droid-train-temporal-cv-20261027-attempt-01"
SENSOR = ROOT / "droid-fresh-next-action-features-attempt-01/sensor_sequences.npy"
LABEL = ROOT / "droid-fresh-next-action-features-attempt-01/labels.npy"
TRANSITION = ROOT / "droid-previous-command-cv-20261027-attempt-01/private-predictions.jsonl"
READOUT = CV / "pinned-readout.py"
PINNED = {
    HISTORY: "6cc30b04d1d10d880a157885102848ac882a88ad8580bd17fb9411744cb309ae",
    CV / "outer-train-predictions.jsonl": (
        "f020b2e3bcd0eb36ccc4117b2946a152cd98b5db27b9efc0e954a21214d7d6aa"
    ),
    CV / "result.json": "8b44d08afe358f5df2c37bca93b2a14998ede472acc8d031c2d2e1d11eb12867",
    READOUT: "186fad84f223748bd4670903c7b85e61aa6513eed8e311065bf58d17176d7728",
    SENSOR: "de8dc6de9afec85932b31d08d94ed58d097896e03ad1098813fd9f2c7ce372b9",
    LABEL: "372579d3a5b639aa0170684abc9863b24d10ab3dee6e3821a3a261817fab80b1",
    TRANSITION: "7a5809871bf3ad5c2f0219eab2eb1a638ab0ac57d2a2aeb06f4fa1aebd50999e",
}
PROTOCOL = Path("docs/pretrained_reuse/R2_DROID_HISTORY_SENSOR_CV_PROTOCOL_20261027.md")
REPO = Path(__file__).resolve().parents[1]
GRID = (100.0, 1000.0, 10000.0)


def inputs(sensor, previous):
    previous = np.asarray(previous, dtype=int)
    delta = temporal_features(sensor)["delta"]
    if previous.shape != (len(delta),) or np.any((previous < 0) | (previous >= 7)):
        raise ValueError("invalid previous-command values")
    history = np.eye(7, dtype=np.float64)[previous]
    return {"history": history, "history_sensor": np.concatenate((history, delta), axis=1)}


def fit_folds(arms, targets, groups, folds, api):
    n = len(targets)
    outputs = {name: np.zeros((n, 7)) for name in arms}
    seen = np.zeros(n, dtype=int)
    models, details = {}, []
    all_indices = set(range(n))
    for fold_id, fold in enumerate(folds):
        train = np.asarray(fold["train_indices"], dtype=int)
        held = np.asarray(fold["held_indices"], dtype=int)
        if (len(set(train)) != len(train) or len(set(held)) != len(held)
                or set(train) & set(held) or set(train) | set(held) != all_indices
                or set(groups[train]) & set(groups[held])):
            raise ValueError("fold overlap or invalid partition")
        seen[held] += 1
        selections = {}
        for name, x in arms.items():
            selection = api["select_l2_strength_group_cv"](
                x[train], targets[train], groups[train].tolist(), candidates=GRID,
                folds=4, seed=17, max_iterations=2000,
            )
            model = api["fit_l2_softmax"](
                x[train], targets[train], l2_strength=selection["selected_l2_strength"],
                max_iterations=2000,
            )
            outputs[name][held] = api["predict_softmax"](model, x[held])
            models[f"{name}-outer-{fold_id}"] = model
            selections[name] = selection
            print(json.dumps({"fold": fold_id, "arm": name,
                              "selected_l2": selection["selected_l2_strength"]}), flush=True)
        details.append({"fold": fold_id, "held_indices": held.tolist(),
                        "train_indices": train.tolist(), "selections": selections})
    if not np.all(seen == 1):
        raise ValueError("outer folds must hold every row out once")
    return outputs, models, details


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() or output.is_relative_to(REPO):
        raise ValueError("new external output directory required")
    if any(digest(path) != expected for path, expected in PINNED.items()):
        raise ValueError("pinned input mismatch")
    history, prior = read_jsonl(HISTORY), read_jsonl(TRANSITION)
    old_predictions = read_jsonl(CV / "outer-train-predictions.jsonl")
    if len(history) != 275 or len(prior) != 275 or len(old_predictions) != 275:
        raise ValueError("unexpected train row count")
    for a, b, c in zip(history, prior, old_predictions, strict=True):
        if any(a[key] != b[key] or a[key] != c[key] for key in ("id", "group_id", "target")):
            raise ValueError("row identity or target mismatch")
        if a["predictions"]["previous_command"] != b["previous_command"]:
            raise ValueError("previous command mismatch")
    groups = np.asarray([row["group_id"] for row in history])
    if len(set(groups)) != len(groups):
        raise ValueError("recording groups must be unique")
    previous = np.asarray([row["previous_command"] for row in prior], dtype=int)
    sensor_map = np.load(SENSOR, mmap_mode="r")
    label_map = np.load(LABEL, mmap_mode="r")
    if sensor_map.shape != (425, 8, 14) or label_map.shape != (425,):
        raise ValueError("unexpected train-first containers")
    sensor = np.array(sensor_map[:275], dtype=np.float64, copy=True)
    targets = np.array(label_map[:275], dtype=int, copy=True)
    del sensor_map, label_map
    if not np.array_equal(targets, [row["target"] for row in history]):
        raise ValueError("train label alignment mismatch")
    arms = inputs(sensor, previous)
    folds = json.loads((CV / "result.json").read_bytes())["folds"]
    api = runpy.run_path(str(READOUT))
    started = time.perf_counter()
    probabilities, models, details = fit_folds(arms, targets, groups, folds, api)
    probabilities["transition"] = np.asarray([row["probabilities"] for row in prior])
    measure = api["multiclass_probability_metrics"]
    metrics = {name: measure(targets, value) for name, value in probabilities.items()}
    subsets = {}
    for name, mask in (("stable", previous == targets), ("changed", previous != targets)):
        subsets[name] = {"count": int(mask.sum()), "metrics": {
            arm: measure(targets[mask], value[mask]) for arm, value in probabilities.items()}}
    if any(digest(path) != expected for path, expected in PINNED.items()):
        raise ValueError("pinned inputs changed during fit")
    output.mkdir(parents=True)
    for name, model in models.items():
        np.savez(output / f"{name}.npz", **model)
    prediction_rows = [{"id": row["id"], "group_id": row["group_id"],
                        "target": int(targets[i]),
                        "probabilities": {k: p[i].tolist() for k, p in probabilities.items()}}
                       for i, row in enumerate(history)]
    prediction_path = output / "private-predictions.jsonl"
    prediction_path.write_text("".join(json.dumps(r) + "\n" for r in prediction_rows),
                               encoding="utf-8")
    result = {"status": "TRAIN_CV_DIAGNOSTIC", "rows": len(history),
              "parameter_counts": {"history": 56, "history_sensor": 840},
              "metrics": metrics, "target_defined_subsets": subsets, "folds": details,
              "macro_nll_delta": (
                  metrics["history_sensor"]["macro_nll"] - metrics["history"]["macro_nll"]
              ),
              "model_hashes": {p.name: digest(p) for p in sorted(output.glob("*.npz"))},
              "predictions_sha256": digest(prediction_path),
              "source_hashes": {str(path): value for path, value in PINNED.items()},
              "protocol_sha256": digest(PROTOCOL), "script_sha256": digest(Path(__file__)),
              "wall_seconds": time.perf_counter() - started,
              "python": platform.python_version(), "numpy": np.__version__,
              "sealed_audit_loaded": False, "gpu_used": False, "product_head_created": False}
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"metrics": metrics, "changed": subsets["changed"],
                      "macro_nll_delta": result["macro_nll_delta"]}, indent=2))


if __name__ == "__main__":
    main()
