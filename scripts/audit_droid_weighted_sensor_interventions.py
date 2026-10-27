"""Replay fixed weighted DROID heads under recording-preserving sensor interventions."""

import argparse
import json
import runpy
import shutil
import time
from pathlib import Path

import numpy as np

from scripts.audit_droid_train_temporal import digest
from scripts.train_droid_dense_change_weighted_cv import (
    change_metrics,
    correct_probabilities,
    training_weights,
)
from scripts.train_droid_dense_history_sensor_cv import (
    DATA,
    READOUT,
    READOUT_SHA,
    REPO,
    SUMMARY_SHA,
    group_accuracy,
    sensor_deltas,
    split_groups,
)

SOURCE = Path("C:/CodexArtifacts/pretrained-reuse-r2") / (
    "droid-dense-change-weighted-cv-20261027-attempt-02"
)
SOURCE_SUMMARY_SHA = "53fe441c36b15bbb6b3b4f9570775adee62373f365c99278c314c819896c3359"
PROTOCOL = Path("docs/pretrained_reuse/R2_DROID_WEIGHTED_SENSOR_INTERVENTION_PROTOCOL_20261027.md")


def within_recording_shift(groups, count):
    donors = np.full(count, -1, dtype=np.int64)
    for item in groups:
        start, end = int(item["start"]), int(item["end"])
        n = end - start
        if n < 24 or start < 0 or end > count or np.any(donors[start:end] != -1):
            raise ValueError("invalid or overlapping recording windows")
        donors[start:end] = start + (np.arange(n) + n // 2) % n
    if np.any(donors < 0) or np.any(donors == np.arange(count)):
        raise ValueError("donor mapping incomplete or self-aligned")
    return donors


def group_bootstrap(targets, aligned, control, group, seed=17, replicates=2000):
    rows = np.arange(len(targets))
    nll_delta = -np.log(np.clip(aligned[rows, targets], 1e-15, 1)) + np.log(
        np.clip(control[rows, targets], 1e-15, 1)
    )
    accuracy_delta = (aligned.argmax(axis=1) == targets).astype(float) - (
        control.argmax(axis=1) == targets
    ).astype(float)
    totals = np.bincount(group, minlength=275).astype(float)
    if np.any(totals == 0):
        raise ValueError("missing recording group")
    loss_sums = np.bincount(group, weights=nll_delta, minlength=275)
    acc_sums = np.bincount(group, weights=accuracy_delta, minlength=275)
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, 275, size=(replicates, 275))
    denominators = totals[samples].sum(axis=1)
    loss = loss_sums[samples].sum(axis=1) / denominators
    accuracy = acc_sums[samples].sum(axis=1) / denominators
    return {"groups": 275, "replicates": replicates, "seed": seed,
            "nll_aligned_minus_control": float(nll_delta.mean()),
            "nll_95_percentile_interval": np.quantile(loss, [0.025, 0.975]).tolist(),
            "accuracy_aligned_minus_control": float(accuracy_delta.mean()),
            "accuracy_95_percentile_interval": np.quantile(accuracy, [0.025, 0.975]).tolist()}


def softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    result = np.exp(shifted)
    return result / result.sum(axis=1, keepdims=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists() or out.is_relative_to(REPO):
        raise ValueError("new external output required")
    if shutil.disk_usage(out.parent).free < 1024**3 + 64 * 1024**2:
        raise ValueError("insufficient disk reserve")
    if (digest(DATA / "summary.json") != SUMMARY_SHA or digest(READOUT) != READOUT_SHA
            or digest(SOURCE / "summary.json") != SOURCE_SUMMARY_SHA):
        raise ValueError("pinned summary or readout mismatch")
    dataset = json.loads((DATA / "summary.json").read_bytes())
    source = json.loads((SOURCE / "summary.json").read_bytes())
    if (digest(Path("scripts/train_droid_dense_change_weighted_cv.py")) != source["script_sha256"]
            or digest(Path("scripts/train_droid_dense_history_sensor_cv.py"))
            != source["baseline_script_sha256"]):
        raise ValueError("pinned helper code mismatch")
    for name, expected in dataset["output_hashes"].items():
        if digest(DATA / name) != expected:
            raise ValueError(f"dataset hash mismatch: {name}")
    for name, expected in source["artifact_hashes"].items():
        if digest(SOURCE / name) != expected:
            raise ValueError(f"saved model hash mismatch: {name}")
    groups = json.loads((DATA / "private-groups.json").read_bytes())
    sensor = np.load(DATA / "sensor.npy", mmap_mode="r")
    previous = np.asarray(np.load(DATA / "previous.npy", mmap_mode="r"), dtype=np.int64)
    target = np.asarray(np.load(DATA / "target.npy", mmap_mode="r"), dtype=np.int64)
    group = np.asarray(np.load(DATA / "group.npy", mmap_mode="r"), dtype=np.int64)
    if (sensor.shape != (80998, 8, 14) or previous.shape != (80998,)
            or target.shape != (80998,) or group.shape != (80998,)):
        raise ValueError("unexpected dataset shapes")
    fold_rows = split_groups(group, groups)
    donors = within_recording_shift(groups, len(target))
    if not np.array_equal(group, group[donors]) or not np.array_equal(fold_rows, fold_rows[donors]):
        raise ValueError("shift crosses recording or fold")
    delta = sensor_deltas(sensor)
    output = {f"{variant}_{condition}": np.zeros((len(target), 7), dtype=np.float32)
              for variant in ("weighted", "corrected")
              for condition in ("aligned", "mean_sensor", "shifted_sensor")}
    prior = np.load(SOURCE / "oof-probabilities.npz", allow_pickle=False)
    replay_max = 0.0
    start_time = time.perf_counter()
    for fold in range(5):
        train, held = np.flatnonzero(fold_rows != fold), np.flatnonzero(fold_rows == fold)
        with np.load(SOURCE / f"history_sensor-fold-{fold}.npz", allow_pickle=False) as f:
            model = {name: f[name] for name in f.files}
        mean = delta[train].mean(axis=0, dtype=np.float64).astype(np.float32)
        scale = delta[train].std(axis=0, dtype=np.float64).astype(np.float32)
        scale[scale < 1e-6] = 1
        ws, wc = training_weights(previous[train], target[train])
        if (not np.array_equal(mean, model["mean"]) or not np.array_equal(scale, model["scale"])
                or float(model["stable_weight"]) != ws or float(model["change_weight"]) != wc
                or model["weight"].shape != (7, 119) or model["bias"].shape != (7,)):
            raise ValueError("saved model or training-only normalizer mismatch")
        weight, bias = model["weight"], model["bias"]
        history_logits = weight[:, previous[held]].T + bias
        for name, values in (("aligned", delta[held]), ("shifted_sensor", delta[donors[held]])):
            z = np.clip((values - mean) / scale, -20, 20)
            output[f"weighted_{name}"][held] = softmax(history_logits + z @ weight[:, 7:].T)
        output["weighted_mean_sensor"][held] = softmax(history_logits)
        for name in ("aligned", "mean_sensor", "shifted_sensor"):
            output[f"corrected_{name}"][held] = correct_probabilities(
                output[f"weighted_{name}"][held], previous[held], ws, wc
            )
        for variant in ("weighted", "corrected"):
            expected = prior[f"{variant}_history_sensor"][held]
            difference = np.abs(output[f"{variant}_aligned"][held] - expected)
            replay_max = max(replay_max, float(np.max(difference)))
    if replay_max > 3e-6:
        raise ValueError("aligned replay did not match pinned OOF probabilities")
    measure = runpy.run_path(str(READOUT))["multiclass_probability_metrics"]
    metrics, changes, comparisons = {}, {}, {}
    for name, p in output.items():
        metrics[name] = measure(target, p)
        pred = p.argmax(axis=1)
        metrics[name]["mean_recording_accuracy"] = group_accuracy(target, pred, group)
        changes[name] = change_metrics(previous, target, pred)
    for variant in ("weighted", "corrected"):
        for condition in ("mean_sensor", "shifted_sensor"):
            comparisons[f"{variant}_aligned_minus_{condition}"] = group_bootstrap(
                target, output[f"{variant}_aligned"], output[f"{variant}_{condition}"], group
            )
    out.mkdir(parents=True)
    np.savez(out / "oof-interventions.npz", **output)
    result = {"status": "TRAIN_GROUP_CV_INTERVENTION_NOT_BLIND", "rows": len(target),
              "metrics": metrics, "change_detection": changes, "paired_bootstrap": comparisons,
              "aligned_replay_max_abs_difference": replay_max,
              "output_hashes": {"oof-interventions.npz": digest(out / "oof-interventions.npz")},
              "dataset_summary_sha256": SUMMARY_SHA, "source_summary_sha256": SOURCE_SUMMARY_SHA,
              "readout_sha256": READOUT_SHA, "protocol_sha256": digest(PROTOCOL),
              "script_sha256": digest(Path(__file__)),
              "wall_seconds": time.perf_counter() - start_time,
              "training_performed": False, "gpu_used": False, "sealed_audit_loaded": False}
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"metrics": metrics, "change_detection": changes,
                      "paired_bootstrap": comparisons}, indent=2))


if __name__ == "__main__":
    main()
