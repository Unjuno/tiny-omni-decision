"""Train-only change-weighted grouped-CV comparison for dense DROID windows."""

import argparse
import json
import runpy
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from scripts.audit_droid_train_temporal import digest
from scripts.train_droid_dense_history_sensor_cv import (
    BATCH,
    DATA,
    EPOCHS,
    LR,
    READOUT,
    READOUT_SHA,
    REPO,
    SUMMARY_SHA,
    WEIGHT_DECAY,
    group_accuracy,
    sensor_deltas,
    split_groups,
)

BASE = Path("C:/CodexArtifacts/pretrained-reuse-r2") / (
    "droid-dense-history-sensor-cv-20261027-attempt-01"
)
BASE_SUMMARY_SHA = "dc9375007f3dc759bc944b29c0e6c17bf6469431d31bdd18728cf2df2aa04c2b"
BASE_OOF_SHA = "62997f9b5da31f882013c9211636332916651757ba2c6a3c8a075c01758a0efe"
BASE_SCRIPT = Path("scripts/train_droid_dense_history_sensor_cv.py")
BASE_SCRIPT_SHA = "1383b97c0413bacf492b084f2e6c31b3ba0282661de8c9c13e8dd0224a9353e1"
PROTOCOL = Path("docs/pretrained_reuse/R2_DROID_DENSE_CHANGE_WEIGHTED_CV_PROTOCOL_20261027.md")


def training_weights(previous, targets):
    previous, targets = np.asarray(previous), np.asarray(targets)
    if previous.ndim != 1 or targets.shape != previous.shape:
        raise ValueError("unaligned training labels")
    changed = previous != targets
    n = len(changed)
    nc = int(changed.sum())
    if n == 0 or nc == 0 or nc == n:
        raise ValueError("training fold must contain stable and changed examples")
    return n / (2 * (n - nc)), n / (2 * nc)


def correct_probabilities(probabilities, previous, stable_weight, change_weight):
    p = np.asarray(probabilities, dtype=np.float64)
    previous = np.asarray(previous, dtype=np.int64)
    if (p.ndim != 2 or p.shape[1] != 7 or previous.shape != (len(p),)
            or np.any((previous < 0) | (previous >= 7)) or not np.isfinite(p).all()
            or np.any(p < 0) or not np.allclose(p.sum(axis=1), 1)
            or not np.isfinite(stable_weight) or not np.isfinite(change_weight)
            or stable_weight <= 0 or change_weight <= 0):
        raise ValueError("invalid probabilities or correction weights")
    weight = np.full(p.shape, change_weight, dtype=np.float64)
    weight[np.arange(len(p)), previous] = stable_weight
    corrected = p / weight
    totals = corrected.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("zero probability mass")
    return corrected / totals


def change_metrics(previous, targets, predictions):
    actual = previous != targets
    predicted = previous != predictions
    tp = int(np.sum(actual & predicted))
    fp = int(np.sum(~actual & predicted))
    return {"changed": int(actual.sum()), "stable": int((~actual).sum()),
            "predicted_change": int(predicted.sum()), "true_change_detected": tp,
            "changed_exact_correct": int(np.sum(actual & (predictions == targets))),
            "change_recall": tp / int(actual.sum()) if actual.any() else None,
            "change_precision": tp / (tp + fp) if tp + fp else None,
            "stable_false_change_rate": fp / int((~actual).sum()) if (~actual).any() else None}


def fit_weighted(features, targets, previous, train, fold, arm):
    stable_weight, change_weight = training_weights(previous[train], targets[train])
    model = torch.nn.Linear(features.shape[1], 7, bias=True, device="cpu")
    torch.nn.init.zeros_(model.weight)
    torch.nn.init.zeros_(model.bias)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    x = torch.from_numpy(features)
    y = torch.from_numpy(targets.astype(np.int64))
    weights = torch.from_numpy(np.where(previous != targets, change_weight, stable_weight)
                               .astype(np.float32))
    rng = np.random.default_rng(17 + fold)
    curve = []
    for epoch in range(EPOCHS):
        order = rng.permutation(train)
        loss_sum = 0.0
        for start in range(0, len(order), BATCH):
            idx = torch.from_numpy(order[start:start + BATCH])
            optimizer.zero_grad(set_to_none=True)
            losses = torch.nn.functional.cross_entropy(model(x[idx]), y[idx], reduction="none")
            loss = (losses * weights[idx]).mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("nonfinite loss")
            loss.backward()
            if any(p.grad is None or not bool(torch.isfinite(p.grad).all())
                   for p in model.parameters()):
                raise FloatingPointError("nonfinite gradient")
            optimizer.step()
            loss_sum += float(loss.detach()) * len(idx)
        curve.append(loss_sum / len(train))
        if (epoch + 1) % 10 == 0:
            print(json.dumps({"fold": fold, "arm": arm, "epoch": epoch + 1,
                              "weighted_train_loss": curve[-1]}), flush=True)
    return model, curve, stable_weight, change_weight


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() or output.is_relative_to(REPO):
        raise ValueError("new external output required")
    if shutil.disk_usage(output.parent).free < 1024**3 + 64 * 1024**2:
        raise ValueError("insufficient disk reserve")
    if (digest(DATA / "summary.json") != SUMMARY_SHA or digest(READOUT) != READOUT_SHA
            or digest(BASE / "summary.json") != BASE_SUMMARY_SHA
            or digest(BASE / "oof-probabilities.npz") != BASE_OOF_SHA
            or digest(BASE_SCRIPT) != BASE_SCRIPT_SHA):
        raise ValueError("pinned input mismatch")
    data_summary = json.loads((DATA / "summary.json").read_bytes())
    for name, expected in data_summary["output_hashes"].items():
        if digest(DATA / name) != expected:
            raise ValueError(f"dataset hash mismatch: {name}")
    groups = json.loads((DATA / "private-groups.json").read_bytes())
    sensor = np.load(DATA / "sensor.npy", mmap_mode="r")
    previous = np.asarray(np.load(DATA / "previous.npy", mmap_mode="r"), dtype=np.int64)
    targets = np.asarray(np.load(DATA / "target.npy", mmap_mode="r"), dtype=np.int64)
    group = np.asarray(np.load(DATA / "group.npy", mmap_mode="r"), dtype=np.int64)
    if (sensor.shape != (80998, 8, 14) or previous.shape != (80998,)
            or targets.shape != (80998,) or group.shape != (80998,)
            or np.any(previous >= 7) or np.any(targets >= 7)):
        raise ValueError("unexpected data shape or labels")
    fold_rows = split_groups(group, groups)
    history = np.eye(7, dtype=np.float32)[previous]
    delta = sensor_deltas(sensor)
    with np.load(BASE / "oof-probabilities.npz", allow_pickle=False) as source:
        probabilities = {f"natural_{key}": np.array(source[key], copy=True)
                         for key in ("history", "history_sensor", "transition")}
    for arm in ("history", "history_sensor"):
        probabilities[f"weighted_{arm}"] = np.zeros((len(targets), 7), dtype=np.float32)
        probabilities[f"corrected_{arm}"] = np.zeros((len(targets), 7), dtype=np.float32)
    torch.set_num_threads(4)
    output.mkdir(parents=True)
    visits = np.zeros(len(targets), dtype=np.uint8)
    curves, fold_details = {}, []
    start_time = time.perf_counter()
    for fold in range(5):
        train, held = np.flatnonzero(fold_rows != fold), np.flatnonzero(fold_rows == fold)
        if set(group[train]) & set(group[held]):
            raise ValueError("recording crossed folds")
        visits[held] += 1
        mean = delta[train].mean(axis=0, dtype=np.float64).astype(np.float32)
        scale = delta[train].std(axis=0, dtype=np.float64).astype(np.float32)
        scale[scale < 1e-6] = 1
        sensor_features = np.clip((delta - mean) / scale, -20, 20)
        arms = {"history": history,
                "history_sensor": np.concatenate((history, sensor_features), axis=1)}
        detail = {"fold": fold, "train_windows": len(train), "held_windows": len(held),
                  "train_changed": int(np.sum(previous[train] != targets[train]))}
        for name, features in arms.items():
            model, curve, ws, wc = fit_weighted(features, targets, previous, train, fold, name)
            with torch.inference_mode():
                raw = torch.softmax(model(torch.from_numpy(features[held])), dim=1).numpy()
            probabilities[f"weighted_{name}"][held] = raw
            probabilities[f"corrected_{name}"][held] = correct_probabilities(
                raw, previous[held], ws, wc
            )
            np.savez(output / f"{name}-fold-{fold}.npz", weight=model.weight.detach().numpy(),
                     bias=model.bias.detach().numpy(), mean=mean, scale=scale,
                     stable_weight=ws, change_weight=wc)
            curves[f"{name}-fold-{fold}"] = curve
            detail["stable_weight"], detail["change_weight"] = ws, wc
        fold_details.append(detail)
    if not np.all(visits == 1):
        raise ValueError("held windows not covered exactly once")
    measure = runpy.run_path(str(READOUT))["multiclass_probability_metrics"]
    metrics, changes = {}, {}
    for name, p in probabilities.items():
        metrics[name] = measure(targets, p)
        pred = p.argmax(axis=1)
        metrics[name]["mean_recording_accuracy"] = group_accuracy(targets, pred, group)
        changes[name] = change_metrics(previous, targets, pred)
    changed = previous != targets
    changed_metrics = {name: measure(targets[changed], p[changed])
                       for name, p in probabilities.items()}
    if any(digest(DATA / name) != expected
           for name, expected in data_summary["output_hashes"].items()):
        raise ValueError("dataset changed during training")
    np.savez(output / "oof-probabilities.npz", **probabilities)
    result = {"status": "TRAIN_GROUP_CV_FOLLOWUP_NOT_BLIND", "rows": len(targets),
              "metrics": metrics, "change_detection": changes, "changed_metrics": changed_metrics,
              "folds": fold_details, "training_loss_curves": curves,
              "artifact_hashes": {p.name: digest(p) for p in sorted(output.glob("*.npz"))},
              "dataset_summary_sha256": SUMMARY_SHA, "baseline_summary_sha256": BASE_SUMMARY_SHA,
              "baseline_oof_sha256": BASE_OOF_SHA, "readout_sha256": READOUT_SHA,
              "baseline_script_sha256": BASE_SCRIPT_SHA,
              "protocol_sha256": digest(PROTOCOL), "script_sha256": digest(Path(__file__)),
              "epochs": EPOCHS, "batch_size": BATCH, "learning_rate": LR,
              "weight_decay": WEIGHT_DECAY, "wall_seconds": time.perf_counter() - start_time,
              "torch_version": torch.__version__, "gpu_used": False,
              "sealed_audit_loaded": False, "product_head_created": False}
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"metrics": metrics, "change_detection": changes}, indent=2))


if __name__ == "__main__":
    main()
