"""Fixed CPU grouped-CV ablation on complete DROID train windows."""

import argparse
import json
import runpy
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from scripts.audit_droid_train_temporal import digest

ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")
DATA = ROOT / "droid-dense-train-windows-20261027-attempt-01"
READOUT = ROOT / "droid-train-temporal-cv-20261027-attempt-01/pinned-readout.py"
SUMMARY_SHA = "31abc71b9d31f69e906c6e98ee712c82384dc90bbd5ab1a9a6122b0c12953982"
READOUT_SHA = "186fad84f223748bd4670903c7b85e61aa6513eed8e311065bf58d17176d7728"
PROTOCOL = Path("docs/pretrained_reuse/R2_DROID_DENSE_HISTORY_SENSOR_CV_PROTOCOL_20261027.md")
REPO = Path(__file__).resolve().parents[1]
EPOCHS, BATCH, LR, WEIGHT_DECAY = 50, 2048, 0.003, 0.01


def split_groups(group, group_info):
    if len(group_info) != 275 or len({x["group_id"] for x in group_info}) != 275:
        raise ValueError("invalid recording group index")
    if np.any(group >= len(group_info)) or len(group) != 80998:
        raise ValueError("invalid group references")
    for i, item in enumerate(group_info):
        start, end = item["start"], item["end"]
        if not (0 <= start < end <= len(group)) or not np.all(group[start:end] == i):
            raise ValueError("group offsets disagree with rows")
    assignments = np.asarray([x["fold"] for x in group_info], dtype=np.int64)
    if set(assignments) != set(range(5)):
        raise ValueError("unexpected fold assignments")
    return assignments[group]


def sensor_deltas(sensor):
    x = np.asarray(sensor, dtype=np.float32)
    if x.ndim != 3 or x.shape[1:] != (8, 14) or not np.isfinite(x).all():
        raise ValueError("expected finite sensor [N,8,14]")
    return np.concatenate((x[:, -1], np.diff(x, axis=1).reshape(len(x), 98)), axis=1)


def fit_arm(features, targets, train, fold_id, arm_name):
    model = torch.nn.Linear(features.shape[1], 7, bias=True, device="cpu")
    torch.nn.init.zeros_(model.weight)
    torch.nn.init.zeros_(model.bias)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    x, y = torch.from_numpy(features), torch.from_numpy(targets.astype(np.int64))
    rng = np.random.default_rng(17 + fold_id)
    curve = []
    for epoch in range(EPOCHS):
        order = rng.permutation(train)
        loss_sum = 0.0
        for start in range(0, len(order), BATCH):
            indices = torch.from_numpy(order[start:start + BATCH])
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(x[indices]), y[indices])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("nonfinite training loss")
            loss.backward()
            if any(p.grad is None or not bool(torch.isfinite(p.grad).all())
                   for p in model.parameters()):
                raise FloatingPointError("nonfinite training gradient")
            optimizer.step()
            loss_sum += float(loss.detach()) * len(indices)
        curve.append(loss_sum / len(train))
        if (epoch + 1) % 10 == 0:
            print(json.dumps({"fold": fold_id, "arm": arm_name, "epoch": epoch + 1,
                              "train_loss": curve[-1]}), flush=True)
    return model, curve


def group_accuracy(targets, predictions, group):
    counts = np.bincount(group, minlength=275)
    correct = np.bincount(group, weights=(predictions == targets), minlength=275)
    if np.any(counts == 0):
        raise ValueError("missing group in evaluation")
    return float(np.mean(correct / counts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() or output.is_relative_to(REPO):
        raise ValueError("new external output required")
    if shutil.disk_usage(output.parent).free < 1024**3 + 64 * 1024**2:
        raise ValueError("insufficient disk reserve")
    if digest(DATA / "summary.json") != SUMMARY_SHA or digest(READOUT) != READOUT_SHA:
        raise ValueError("pinned metadata or readout mismatch")
    summary = json.loads((DATA / "summary.json").read_bytes())
    for name, expected in summary["output_hashes"].items():
        if digest(DATA / name) != expected:
            raise ValueError(f"dataset hash mismatch: {name}")
    groups = json.loads((DATA / "private-groups.json").read_bytes())
    sensor = np.load(DATA / "sensor.npy", mmap_mode="r")
    previous = np.load(DATA / "previous.npy", mmap_mode="r")
    targets = np.load(DATA / "target.npy", mmap_mode="r")
    group = np.load(DATA / "group.npy", mmap_mode="r")
    if (sensor.shape != (80998, 8, 14) or previous.shape != (80998,)
            or targets.shape != (80998,) or group.shape != (80998,)
            or np.any(previous >= 7) or np.any(targets >= 7)):
        raise ValueError("unexpected dataset shape or labels")
    fold_per_row = split_groups(group, groups)
    history = np.eye(7, dtype=np.float32)[previous]
    delta = sensor_deltas(sensor)
    previous = np.asarray(previous, dtype=np.int64)
    targets = np.asarray(targets, dtype=np.int64)
    group = np.asarray(group, dtype=np.int64)
    api = runpy.run_path(str(READOUT))
    measure = api["multiclass_probability_metrics"]
    torch.set_num_threads(4)
    output.mkdir(parents=True)
    probabilities = {name: np.zeros((len(targets), 7), dtype=np.float32)
                     for name in ("history", "history_sensor", "transition")}
    visits = np.zeros(len(targets), dtype=np.uint8)
    curves, fold_details = {}, []
    started = time.perf_counter()
    for fold in range(5):
        train = np.flatnonzero(fold_per_row != fold)
        held = np.flatnonzero(fold_per_row == fold)
        if set(group[train]) & set(group[held]):
            raise ValueError("recording crossed folds")
        visits[held] += 1
        mean = delta[train].mean(axis=0, dtype=np.float64).astype(np.float32)
        scale = delta[train].std(axis=0, dtype=np.float64).astype(np.float32)
        scale[scale < 1e-6] = 1.0
        scaled = np.clip((delta - mean) / scale, -20, 20)
        arms = {"history": history, "history_sensor": np.concatenate((history, scaled), axis=1)}
        counts = np.ones((7, 7), dtype=np.float64)
        np.add.at(counts, (previous[train], targets[train]), 1)
        counts /= counts.sum(axis=1, keepdims=True)
        probabilities["transition"][held] = counts[previous[held]]
        detail = {"fold": fold, "train_windows": len(train), "held_windows": len(held),
                  "train_groups": len(set(group[train])), "held_groups": len(set(group[held]))}
        for name, features in arms.items():
            model, curve = fit_arm(features, targets, train, fold, name)
            with torch.inference_mode():
                probs = torch.softmax(model(torch.from_numpy(features[held])), dim=1).numpy()
            probabilities[name][held] = probs
            np.savez(output / f"{name}-fold-{fold}.npz", weight=model.weight.detach().numpy(),
                     bias=model.bias.detach().numpy(), mean=mean, scale=scale)
            curves[f"{name}-fold-{fold}"] = curve
            print(json.dumps({"fold": fold, "arm": name, "train_loss_first": curve[0],
                              "train_loss_last": curve[-1]}), flush=True)
        fold_details.append(detail)
    if not np.all(visits == 1):
        raise ValueError("held windows are not covered exactly once")
    metrics = {name: measure(targets, value) for name, value in probabilities.items()}
    for name, value in probabilities.items():
        metrics[name]["mean_recording_accuracy"] = group_accuracy(
            targets, value.argmax(axis=1), group
        )
    subsets = {}
    for name, mask in (("stable", previous == targets), ("changed", previous != targets)):
        subsets[name] = {"count": int(mask.sum()), "metrics": {
            arm: measure(targets[mask], value[mask]) for arm, value in probabilities.items()}}
    if digest(DATA / "summary.json") != SUMMARY_SHA:
        raise ValueError("dataset summary changed during run")
    np.savez(output / "oof-probabilities.npz", **probabilities)
    result = {"status": "TRAIN_GROUP_CV_DIAGNOSTIC", "rows": len(targets),
              "metrics": metrics, "target_defined_subsets": subsets,
              "macro_nll_delta": (metrics["history_sensor"]["macro_nll"]
                                  - metrics["history"]["macro_nll"]),
              "folds": fold_details, "training_loss_curves": curves,
              "epochs": EPOCHS, "batch_size": BATCH, "learning_rate": LR,
              "weight_decay": WEIGHT_DECAY, "parameter_counts": {"history": 56,
                                                                     "history_sensor": 840},
              "artifact_hashes": {p.name: digest(p) for p in sorted(output.glob("*.npz"))},
              "dataset_summary_sha256": SUMMARY_SHA, "readout_sha256": READOUT_SHA,
              "protocol_sha256": digest(PROTOCOL), "script_sha256": digest(Path(__file__)),
              "wall_seconds": time.perf_counter() - started, "torch_version": torch.__version__,
              "gpu_used": False, "sealed_audit_loaded": False, "product_head_created": False}
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"metrics": metrics, "changed": subsets["changed"],
                      "macro_nll_delta": result["macro_nll_delta"]}, indent=2))


if __name__ == "__main__":
    main()
