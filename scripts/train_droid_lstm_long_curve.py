"""Fixed 200-epoch CPU learning curve on previously observed DROID train folds."""

import argparse
import json
import runpy
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from scripts.audit_droid_train_temporal import digest
from scripts.audit_droid_weighted_sensor_interventions import group_bootstrap
from scripts.train_droid_dense_change_weighted_cv import (
    change_metrics,
    correct_probabilities,
    training_row_weights,
)
from scripts.train_droid_dense_history_sensor_cv import (
    BATCH,
    DATA,
    LR,
    READOUT,
    READOUT_SHA,
    REPO,
    SUMMARY_SHA,
    WEIGHT_DECAY,
    group_accuracy,
    split_groups,
)
from scripts.train_droid_dense_lstm_cv import SmallTemporalDecision

ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")
PRIOR = ROOT / "droid-dense-lstm-cv-20261027-attempt-01"
PRIOR_SUMMARY_SHA = "9bd1afbc137aa0d66903bb695bffcf6c37eecb0aac5a679c403953b31c859c9c"
PRIOR_OOF_SHA = "faf346ae125a87268f23d0280dc2dcf9b62b00537c96ad5caed9353f0964acec"
PRIOR_SCRIPT_SHA = "7da0f10c03cb2a432b969778be414a9357e4d776ccb0eb14d8df4175c2f6b33a"
PROTOCOL = Path("docs/pretrained_reuse/R2_DROID_LSTM_LONG_CURVE_PROTOCOL_20261027.md")
MILESTONES = (50, 100, 150, 200)
MAX_EPOCHS = 200


def score(model, sensor, history, indices):
    model.eval()
    blocks = []
    with torch.inference_mode():
        for start in range(0, len(indices), BATCH):
            idx = indices[start:start + BATCH]
            logits = model(torch.from_numpy(sensor[idx]), torch.from_numpy(history[idx]))
            blocks.append(torch.softmax(logits, dim=1).numpy())
    model.train()
    return np.concatenate(blocks, axis=0)


def train_fold(
    sensor, history, targets, previous, train, held, fold, output, baseline, mean, scale
):
    weights, stable, changed = training_row_weights(previous, targets, train)
    torch.manual_seed(17 + fold)
    model = SmallTemporalDecision().cpu()
    if sum(p.numel() for p in model.parameters()) != 880:
        raise ValueError("unexpected parameter count")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    x, h = torch.from_numpy(sensor), torch.from_numpy(history)
    y, w = torch.from_numpy(targets.astype(np.int64)), torch.from_numpy(weights)
    rng = np.random.default_rng(17 + fold)
    curve, predictions = [], {}
    for epoch in range(1, MAX_EPOCHS + 1):
        if shutil.disk_usage(output).free < 1024**3 + 8 * 1024**2:
            raise OSError("disk reserve reached during training")
        order = rng.permutation(train)
        loss_sum = 0.0
        for start in range(0, len(order), BATCH):
            idx = torch.from_numpy(order[start:start + BATCH])
            optimizer.zero_grad(set_to_none=True)
            losses = torch.nn.functional.cross_entropy(
                model(x[idx], h[idx]), y[idx], reduction="none"
            )
            loss = (losses * w[idx]).mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("nonfinite training loss")
            loss.backward()
            if any(p.grad is None or not bool(torch.isfinite(p.grad).all())
                   for p in model.parameters()):
                raise FloatingPointError("nonfinite training gradient")
            optimizer.step()
            loss_sum += float(loss.detach()) * len(idx)
        curve.append(loss_sum / len(train))
        if epoch % 25 == 0:
            print(json.dumps({"fold": fold, "epoch": epoch, "train_loss": curve[-1]}), flush=True)
        if epoch in MILESTONES:
            raw = score(model, sensor, history, held)
            if epoch == 50 and np.max(np.abs(raw - baseline)) > 3e-6:
                raise ValueError("50-epoch replay mismatch; abort longer comparison")
            corrected = correct_probabilities(raw, previous[held], stable, changed)
            predictions[epoch] = (raw, corrected)
            state = {
                key: value.detach().numpy().copy() for key, value in model.state_dict().items()
            }
            np.savez(output / f"lstm-fold-{fold}-epoch-{epoch}.npz", **state,
                     mean=mean, scale=scale, stable_weight=stable, change_weight=changed)
            np.savez(output / f"held-fold-{fold}-epoch-{epoch}.npz", raw=raw,
                     corrected=corrected, held_indices=held)
    return predictions, curve, stable, changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() or output.is_relative_to(REPO):
        raise ValueError("new external output required")
    if shutil.disk_usage(output.parent).free < 1024**3 + 128 * 1024**2:
        raise ValueError("insufficient disk reserve for learning curve")
    if (digest(DATA / "summary.json") != SUMMARY_SHA or digest(READOUT) != READOUT_SHA
            or digest(PRIOR / "summary.json") != PRIOR_SUMMARY_SHA
            or digest(PRIOR / "oof-probabilities.npz") != PRIOR_OOF_SHA
            or digest(Path("scripts/train_droid_dense_lstm_cv.py")) != PRIOR_SCRIPT_SHA):
        raise ValueError("pinned dataset, prior run or code mismatch")
    dataset = json.loads((DATA / "summary.json").read_bytes())
    prior_summary = json.loads((PRIOR / "summary.json").read_bytes())
    dependencies = {
        "scripts/train_droid_dense_change_weighted_cv.py": prior_summary["weighted_summary_sha256"],
    }
    # Follow the prior run's pinned helper hashes rather than the current environment defaults.
    weighted_summary = json.loads((ROOT / "droid-dense-change-weighted-cv-20261027-attempt-02"
                                   / "summary.json").read_bytes())
    if digest(ROOT / "droid-dense-change-weighted-cv-20261027-attempt-02" / "summary.json") != (
        dependencies["scripts/train_droid_dense_change_weighted_cv.py"]
    ):
        raise ValueError("prior weighted summary mismatch")
    for path, expected in (
        (Path("scripts/train_droid_dense_change_weighted_cv.py"),
         weighted_summary["script_sha256"]),
        (Path("scripts/train_droid_dense_history_sensor_cv.py"),
         weighted_summary["baseline_script_sha256"]),
        (Path("scripts/audit_droid_weighted_sensor_interventions.py"),
         prior_summary["bootstrap_script_sha256"]),
    ):
        if digest(path) != expected:
            raise ValueError("pinned helper code mismatch")
    for name, expected in dataset["output_hashes"].items():
        if digest(DATA / name) != expected:
            raise ValueError("dataset output hash mismatch")
    groups = json.loads((DATA / "private-groups.json").read_bytes())
    sensor = np.load(DATA / "sensor.npy", mmap_mode="r")
    previous = np.asarray(np.load(DATA / "previous.npy", mmap_mode="r"), dtype=np.int64)
    targets = np.asarray(np.load(DATA / "target.npy", mmap_mode="r"), dtype=np.int64)
    group = np.asarray(np.load(DATA / "group.npy", mmap_mode="r"), dtype=np.int64)
    if (sensor.shape != (80998, 8, 14) or previous.shape != (80998,)
            or targets.shape != (80998,) or group.shape != (80998,)
            or np.any(previous >= 7) or np.any(targets >= 7) or not np.isfinite(sensor).all()):
        raise ValueError("unexpected dataset shape or values")
    fold_rows = split_groups(group, groups)
    history = np.eye(7, dtype=np.float32)[previous]
    with np.load(PRIOR / "oof-probabilities.npz", allow_pickle=False) as file:
        prior = np.array(file["lstm_raw"], copy=True)
    probabilities = {f"{mode}_{epoch}": np.zeros((len(targets), 7), dtype=np.float32)
                     for mode in ("raw", "corrected") for epoch in MILESTONES}
    torch.set_num_threads(4)
    output.mkdir(parents=True)
    visits = np.zeros(len(targets), dtype=np.uint8)
    curves, fold_details = {}, []
    started = time.perf_counter()
    for fold in range(5):
        train, held = np.flatnonzero(fold_rows != fold), np.flatnonzero(fold_rows == fold)
        if set(group[train]) & set(group[held]):
            raise ValueError("recording crossed folds")
        visits[held] += 1
        mean = sensor[train].mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
        scale = sensor[train].std(axis=(0, 1), dtype=np.float64).astype(np.float32)
        scale[scale < 1e-6] = 1
        scaled = np.clip((np.asarray(sensor) - mean) / scale, -20, 20).astype(np.float32)
        results, curve, ws, wc = train_fold(
            scaled, history, targets, previous, train, held, fold, output, prior[held], mean, scale
        )
        for epoch in MILESTONES:
            raw, corrected = results[epoch]
            probabilities[f"raw_{epoch}"][held] = raw
            probabilities[f"corrected_{epoch}"][held] = corrected
        curves[str(fold)] = curve
        fold_details.append({"fold": fold, "train_windows": len(train), "held_windows": len(held),
                             "train_groups": len(set(group[train])),
                             "held_groups": len(set(group[held])),
                             "stable_weight": ws, "change_weight": wc})
    if not np.all(visits == 1):
        raise ValueError("held windows not covered exactly once")
    measure = runpy.run_path(str(READOUT))["multiclass_probability_metrics"]
    metrics, changes = {}, {}
    for name, value in probabilities.items():
        metrics[name] = measure(targets, value)
        pred = value.argmax(axis=1)
        metrics[name]["mean_recording_accuracy"] = group_accuracy(targets, pred, group)
        changes[name] = change_metrics(previous, targets, pred)
    bootstrap = {mode: group_bootstrap(targets, probabilities[f"{mode}_200"],
                                       probabilities[f"{mode}_50"], group)
                 for mode in ("raw", "corrected")}
    if any(digest(DATA / name) != expected for name, expected in dataset["output_hashes"].items()):
        raise ValueError("dataset changed during training")
    np.savez(output / "oof-probabilities.npz", **probabilities)
    result = {"status": "TRAIN_GROUP_CV_LONG_CURVE_NOT_BLIND", "rows": len(targets),
              "metrics": metrics, "change_detection": changes, "paired_bootstrap": bootstrap,
              "folds": fold_details, "training_loss_curves": curves, "milestones": MILESTONES,
              "parameter_count": 880, "batch_size": BATCH, "learning_rate": LR,
              "weight_decay": WEIGHT_DECAY, "artifact_hashes": {
                  p.name: digest(p) for p in sorted(output.glob("*.npz"))},
              "dataset_summary_sha256": SUMMARY_SHA, "prior_summary_sha256": PRIOR_SUMMARY_SHA,
              "prior_oof_sha256": PRIOR_OOF_SHA, "prior_script_sha256": PRIOR_SCRIPT_SHA,
              "protocol_sha256": digest(PROTOCOL), "script_sha256": digest(Path(__file__)),
              "wall_seconds": time.perf_counter() - started, "torch_version": torch.__version__,
              "mkldnn_enabled": torch.backends.mkldnn.enabled, "gpu_used": False,
              "exact_resume_supported": False, "sealed_audit_loaded": False,
              "product_head_created": False}
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"metrics": metrics, "change_detection": changes,
                      "paired_bootstrap": bootstrap}, indent=2))


if __name__ == "__main__":
    main()
