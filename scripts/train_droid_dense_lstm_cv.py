"""Fixed CPU LSTM grouped-CV follow-up on past-only DROID train windows."""

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
    BASE_OOF_SHA,
    BASE_SUMMARY_SHA,
    change_metrics,
    correct_probabilities,
    training_row_weights,
)
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
    split_groups,
)

ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")
BASE = ROOT / "droid-dense-change-weighted-cv-20261027-attempt-02"
BASE_SUMMARY = "53fe441c36b15bbb6b3b4f9570775adee62373f365c99278c314c819896c3359"
BASE_OOF = "20f08c39df29716894501dd5c7f7081d9d09177a5b8c2a834093d9fe30739c85"
PROTOCOL = Path("docs/pretrained_reuse/R2_DROID_DENSE_LSTM_CV_PROTOCOL_20261027.md")
BOOTSTRAP_SHA = "7f982b0ac4a0fb5f1ecf750695705f78910eb710e751725a76a3cfe11d75287c"


class SmallTemporalDecision(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.temporal = torch.nn.LSTM(14, 8, batch_first=True)
        self.decision = torch.nn.Linear(15, 7)

    def forward(self, sensor, previous_onehot):
        if sensor.ndim != 3 or sensor.shape[1:] != (8, 14):
            raise ValueError("expected sensor [batch,8,14]")
        if previous_onehot.shape != (len(sensor), 7):
            raise ValueError("expected previous-command one-hot [batch,7]")
        sequence, _ = self.temporal(sensor)
        return self.decision(torch.cat((previous_onehot, sequence[:, -1]), dim=1))


def fit_fold(sensor, history, targets, previous, train, fold, output):
    weights, stable, changed = training_row_weights(previous, targets, train)
    torch.manual_seed(17 + fold)
    model = SmallTemporalDecision().cpu()
    if sum(p.numel() for p in model.parameters()) != 880:
        raise ValueError("unexpected recurrent parameter count")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    x = torch.from_numpy(sensor)
    h = torch.from_numpy(history)
    y = torch.from_numpy(targets.astype(np.int64))
    w = torch.from_numpy(weights)
    rng = np.random.default_rng(17 + fold)
    curve = []
    for epoch in range(EPOCHS):
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
                raise FloatingPointError("nonfinite loss")
            loss.backward()
            if any(p.grad is None or not bool(torch.isfinite(p.grad).all())
                   for p in model.parameters()):
                raise FloatingPointError("nonfinite gradient")
            optimizer.step()
            loss_sum += float(loss.detach()) * len(idx)
        curve.append(loss_sum / len(train))
        if (epoch + 1) % 10 == 0:
            print(json.dumps({"fold": fold, "epoch": epoch + 1,
                              "weighted_train_loss": curve[-1]}), flush=True)
    return model, curve, stable, changed


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
            or digest(BASE / "summary.json") != BASE_SUMMARY
            or digest(BASE / "oof-probabilities.npz") != BASE_OOF):
        raise ValueError("pinned dataset or comparator mismatch")
    data_summary = json.loads((DATA / "summary.json").read_bytes())
    base_summary = json.loads((BASE / "summary.json").read_bytes())
    if (digest(Path("scripts/train_droid_dense_change_weighted_cv.py"))
            != base_summary["script_sha256"]
            or digest(Path("scripts/train_droid_dense_history_sensor_cv.py"))
            != base_summary["baseline_script_sha256"]
            or digest(Path("scripts/audit_droid_weighted_sensor_interventions.py"))
            != BOOTSTRAP_SHA):
        raise ValueError("pinned weighting implementation mismatch")
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
            or np.any(previous >= 7) or np.any(targets >= 7) or not np.isfinite(sensor).all()):
        raise ValueError("unexpected data shape, labels or nonfinite sensor")
    fold_rows = split_groups(group, groups)
    history = np.eye(7, dtype=np.float32)[previous]
    with np.load(BASE / "oof-probabilities.npz", allow_pickle=False) as f:
        probabilities = {"linear_raw": np.array(f["weighted_history_sensor"], copy=True),
                         "linear_corrected": np.array(f["corrected_history_sensor"], copy=True)}
    probabilities["lstm_raw"] = np.zeros((len(targets), 7), dtype=np.float32)
    probabilities["lstm_corrected"] = np.zeros((len(targets), 7), dtype=np.float32)
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
        model, curve, ws, wc = fit_fold(scaled, history, targets, previous, train, fold, output)
        model.eval()
        with torch.inference_mode():
            batch_probabilities = []
            for start in range(0, len(held), BATCH):
                idx = held[start:start + BATCH]
                logits = model(torch.from_numpy(scaled[idx]), torch.from_numpy(history[idx]))
                batch_probabilities.append(torch.softmax(logits, dim=1).numpy())
        raw = np.concatenate(batch_probabilities, axis=0)
        probabilities["lstm_raw"][held] = raw
        probabilities["lstm_corrected"][held] = correct_probabilities(raw, previous[held], ws, wc)
        state = {key: value.detach().numpy() for key, value in model.state_dict().items()}
        np.savez(output / f"lstm-fold-{fold}.npz", **state, mean=mean, scale=scale,
                 stable_weight=ws, change_weight=wc)
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
    bootstrap = {mode: group_bootstrap(targets, probabilities[f"lstm_{mode}"],
                                       probabilities[f"linear_{mode}"], group)
                 for mode in ("raw", "corrected")}
    if any(digest(DATA / name) != expected
           for name, expected in data_summary["output_hashes"].items()):
        raise ValueError("dataset changed during training")
    np.savez(output / "oof-probabilities.npz", **probabilities)
    result = {"status": "TRAIN_GROUP_CV_LSTM_FOLLOWUP_NOT_BLIND", "rows": len(targets),
              "metrics": metrics, "change_detection": changes, "paired_bootstrap": bootstrap,
              "folds": fold_details, "training_loss_curves": curves,
              "parameter_count": 880, "comparator_parameters": 840, "epochs": EPOCHS,
              "batch_size": BATCH, "learning_rate": LR, "weight_decay": WEIGHT_DECAY,
              "artifact_hashes": {p.name: digest(p) for p in sorted(output.glob("*.npz"))},
              "dataset_summary_sha256": SUMMARY_SHA, "weighted_summary_sha256": BASE_SUMMARY,
              "weighted_oof_sha256": BASE_OOF, "natural_summary_sha256": BASE_SUMMARY_SHA,
              "natural_oof_sha256": BASE_OOF_SHA, "readout_sha256": READOUT_SHA,
              "bootstrap_script_sha256": BOOTSTRAP_SHA,
              "protocol_sha256": digest(PROTOCOL), "script_sha256": digest(Path(__file__)),
              "wall_seconds": time.perf_counter() - started, "torch_version": torch.__version__,
              "mkldnn_enabled": torch.backends.mkldnn.enabled, "gpu_used": False,
              "sealed_audit_loaded": False, "product_head_created": False}
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"metrics": metrics, "change_detection": changes,
                      "paired_bootstrap": bootstrap}, indent=2))


if __name__ == "__main__":
    main()
