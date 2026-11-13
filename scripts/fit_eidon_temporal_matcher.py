"""Fit pooled and LSTM readouts for a contributor-held-out video/IMU match task."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from scripts.range_generalization import sha, write
from scripts.train_sensor_subjects import metrics
from tiny_omni_decision.eidon_pairing import build_four_choice_queries


class PairMatcher(nn.Module):
    def __init__(self, mode: str, hidden: int = 32):
        super().__init__()
        if mode not in {"pooled", "lstm"}:
            raise ValueError("mode must be pooled or lstm")
        self.mode = mode
        if mode == "lstm":
            self.video_encoder = nn.LSTM(768, hidden, batch_first=True)
            self.sensor_encoder = nn.LSTM(54, hidden, batch_first=True)
        else:
            self.video_encoder = nn.Linear(768, hidden)
            self.sensor_encoder = nn.Linear(54, hidden)
        self.video_projection = nn.Linear(hidden, hidden)
        self.sensor_projection = nn.Linear(hidden, hidden)
        self.score = nn.Sequential(nn.Linear(hidden * 4, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def _encode(self, module, values):
        if self.mode == "lstm":
            _, (hidden, _) = module(values)
            return hidden[-1]
        return torch.nn.functional.gelu(module(values).mean(dim=1))

    def forward(self, video, sensor_options):
        if (video.ndim != 3 or video.shape[1:] != (8, 768)
                or sensor_options.ndim != 4
                or sensor_options.shape[1:] != (4, 8, 54)):
            raise ValueError("expected video [B,8,768] and sensor options [B,4,8,54]")
        batch, options = sensor_options.shape[:2]
        video_state = torch.tanh(self.video_projection(self._encode(self.video_encoder, video)))
        sensors = sensor_options.reshape(batch * options, 8, 54)
        sensor_state = torch.tanh(
            self.sensor_projection(self._encode(self.sensor_encoder, sensors))
        ).reshape(batch, options, -1)
        videos = video_state[:, None, :].expand_as(sensor_state)
        features = torch.cat((videos, sensor_state, torch.abs(videos - sensor_state),
                              videos * sensor_state), dim=-1)
        return self.score(features).squeeze(-1)


def _read_features(folder: Path, split: str, width: int):
    report = json.loads((folder / "result.json").read_text(encoding="utf-8"))
    feature_path = folder / "features.npz"
    if report.get("split") != split or report.get("feature_sha256") != sha(feature_path):
        raise ValueError(f"{split} feature cache hash/split mismatch: {folder}")
    with np.load(feature_path, allow_pickle=False) as data:
        ids = data["recording_ids"].astype(np.int64)
        assets = data["asset_sha256"].astype(str) if "asset_sha256" in data else None
        values = data["values"].astype(np.float32)
    if values.shape != (len(ids), 8, width):
        raise ValueError("unexpected cached feature shape")
    if not np.isfinite(values).all() or len(set(ids.tolist())) != len(ids):
        raise ValueError("feature cache is non-finite or has duplicate IDs")
    feature_map = dict(zip(ids.tolist(), values, strict=True))
    asset_map = None if assets is None else dict(zip(ids.tolist(), assets, strict=True))
    return feature_map, asset_map


def load_split(root: Path, split: str):
    base = root / "eidon-sensor-features-20261029-attempt-05" / split
    sensor, _ = _read_features(base, split, 54)
    video_folder = root / "eidon-videoprism-features-20261029-attempt-01" / split
    video, video_assets = _read_features(video_folder, split, 768)
    records = json.loads((base / "records.json").read_text(encoding="utf-8"))
    if set(sensor) != set(video) or set(sensor) != {row["recording_id"] for row in records}:
        raise ValueError("sensor/video feature identities differ")
    for recording_id in sensor:
        record = next(item for item in records if item["recording_id"] == recording_id)
        if record["asset_sha256"] != video_assets[recording_id]:
            raise ValueError("sensor/video asset hashes differ")
    rows = [{key: record[key] for key in (
        "recording_id", "contributor_id", "task_type", "split", "asset_sha256"
    )} for record in records]
    queries = build_four_choice_queries(rows, seed=17)
    return sensor, video, rows, queries


def make_episodes(sensor, video, queries):
    videos, options, targets, tasks, contributors = [], [], [], [], []
    for query in queries:
        videos.append(video[query["query_recording_id"]])
        options.append(np.stack([sensor[recording_id] for recording_id in query["options"]]))
        targets.append(query["target_index"])
        tasks.append(query["task_type"])
        contributors.append(query["contributor_id"])
    return (np.stack(videos), np.stack(options), np.asarray(targets, dtype=np.int64),
            np.asarray(tasks), np.asarray(contributors))


def _probabilities(model, video, sensors):
    with torch.inference_mode():
        return model(video, sensors).softmax(dim=-1).cpu().numpy()


def _metrics(probabilities, targets):
    result = metrics(probabilities, targets)
    result["mean_confidence"] = float(np.max(probabilities, axis=1).mean())
    return result


def _normalization(train_sensor, train_video):
    sensor_mean = train_sensor.mean(axis=(0, 1), keepdims=True)
    sensor_std = train_sensor.std(axis=(0, 1), keepdims=True)
    video_mean = train_video.mean(axis=(0, 1), keepdims=True)
    video_std = train_video.std(axis=(0, 1), keepdims=True)
    sensor_std = np.where(sensor_std > 1e-6, sensor_std, 1)
    video_std = np.where(video_std > 1e-6, video_std, 1)
    return (sensor_mean.astype(np.float32), sensor_std.astype(np.float32),
            video_mean.astype(np.float32), video_std.astype(np.float32))


def fit_mode(mode, train_data, validation_data, normalization, output_dir,
             max_epochs: int, early_stopping: bool):
    train_video, train_options, train_target, train_task, _ = train_data
    val_video, val_options, val_target, val_task, val_contributor = validation_data
    sensor_mean, sensor_std, video_mean, video_std = normalization
    train_video = torch.from_numpy((train_video - video_mean) / video_std)
    train_options = torch.from_numpy((train_options - sensor_mean) / sensor_std)
    val_video_tensor = torch.from_numpy((val_video - video_mean) / video_std)
    val_options_tensor = torch.from_numpy((val_options - sensor_mean) / sensor_std)
    train_y = torch.from_numpy(train_target)
    torch.manual_seed(17)
    model = PairMatcher(mode)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    best_nll, best_epoch, best_state, stale = float("inf"), 0, None, 0
    history = []
    started = time.perf_counter()
    for epoch in range(max_epochs + 1):
        if epoch:
            optimizer.zero_grad(set_to_none=True)
            logits = model(train_video, train_options)
            loss = nn.functional.cross_entropy(logits, train_y)
            if not torch.isfinite(loss):
                raise ValueError("non-finite Eidon training loss")
            loss.backward()
            if any(parameter.grad is not None
                   and not torch.isfinite(parameter.grad).all()
                   for parameter in model.parameters()):
                raise ValueError("non-finite Eidon gradient")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        train_prob = _probabilities(model, train_video, train_options)
        val_prob = _probabilities(model, val_video_tensor, val_options_tensor)
        train_metrics = _metrics(train_prob, train_target)
        val_metrics = _metrics(val_prob, val_target)
        history.append({"epoch": epoch, "training": train_metrics,
                        "development": val_metrics})
        if val_metrics["nll"] < best_nll - 1e-5:
            best_nll, best_epoch = val_metrics["nll"], epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if early_stopping and epoch >= 20 and stale >= 20:
            break
    if best_state is None:
        raise ValueError("no valid development checkpoint was selected")
    final_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    best_prob = _probabilities(model, val_video_tensor, val_options_tensor)
    if (_metrics(best_prob, val_target)["nll"]
            != history[best_epoch]["development"]["nll"]):
        raise ValueError("best checkpoint reload does not reproduce selector metric")
    reordered_video = val_video_tensor
    reordered_sensors = val_options_tensor.flip(1)
    reordered_prob = _probabilities(model, reordered_video, reordered_sensors)
    np.testing.assert_allclose(reordered_prob, best_prob[:, ::-1], rtol=1e-6, atol=1e-7)
    results = {"matched": best_prob}
    targets_by_control = {"matched": val_target}
    results["option_order_reversed"] = reordered_prob
    targets_by_control["option_order_reversed"] = 3 - val_target
    results["sensor_candidate_features_rotated"] = _probabilities(
        model, val_video_tensor, torch.roll(val_options_tensor, 1, dims=1)
    )
    targets_by_control["sensor_candidate_features_rotated"] = val_target
    results["video_temporal_reverse"] = _probabilities(
        model, val_video_tensor.flip(1), val_options_tensor
    )
    results["sensor_temporal_reverse"] = _probabilities(
        model, val_video_tensor, val_options_tensor.flip(2)
    )
    results["video_missing"] = _probabilities(
        model, torch.zeros_like(val_video_tensor), val_options_tensor
    )
    results["sensor_missing"] = _probabilities(
        model, val_video_tensor, torch.zeros_like(val_options_tensor)
    )
    targets_by_control.update(
        {name: val_target for name in results if name not in targets_by_control}
    )
    task_metrics = {}
    for task in sorted(set(val_task.tolist())):
        mask = val_task == task
        task_metrics[task] = {name: _metrics(prob[mask], targets_by_control[name][mask])
                              for name, prob in results.items()}
    predictions = {name: probability for name, probability in results.items()}
    prediction_path = output_dir / f"{mode}-predictions.npz"
    np.savez_compressed(prediction_path, **predictions, targets=val_target,
                        task_types=val_task, contributors=val_contributor)
    checkpoint_path = output_dir / f"{mode}-best.pt"
    torch.save(best_state, checkpoint_path)
    final_path = output_dir / f"{mode}-final.pt"
    torch.save(final_state, final_path)
    report = {
        "mode": mode,
        "parameters": parameters,
        "best_epoch": best_epoch,
        "completed_epochs": history[-1]["epoch"],
        "selector": "minimum development NLL; min_delta=1e-5",
        "early_stopping": early_stopping,
        "max_epochs": max_epochs,
        "best_development": _metrics(best_prob, val_target),
        "training_at_best": history[best_epoch]["training"],
        "controls": {name: _metrics(prob, targets_by_control[name])
                     for name, prob in results.items()},
        "option_order_equivariance_max_abs": float(
            np.max(np.abs(reordered_prob - best_prob[:, ::-1]))
        ),
        "by_task": task_metrics,
        "history": history,
        "checkpoint_sha256": sha(checkpoint_path),
        "final_checkpoint_sha256": sha(final_path),
        "predictions_sha256": sha(prediction_path),
        "wall_seconds": round(time.perf_counter() - started, 3),
    }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--disable-early-stopping", action="store_true")
    args = parser.parse_args()
    if args.max_epochs < 1:
        parser.error("--max-epochs must be positive")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output {args.output_dir}")
    train = load_split(args.artifact_root, "train")
    development = load_split(args.artifact_root, "development")
    train_contributors = {row["contributor_id"] for row in train[2]}
    development_contributors = {row["contributor_id"] for row in development[2]}
    train_assets = {row["asset_sha256"] for row in train[2]}
    development_assets = {row["asset_sha256"] for row in development[2]}
    if train_contributors & development_contributors or train_assets & development_assets:
        raise ValueError("train/development contributor or asset leakage")
    train_sensor = np.stack(list(train[0].values()))
    train_video = np.stack([train[1][record["recording_id"]] for record in train[2]])
    normalization = _normalization(train_sensor, train_video)
    train_episodes = make_episodes(train[0], train[1], train[3])
    dev_episodes = make_episodes(development[0], development[1], development[3])
    config = {
        "seed": 17,
        "modes": ["pooled", "lstm"],
        "hidden_dimension": 32,
        "max_epochs": args.max_epochs,
        "early_stopping": not args.disable_early_stopping,
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.01,
        "grad_clip": 1.0,
        "training_queries": len(train_episodes[2]),
        "training_groups": len({(row["contributor_id"], row["task_type"])
                                 for row in train[2]}),
        "development_queries": len(dev_episodes[2]),
        "development_groups": len({(row["contributor_id"], row["task_type"])
                                    for row in development[2]}),
        "train_sensor_sha256": sha(args.artifact_root /
                                   "eidon-sensor-features-20261029-attempt-05/train/features.npz"),
        "development_sensor_sha256": sha(args.artifact_root /
                                         "eidon-sensor-features-20261029-attempt-05/development/features.npz"),
        "train_video_sha256": sha(args.artifact_root /
                                  "eidon-videoprism-features-20261029-attempt-01/train/features.npz"),
        "development_video_sha256": sha(args.artifact_root /
                                        "eidon-videoprism-features-20261029-attempt-01/development/features.npz"),
        "device": "cpu",
        "sealed_audit_read": False,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "script_sha256": sha(Path(__file__)),
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write(args.output_dir / "config.json", config)
    np.savez_compressed(args.output_dir / "normalization.npz",
                        sensor_mean=normalization[0], sensor_std=normalization[1],
                        video_mean=normalization[2], video_std=normalization[3])
    results = []
    for mode in ("pooled", "lstm"):
        results.append(fit_mode(mode, train_episodes, dev_episodes,
                                normalization, args.output_dir, args.max_epochs,
                                not args.disable_early_stopping))
        write(args.output_dir / f"{mode}-results.json", results[-1])
    write(args.output_dir / "result.json", {
        "config": config,
        "uniform_baseline": {
            "count": len(dev_episodes[2]),
            "expected_accuracy": 0.25,
            "nll": float(np.log(4)),
            "brier": 0.75,
            "ece": 0.0,
            "mean_confidence": 0.25,
        },
        "candidates": results,
    })
    print(json.dumps({"candidates": [{"mode": row["mode"],
                                      "best_epoch": row["best_epoch"],
                                      "best_development": row["best_development"],
                                      "parameters": row["parameters"]}
                                     for row in results]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
