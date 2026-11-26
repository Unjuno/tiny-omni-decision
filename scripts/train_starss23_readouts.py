"""Execute the preregistered frozen STARSS23 readout comparison on CPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.nn import functional as F

from scripts.extract_frozen_videoprism_features import CHECKPOINT_SHA256, SOURCE_COMMIT, sha256_file
from scripts.extract_starss23_corpus_features import read_manifest
from scripts.extract_starss23_features import AUDIO_SHA, AUDIO_SOURCE
from scripts.fetch_starss23_frozen_media import write_json
from scripts.probe_starss23_windows import EVENTS
from scripts.starss23_readout_common import binary_metrics, different_recording_permutation
from tiny_omni_decision.decision import FrozenFeatureCandidateScorer

POLICY_SHA = "49c00e513cacb8872ce957b1f9c97efb2b22bac13ecfbea9729a2fd134dcc6b8"


def candidate_inputs(features, query_indices):
    observation = features[query_indices // 12]
    event = F.one_hot(query_indices % 12, 12).float()
    shared = torch.cat((observation, event), dim=-1)
    return torch.cat(
        (shared[:, None, :].expand(-1, 2, -1), torch.eye(2).expand(len(query_indices), -1, -1)),
        dim=-1,
    )


def standardize(features, train_mask):
    if not train_mask.any():
        raise ValueError("no training observations")
    mean = features[train_mask].mean(axis=0)
    std = features[train_mask].std(axis=0)
    std = np.where(std > 1e-6, std, 1.0)
    return ((features - mean) / std).astype(np.float32), mean, std


def predict(model, features, reverse_options=False):
    model.eval()
    outputs = []
    with torch.inference_mode():
        for indices in torch.arange(len(features) * 12).split(128):
            inputs = candidate_inputs(features, indices)
            logits = model(inputs.flip(1) if reverse_options else inputs)
            outputs.append(logits.flip(1) if reverse_options else logits)
    return torch.cat(outputs).reshape(-1, 12, 2)


def temperature_from_calibration(logits, targets):
    grid = np.geomspace(0.25, 4.0, 17)
    losses = [
        float(F.cross_entropy((logits / float(t)).reshape(-1, 2), targets.reshape(-1)))
        for t in grid
    ]
    return float(grid[int(np.argmin(losses))]), losses


def metrics(targets, probabilities):
    by_event = {
        event: binary_metrics(targets[:, i], probabilities[:, i]) for i, event in enumerate(EVENTS)
    }
    macro = {
        key: float(np.mean([r[key] for r in by_event.values()]))
        for key in ("accuracy", "nll", "brier", "ece")
    }
    return {"macro": macro, "per_event": by_event}


def paired_room_bootstrap(targets, left, right, rooms, replicates=10000):
    groups = sorted(set(rooms))
    if not groups:
        raise ValueError("no bootstrap rooms")
    room_array = np.array(rooms)
    draws = np.random.default_rng(17).multinomial(
        len(groups), [1 / len(groups)] * len(groups), size=replicates
    )
    cache = {}
    differences = []
    for draw in draws:
        key = tuple(draw.tolist())
        if key not in cache:
            indices = np.concatenate(
                [
                    np.tile(np.flatnonzero(room_array == group), count)
                    for group, count in zip(groups, draw, strict=True)
                    if count
                ]
            )
            a = metrics(targets[indices], left[indices])
            b = metrics(targets[indices], right[indices])
            cache[key] = [
                a["macro"][metric] - b["macro"][metric]
                for metric in ("accuracy", "nll", "brier", "ece")
            ]
        differences.append(cache[key])
    intervals = np.percentile(differences, [2.5, 97.5], axis=0)
    return {
        "unit": "room",
        "groups": len(groups),
        "replicates": replicates,
        "seed": 17,
        "method": "paired percentile 95; left minus right",
        "low_power": len(groups) < 200,
        "macro_intervals": {
            metric: intervals[:, i].tolist()
            for i, metric in enumerate(("accuracy", "nll", "brier", "ece"))
        },
    }


def load_feature_matrix(directory, modality, rows, manifest_sha):
    report = json.loads((directory / "result.json").read_text())
    path = directory / "features.npy"
    if report.get("status") != "PASS" or report.get("modality") != modality:
        raise ValueError("incomplete or wrong-modality features")
    expected = (
        (AUDIO_SHA, AUDIO_SOURCE) if modality == "audio" else (CHECKPOINT_SHA256, SOURCE_COMMIT)
    )
    if (report.get("checkpoint_sha256"), report.get("source_commit")) != expected:
        raise ValueError("encoder checkpoint/source mismatch")
    if report["manifest_sha256"] != manifest_sha or sha256_file(path) != report["feature_sha256"]:
        raise ValueError("feature provenance mismatch")
    if [(r["id"], r["split"]) for r in report["features"]] != [(r["id"], r["split"]) for r in rows]:
        raise ValueError("feature order/splits differ from manifest")
    if any(
        f["recording"] != r["recording"] or f["media_sha256"] != r["media"][modality]["sha256"]
        for f, r in zip(report["features"], rows, strict=True)
    ):
        raise ValueError("feature media identity mismatch")
    data = np.load(path, allow_pickle=False)
    if (
        data.shape != (len(rows), 384 if modality == "audio" else 768)
        or not np.isfinite(data).all()
    ):
        raise ValueError("invalid feature matrix")
    return data


def train_condition(features, targets, masks, config, output):
    torch.manual_seed(17)
    model = FrozenFeatureCandidateScorer(1166, 64)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    x = {s: torch.from_numpy(features[m]) for s, m in masks.items()}
    y = {s: torch.from_numpy(targets[m]).long() for s, m in masks.items()}
    best_loss = float("inf")
    history = []
    generator = torch.Generator().manual_seed(17)
    best_state = None
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        total = 0.0
        count = 0
        for indices in torch.randperm(len(x["train"]) * 12, generator=generator).split(
            config["batch_questions"]
        ):
            logits = model(candidate_inputs(x["train"], indices))
            loss = F.cross_entropy(logits, y["train"].reshape(-1)[indices])
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss")
            optimizer.zero_grad()
            loss.backward()
            if any(
                p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()
            ):
                raise ValueError("nonfinite gradient")
            optimizer.step()
            total += float(loss.detach()) * len(indices)
            count += len(indices)
        val_logits = predict(model, x["validation"])
        val_loss = float(F.cross_entropy(val_logits.reshape(-1, 2), y["validation"].reshape(-1)))
        history.append(
            {"epoch": epoch, "rolling_train_ce": total / count, "validation_nll": val_loss}
        )
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, output / "best.pt")
        write_json(output / "history.json", history)
    torch.save(model.state_dict(), output / "final.pt")
    model.load_state_dict(torch.load(output / "best.pt", weights_only=True))
    if not all(torch.equal(v, best_state[k]) for k, v in model.state_dict().items()):
        raise ValueError("best reload mismatch")
    cal_logits = predict(model, x["calibration"])
    temperature, cal_losses = temperature_from_calibration(cal_logits, y["calibration"])
    val_logits = predict(model, x["validation"])
    if not torch.allclose(val_logits, predict(model, x["validation"], True), atol=1e-6, rtol=1e-5):
        raise ValueError("option permutation changed logits")
    raw = val_logits.softmax(-1)[..., 1].numpy()
    calibrated = (val_logits / temperature).softmax(-1)[..., 1].numpy()
    train_prob = (predict(model, x["train"]) / temperature).softmax(-1)[..., 1].numpy()
    result = {
        "best_epoch": best_epoch,
        "temperature": temperature,
        "calibration_grid_nll": cal_losses,
        "parameters": sum(p.numel() for p in model.parameters()),
        "best_sha256": sha256_file(output / "best.pt"),
        "final_sha256": sha256_file(output / "final.pt"),
        "best_bytes": (output / "best.pt").stat().st_size,
        "validation_raw": metrics(targets[masks["validation"]], raw),
        "validation_calibrated": metrics(targets[masks["validation"]], calibrated),
        "train_fixed_calibrated": metrics(targets[masks["train"]], train_prob),
    }
    np.savez(
        output / "validation_predictions.npz",
        targets=targets[masks["validation"]],
        positive_probabilities=calibrated,
        raw_positive_probabilities=raw,
    )
    write_json(output / "result.json", result)
    return model, temperature, calibrated, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("manifest-dir", "audio-features", "video-features", "config", "output-dir"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8-sig"))
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(canonical).hexdigest() != POLICY_SHA:
        raise ValueError("configuration differs from preregistration")
    rows = read_manifest(args.manifest_dir)
    window_report = json.loads((args.manifest_dir / "result.json").read_text())
    if window_report["summary_sha256"] != config["split_summary_sha256"]:
        raise ValueError("windows do not use preregistered frozen split summary")
    manifest_sha = sha256_file(args.manifest_dir / "result.json")
    audio = load_feature_matrix(args.audio_features, "audio", rows, manifest_sha)
    video = load_feature_matrix(args.video_features, "video", rows, manifest_sha)
    masks = {
        s: np.array([r["split"] == s for r in rows]) for s in ("train", "calibration", "validation")
    }
    if any(not m.any() for m in masks.values()):
        raise ValueError("all three independent splits are required")
    for row in rows:
        if [q["event"] for q in row["queries"]] != list(EVENTS) or any(
            q["options"] != ["no", "yes"] for q in row["queries"]
        ):
            raise ValueError("query vocabulary/order mismatch")
    targets = np.array([[q["target"] for q in r["queries"]] for r in rows], dtype=np.int64)
    if not np.isin(targets, [0, 1]).all():
        raise ValueError("invalid binary target")
    values, mean, std = standardize(np.concatenate((audio, video), axis=1), masks["train"])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    np.savez(args.output_dir / "normalization.npz", mean=mean, std=std)
    write_json(args.output_dir / "effective_config.json", config)
    write_json(
        args.output_dir / "run_inputs.json",
        {
            "argv": sys.argv,
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "manifest_sha256": manifest_sha,
            "audio_report_sha256": sha256_file(args.audio_features / "result.json"),
            "video_report_sha256": sha256_file(args.video_features / "result.json"),
            "split_windows": {s: int(m.sum()) for s, m in masks.items()},
            "device": "cpu",
            "paid_compute": False,
        },
    )
    write_json(
        args.output_dir / "validation_ids.json",
        [r["id"] for r in rows if r["split"] == "validation"],
    )
    torch.set_num_threads(4)
    started = time.perf_counter()
    results = {}
    for condition in config["conditions"]:
        features = values.copy()
        if condition == "audio_only":
            features[:, 384:] = 0
        elif condition == "video_only":
            features[:, :384] = 0
        output = args.output_dir / condition
        output.mkdir()
        model, temp, probs, result = train_condition(
            features, targets, masks, config["readout"], output
        )
        results[condition] = result
        if condition == "audio_video":
            val = features[masks["validation"]]
            val_rows = [r for r in rows if r["split"] == "validation"]
            order = different_recording_permutation([r["recording"] for r in val_rows])
            controls = {}
            for control in ("zero_audio", "zero_video", "different_recording_video"):
                changed = val.copy()
                if control == "zero_audio":
                    changed[:, :384] = 0
                elif control == "zero_video":
                    changed[:, 384:] = 0
                else:
                    changed[:, 384:] = val[order, 384:]
                p = (predict(model, torch.from_numpy(changed)) / temp).softmax(-1)[..., 1].numpy()
                controls[control] = metrics(targets[masks["validation"]], p)
                np.save(output / f"{control}.npy", p, allow_pickle=False)
            results["joint_controls"] = controls
            write_json(output / "video_permutation.json", order)
    prior = (targets[masks["train"]].sum(axis=0) + 1) / (int(masks["train"].sum()) + 2)
    results["train_prior"] = metrics(
        targets[masks["validation"]], np.broadcast_to(prior, targets[masks["validation"]].shape)
    )
    results.update(
        status="PASS",
        wall_seconds=time.perf_counter() - started,
        manifest_sha256=manifest_sha,
        policy_sha256=POLICY_SHA,
        runner_sha256=sha256_file(Path(__file__)),
        blind_audit=False,
        unsupported_positive_training_events=[
            e for i, e in enumerate(EVENTS) if not targets[masks["train"], i].any()
        ],
    )
    joint = np.load(args.output_dir / "audio_video/validation_predictions.npz", allow_pickle=False)[
        "positive_probabilities"
    ]
    validation_targets = targets[masks["validation"]]
    validation_rooms = [r["room"] for r in rows if r["split"] == "validation"]
    comparisons = {}
    for condition in ("audio_only", "video_only"):
        other = np.load(
            args.output_dir / condition / "validation_predictions.npz", allow_pickle=False
        )["positive_probabilities"]
        comparisons[condition] = paired_room_bootstrap(
            validation_targets, joint, other, validation_rooms
        )
    for control in ("zero_audio", "zero_video", "different_recording_video"):
        other = np.load(args.output_dir / "audio_video" / f"{control}.npy", allow_pickle=False)
        comparisons[control] = paired_room_bootstrap(
            validation_targets, joint, other, validation_rooms
        )
    results["paired_room_bootstrap"] = comparisons
    results["wall_seconds"] = time.perf_counter() - started
    write_json(args.output_dir / "result.json", results)


if __name__ == "__main__":
    main()
