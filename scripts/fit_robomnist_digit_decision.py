"""Fit task-native 10-choice RoboMNIST readouts on original TRAIN groups only."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts import fit_robomnist_train_cv as train_cv  # noqa: E402
from tiny_omni_decision.robomnist_digit_decision import (  # noqa: E402
    condition_folds,
    multiclass_metrics,
    option_ordered_probabilities,
    parse_group,
)
from tiny_omni_decision.robomnist_train_cv import pooled_normalize  # noqa: E402

POINTS = (1, 25, 50, 100, 200, 400)


class Readout(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.linear = nn.Linear(width, 10)

    @property
    def trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        return self.linear(x)


def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest()


def run(manifest, cache, out):
    if out.exists():
        raise FileExistsError(out)
    if shutil.disk_usage(out.parent).free < 2 * 1024**3:
        raise OSError("2GiB reserve required")
    rows, _, features, hashes, cachemeta = train_cv.load_train(cache, manifest)
    groupids = set(features)
    folds = condition_folds(groupids, folds=5, seed=17)
    labels = {g: parse_group(g)[1] for g in groupids}
    condition_labels = {parse_group(g)[0]: labels[g] for g in groupids}
    if len(condition_labels) != 60 or len(groupids) != 780:
        raise ValueError("unexpected TRAIN group/condition coverage")
    out.mkdir()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    cfg = {
        "scope": "TRAIN-only task-native dev; original validation and audit features closed",
        "task": "which digit 0..9 did the robot draw",
        "labels_from": "Ac source supervision only; never encoder input",
        "seed": 17,
        "folds": 5,
        "fold_unit": "entire Ar x velocity x Ac condition; all Rp groups stay together",
        "condition_folds_sha256": digest(
            {
                c: next(iter({folds[g] for g in groupids if parse_group(g)[0] == c}))
                for c in sorted(condition_labels)
            }
        ),
        "train_groups_sha256": digest(sorted(groupids)),
        "feature_hashes_sha256": digest(hashes),
        "modalities": ["audio_only", "video_only", "joint_audio_video"],
        "options": list(range(10)),
        "pooling": "featurewise mean and standard deviation across eight windows/frames",
        "normalization": "fit on unique fitting-fold source groups only",
        "schedule": {
            "seed": 17,
            "updates": 400,
            "checkpoints": POINTS,
            "optimizer": "AdamW",
            "lr": 0.01,
            "weight_decay": 0.0001,
            "gradient_clip": 5,
            "batch": "full fit fold",
        },
        "device": "CPU",
        "torch_threads": 4,
        "no_temporal_alignment_claim": True,
    }
    train_cv.original._write_json(out / "effective-config.json", cfg)
    meta = {
        "status": "RUNNING",
        "started_at": datetime.now(UTC).isoformat(),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "config_sha256": train_cv.original.sha256_file(out / "effective-config.json"),
        "config": cfg,
        "implementation_sha256": {
            str(p.relative_to(ROOT)): train_cv.original.sha256_file(p)
            for p in [
                Path(__file__).resolve(),
                ROOT / "src/tiny_omni_decision/robomnist_digit_decision.py",
                ROOT / "src/tiny_omni_decision/robomnist_train_cv.py",
                ROOT / "scripts/extract_robomnist_av_match_feature_cache.py",
            ]
        },
        "manifest_sha256": train_cv.original.MANIFEST_SHA256,
        "encoder_contract": cachemeta["effective"],
        "train_rows": len(rows),
        "train_groups": len(groupids),
        "conditions": len(condition_labels),
        "original_validation_features_read": False,
        "sealed_audit_opened": False,
        "gpu_used": False,
        "cost_usd": 0,
    }
    train_cv.original._write_json(out / "run.json", meta)
    start = time.perf_counter()
    oof = {
        name: {s: np.full((len(groupids), 10), np.nan, np.float32) for s in POINTS}
        for name in ("audio_only", "video_only", "joint")
    }
    ordered = sorted(groupids)
    idx = {g: i for i, g in enumerate(ordered)}
    records = []
    ckhash = {}
    timings = []
    targets = np.asarray([labels[g] for g in ordered], np.int64)
    try:
        for fold in range(5):
            fitgroups = {g for g in groupids if folds[g] != fold}
            heldgroups = {g for g in groupids if folds[g] == fold}
            fitconditions = {parse_group(g)[0] for g in fitgroups}
            heldconditions = {parse_group(g)[0] for g in heldgroups}
            if fitgroups & heldgroups or fitconditions & heldconditions:
                raise ValueError("group/condition crosses folds")
            summaries, norm = pooled_normalize(features, fitgroups)
            np.savez_compressed(out / f"fold-{fold}-normalizer.npz", **norm)
            x = {
                name: torch.from_numpy(
                    np.stack(
                        [
                            np.concatenate(summaries[g])
                            if name == "joint"
                            else summaries[g][0 if name == "audio_only" else 1]
                            for g in ordered
                        ]
                    )
                )
                for name in ("audio_only", "video_only", "joint")
            }
            fitidx = np.asarray([idx[g] for g in sorted(fitgroups)])
            heldidx = np.asarray([idx[g] for g in sorted(heldgroups)])
            for name in oof:
                torch.manual_seed(17)
                model = Readout(x[name].shape[1])
                opt = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0001)
                y = torch.from_numpy(targets[fitidx])
                xf = x[name][fitidx]
                xv = x[name][heldidx]
                yv = targets[heldidx]
                for step in range(1, 401):
                    tick = time.perf_counter()
                    model.train()
                    opt.zero_grad(set_to_none=True)
                    loss = F.cross_entropy(model(xf), y)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("nonfinite train CE")
                    loss.backward()
                    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                    if not torch.isfinite(gn):
                        raise FloatingPointError("nonfinite gradient")
                    opt.step()
                    elapsed = time.perf_counter() - tick
                    if fold == 0 and name == "joint" and step in (2, 3, 4):
                        timings.append(elapsed)
                    if step not in POINTS:
                        continue
                    model.eval()
                    with torch.inference_mode():
                        trainp = option_ordered_probabilities(model(xf), list(range(10))).numpy()
                        heldp = option_ordered_probabilities(model(xv), list(range(10))).numpy()
                    oof[name][step][heldidx] = heldp
                    path = out / f"{name}-fold-{fold}-step-{step}.pt"
                    torch.save(model.state_dict(), path)
                    reloaded = Readout(x[name].shape[1])
                    reloaded.load_state_dict(
                        torch.load(path, map_location="cpu", weights_only=True)
                    )
                    with torch.inference_mode():
                        restored = option_ordered_probabilities(
                            reloaded(xv), list(range(10))
                        ).numpy()
                    if not np.array_equal(heldp, restored):
                        raise RuntimeError("save/reload mismatch")
                    ckhash[path.name] = train_cv.original.sha256_file(path)
                    rec = {
                        "modality": name,
                        "fold": fold,
                        "step": step,
                        "fit_groups": len(fitgroups),
                        "held_groups": len(heldgroups),
                        "fit_conditions": len(fitconditions),
                        "held_conditions": len(heldconditions),
                        "fit_ids_sha256": digest(sorted(fitgroups)),
                        "held_ids_sha256": digest(sorted(heldgroups)),
                        "normalizer_sha256": train_cv.original.sha256_file(
                            out / f"fold-{fold}-normalizer.npz"
                        ),
                        "train": multiclass_metrics(trainp, targets[fitidx]),
                        "held_fold": multiclass_metrics(heldp, yv),
                        "trainable_parameters": model.trainable_parameters,
                        "gradient_norm": float(gn),
                        "elapsed_seconds": time.perf_counter() - start,
                    }
                    records.append(rec)
                    with (out / "history.jsonl").open("a", encoding="utf-8") as f:
                        f.write(json.dumps(rec) + "\n")
            print(json.dumps({"fold": fold, "elapsed": time.perf_counter() - start}), flush=True)
        curves = {}
        for name in oof:
            curves[name] = []
            for step in POINTS:
                if not np.isfinite(oof[name][step]).all():
                    raise ValueError("OOF predictions incomplete")
                curves[name].append({"step": step, **multiclass_metrics(oof[name][step], targets)})
        with (out / "oof-predictions.jsonl").open("x", encoding="utf-8") as f:
            for name in oof:
                for step in POINTS:
                    for g, p in zip(ordered, oof[name][step], strict=True):
                        f.write(
                            json.dumps(
                                {
                                    "group_id": g,
                                    "condition": parse_group(g)[0],
                                    "target": labels[g],
                                    "fold": folds[g],
                                    "modality": name,
                                    "step": step,
                                    "prediction": int(p.argmax()),
                                    "probabilities": p.tolist(),
                                }
                            )
                            + "\n"
                        )
        if torch.cuda.is_initialized():
            raise RuntimeError("CUDA unexpectedly initialized")
        result = {
            **meta,
            "status": "COMPLETE",
            "finished_at": datetime.now(UTC).isoformat(),
            "elapsed_seconds": time.perf_counter() - start,
            "step_seconds_samples": timings,
            "curves": curves,
            "fold_records": records,
            "checkpoint_sha256": ckhash,
            "checkpoint_count": len(ckhash),
            "reload_exact": True,
            "predictions_sha256": train_cv.original.sha256_file(out / "oof-predictions.jsonl"),
            "trainable_params_by_modality": {
                "audio_only": 7690,
                "video_only": 15370,
                "joint": 23050,
            },
        }
        train_cv.original._write_json(out / "run.json", result)
        print(json.dumps({"status": "COMPLETE", "curves": curves}), flush=True)
    except BaseException as e:
        train_cv.original._write_json(
            out / "run.json",
            {
                **meta,
                "status": "INCOMPLETE",
                "error": repr(e),
                "elapsed_seconds": time.perf_counter() - start,
            },
        )
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.manifest.resolve(strict=True), a.cache.resolve(strict=True), a.output.resolve())
