"""Compare ordered and fixed-shuffled temporal AV readouts on TRAIN folds only."""

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
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts import fit_robomnist_train_cv as base  # noqa: E402
from tiny_omni_decision.robomnist_av_match_evaluation import classification_metrics  # noqa: E402
from tiny_omni_decision.robomnist_temporal_cv import (  # noqa: E402
    TemporalAVMatchScorer,
    normalize_sequences,
    permute_by_group,
)  # noqa: E402

POINTS = (1, 25, 50, 100, 200, 400)


def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest()


def run(manifest, cache, out):
    if out.exists():
        raise FileExistsError(out)
    if shutil.disk_usage(out.parent).free < 2 * 1024**3:
        raise OSError("2 GiB reserve required")
    rows, folds, features, hashes, cachemeta = base.load_train(cache, manifest)
    # Full sequences only; never use the pooled model or its training artifact.
    foldmap = folds
    out.mkdir()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    cfg = {
        "scope": "TRAIN-only development, no original validation arrays or sealed audit",
        "seed": 17,
        "folds": 5,
        "steps": 400,
        "points": POINTS,
        "model": "two separate one-layer LSTMs",
        "hidden_size": 16,
        "trainable_parameters": 76032,
        "baseline_parameters": 73728,
        "relative_parameter_increase": (76032 / 73728) - 1,
        "arms": ["ordered", "fixed_groupwise_independent_stream_shuffle"],
        "training": "same seed, folds, AdamW lr=.01 wd=.0001, grad clip5, full batch",
        "normalizer": "fit-only unique groups, timestep feature mean/std across eight tokens",
        "no_cross_modal_clock_alignment": True,
        "manifest_sha256": base.original.MANIFEST_SHA256,
        "train_rows_sha256": digest(rows),
        "features_sha256": digest(hashes),
        "fold_map_sha256": digest(foldmap),
    }
    base.original._write_json(out / "effective-config.json", cfg)
    meta = {
        "status": "RUNNING",
        "started_at": datetime.now(UTC).isoformat(),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "config_sha256": base.original.sha256_file(out / "effective-config.json"),
        "config": cfg,
        "implementation_hashes": {
            str(p.relative_to(ROOT)): base.original.sha256_file(p)
            for p in [
                Path(__file__).resolve(),
                Path(base.__file__).resolve(),
                ROOT / "src/tiny_omni_decision/robomnist_temporal_cv.py",
                ROOT / "src/tiny_omni_decision/robomnist_train_cv.py",
                ROOT / "src/tiny_omni_decision/robomnist_av_match_evaluation.py",
            ]
        },
        "encoder_contract": cachemeta["effective"],
        "cache": str(cache),
        "train_rows": len(rows),
        "groups": len(features),
        "original_validation_features_read": False,
        "sealed_audit_opened": False,
        "gpu_used": False,
    }
    base.original._write_json(out / "run.json", meta)
    start = time.perf_counter()
    records = []
    hashesout = {}
    secs = []
    oof = {
        arm: {s: np.full((len(rows), 2), np.nan, np.float32) for s in POINTS}
        for arm in ("ordered", "shuffled")
    }
    try:
        for fold in range(5):
            trainidx = [i for i, r in enumerate(rows) if foldmap[r["pair_unit_id"]] != fold]
            heldidx = [i for i, r in enumerate(rows) if foldmap[r["pair_unit_id"]] == fold]
            tr = [rows[i] for i in trainidx]
            va = [rows[i] for i in heldidx]
            fitgroups = {r[k] for r in tr for k in ("audio_group", "video_group")}
            heldgroups = {r[k] for r in va for k in ("audio_group", "video_group")}
            if fitgroups & heldgroups:
                raise ValueError("cross fold leakage")
            normed, norm = normalize_sequences(features, fitgroups)
            np.savez_compressed(out / f"fold-{fold}-normalizer.npz", **norm)
            variants = {"ordered": normed, "shuffled": permute_by_group(normed, seed=17)}
            for arm, view in variants.items():

                def arrays(part, source_view=view):
                    return (
                        torch.from_numpy(
                            np.stack([source_view[r["audio_group"]][0] for r in part])
                        ),
                        torch.from_numpy(
                            np.stack([source_view[r["video_group"]][1] for r in part])
                        ),
                        torch.tensor([r["target"] for r in part], dtype=torch.long),
                    )

                a, v, y = arrays(tr)
                ha, hv, hy = arrays(va)
                torch.manual_seed(17)
                model = TemporalAVMatchScorer()
                opt = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0001)
                for step in range(1, 401):
                    tick = time.perf_counter()
                    model.train()
                    opt.zero_grad(set_to_none=True)
                    loss = F.cross_entropy(model(a, v), y)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("non-finite CE")
                    loss.backward()
                    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    if not torch.isfinite(gn):
                        raise FloatingPointError("non-finite gradient")
                    opt.step()
                    elapsed = time.perf_counter() - tick
                    if fold == 0 and arm == "ordered" and step in (2, 3, 4):
                        secs.append(elapsed)
                    if step not in POINTS:
                        continue
                    model.eval()
                    tp = base.predict(model, (a, v))
                    vp = base.predict(model, (ha, hv))
                    oof[arm][step][heldidx] = vp
                    cp = out / f"{arm}-fold-{fold}-step-{step}.pt"
                    torch.save(model.state_dict(), cp)
                    reload = TemporalAVMatchScorer()
                    reload.load_state_dict(torch.load(cp, map_location="cpu", weights_only=True))
                    reload.eval()
                    if not np.array_equal(vp, base.predict(reload, (ha, hv))):
                        raise RuntimeError("reload mismatch")
                    hashesout[cp.name] = base.original.sha256_file(cp)
                    rec = {
                        "arm": arm,
                        "fold": fold,
                        "step": step,
                        "fit_groups": len(fitgroups),
                        "held_groups": len(heldgroups),
                        "fit_pairs": len(tr) // 4,
                        "held_pairs": len(va) // 4,
                        "fit_groups_sha256": digest(sorted(fitgroups)),
                        "held_groups_sha256": digest(sorted(heldgroups)),
                        "normalizer_sha256": base.original.sha256_file(
                            out / f"fold-{fold}-normalizer.npz"
                        ),
                        "train": classification_metrics(tp, y.numpy()),
                        "held_fold": classification_metrics(vp, hy.numpy()),
                        "gradient_norm": float(gn),
                        "elapsed_seconds": time.perf_counter() - start,
                    }
                    records.append(rec)
                    with (out / "history.jsonl").open("a", encoding="utf8") as f:
                        f.write(json.dumps(rec) + "\n")
            print(json.dumps({"fold": fold, "elapsed": time.perf_counter() - start}), flush=True)
        curves = {}
        with (out / "oof-predictions.jsonl").open("x", encoding="utf8") as f:
            targets = np.array([r["target"] for r in rows])
            for arm in oof:
                curves[arm] = []
                for step in POINTS:
                    probs = oof[arm][step]
                    if not np.isfinite(probs).all():
                        raise ValueError("incomplete OOF predictions")
                    curves[arm].append({"step": step, **classification_metrics(probs, targets)})
                    for row, p in zip(rows, probs, strict=True):
                        f.write(
                            json.dumps(
                                {
                                    **row,
                                    "arm": arm,
                                    "step": step,
                                    "fold": foldmap[row["pair_unit_id"]],
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
            "step_seconds_samples": secs,
            "mean_step_seconds": float(np.mean(secs)),
            "curves": curves,
            "fold_records": records,
            "checkpoint_sha256": hashesout,
            "checkpoint_count": len(hashesout),
            "reload_exact": True,
            "predictions_sha256": base.original.sha256_file(out / "oof-predictions.jsonl"),
            "cost_usd": 0,
        }
        base.original._write_json(out / "run.json", result)
        print(json.dumps({"status": "COMPLETE", "curves": curves}), flush=True)
    except BaseException as e:
        base.original._write_json(
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
