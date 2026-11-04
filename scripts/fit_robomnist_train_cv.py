"""Fixed learning curves using only original RoboMNIST TRAIN groups."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
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
from scripts import fit_robomnist_av_match_readout as original  # noqa: E402
from tiny_omni_decision.robomnist_av_match_evaluation import classification_metrics  # noqa: E402
from tiny_omni_decision.robomnist_av_match_readout import BilinearAVMatchScorer  # noqa: E402
from tiny_omni_decision.robomnist_train_cv import grouped_folds, pooled_normalize  # noqa: E402

POINTS = (1, 25, 50, 100, 200, 400)
ORIGINAL_SCALE = math.sqrt(32)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def load_train(cache, manifest_path):
    if original.sha256_file(manifest_path) != original.MANIFEST_SHA256:
        raise ValueError("original manifest hash changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest["examples"]["train"]
    groups = {r[k] for r in rows for k in ("audio_group", "video_group")}
    forbidden = {r[k] for r in manifest["examples"]["validation"]
                 for k in ("audio_group", "video_group")}
    if groups & forbidden or len(groups) != 780 or len(rows) != 1560:
        raise ValueError("original TRAIN identity/count contract failed")
    folds = grouped_folds(rows)
    meta = json.loads((cache / "run.json").read_text(encoding="utf-8"))
    if (meta.get("status") != "COMPLETE" or meta.get("sealed_audit_opened") is not False
            or meta.get("effective", {}).get("manifest_sha256") != original.MANIFEST_SHA256):
        raise ValueError("frozen cache metadata mismatch")
    # Only this whitelist is passed to the loader. Validation identities above
    # are metadata for the disjointness check; no validation arrays are opened.
    features, hashes = original._load_group_features(cache, groups)
    if set(features) != groups or set(hashes) != groups:
        raise ValueError("feature loader returned IDs outside TRAIN whitelist")
    return rows, folds, features, hashes, meta


def predict(model, inputs):
    with torch.inference_mode():
        return model(*inputs).softmax(-1).numpy()


class ScoreScaleScorer(BilinearAVMatchScorer):
    """Development arm changing only the existing cosine logit multiplier."""

    def __init__(self, score_scale):
        super().__init__(rank=32)
        if not math.isfinite(score_scale) or score_scale <= 0:
            raise ValueError("score scale must be finite and positive")
        self.score_scale = score_scale

    def forward(self, audio, video):
        return super().forward(audio, video) * (self.score_scale / math.sqrt(32))


def run(manifest_path, cache, output, *, score_scale=ORIGINAL_SCALE):
    if output.exists():
        raise FileExistsError(output)
    if shutil.disk_usage(output.parent).free < 2 * 1024**3:
        raise OSError("2 GiB output reserve is required")
    rows, fold_map, features, hashes, cache_meta = load_train(cache, manifest_path)
    output.mkdir()
    started = time.perf_counter()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    config = {
        "scope": "TRAIN-only internal development; not independent product evaluation",
        "score_scale": score_scale,
        "seed": 17, "folds": 5, "steps": 400, "points": POINTS, "rank": 32,
        "learning_rate": 0.01, "weight_decay": 0.0001, "optimizer": "AdamW",
        "gradient_clip": 5.0, "batch": "all fitting rows each update",
        "selection": "none; all preregistered steps reported", "device": "cpu",
        "torch_threads": 4, "manifest_sha256": original.MANIFEST_SHA256,
        "train_rows_sha256": digest(rows), "train_features_sha256": digest(hashes),
        "fold_map_sha256": digest(fold_map), "fold_map": fold_map,
    }
    original._write_json(output / "effective-config.json", config)
    meta = {
        "status": "RUNNING", "started_at": datetime.now(UTC).isoformat(),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "implementation_hashes": {
            str(p.relative_to(ROOT)): original.sha256_file(p) for p in (
                Path(__file__).resolve(), Path(original.__file__).resolve(),
                ROOT / "src/tiny_omni_decision/robomnist_train_cv.py",
                ROOT / "src/tiny_omni_decision/robomnist_av_match_readout.py",
                ROOT / "src/tiny_omni_decision/robomnist_av_match_evaluation.py")},
        "config_sha256": original.sha256_file(output / "effective-config.json"),
        "python": platform.python_version(), "torch": torch.__version__,
        "numpy": np.__version__, "platform": platform.platform(),
        "cpu": platform.processor(), "cache": str(cache),
        "encoder_contract": cache_meta["effective"],
        "original_validation_features_read": False, "sealed_audit_opened": False,
        "original_checkpoint_loaded": False, "gpu_used": False,
        "train_groups": len(features), "pair_units": len(fold_map), "rows": len(rows),
    }
    original._write_json(output / "run.json", meta)
    try:
        oof = {point: np.full((len(rows), 2), np.nan, dtype=np.float32) for point in POINTS}
        fold_records = []
        seconds_per_step = []
        checkpoint_hashes = {}
        for fold in range(5):
            fit_idx = [i for i, row in enumerate(rows) if fold_map[row["pair_unit_id"]] != fold]
            held_idx = [i for i, row in enumerate(rows) if fold_map[row["pair_unit_id"]] == fold]
            fit_rows, held_rows = [rows[i] for i in fit_idx], [rows[i] for i in held_idx]
            fit_groups = {r[k] for r in fit_rows for k in ("audio_group", "video_group")}
            held_groups = {r[k] for r in held_rows for k in ("audio_group", "video_group")}
            if fit_groups & held_groups:
                raise ValueError("cross-fold source overlap")
            summaries, normalizer = pooled_normalize(features, fit_groups)
            normalizer_path = output / f"fold-{fold}-normalizer.npz"
            np.savez_compressed(normalizer_path, **normalizer)
            a, v, target, _ = original._arrays(fit_rows, summaries)
            ha, hv, ht, _ = original._arrays(held_rows, summaries)
            torch.manual_seed(17)
            model = ScoreScaleScorer(score_scale)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0001)
            for step in range(1, 401):
                tick = time.perf_counter()
                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(a, v), target)
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite training CE")
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError("nonfinite gradient norm")
                optimizer.step()
                elapsed_step = time.perf_counter() - tick
                if fold == 0 and step in (2, 3, 4):
                    seconds_per_step.append(elapsed_step)
                if step not in POINTS:
                    continue
                model.eval()
                train_p, held_p = predict(model, (a, v)), predict(model, (ha, hv))
                oof[step][held_idx] = held_p
                path = output / f"fold-{fold}-step-{step}.pt"
                torch.save(model.state_dict(), path)
                restored = ScoreScaleScorer(score_scale)
                restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
                restored.eval()
                if not np.array_equal(held_p, predict(restored, (ha, hv))):
                    raise RuntimeError("checkpoint reload differs")
                checkpoint_hashes[path.name] = original.sha256_file(path)
                record = {
                    "fold": fold, "step": step, "fit_groups": len(fit_groups),
                    "held_groups": len(held_groups), "fit_pair_units": len(fit_rows) // 4,
                    "held_pair_units": len(held_rows) // 4,
                    "fit_ids_sha256": digest(sorted(fit_groups)),
                    "held_ids_sha256": digest(sorted(held_groups)),
                    "normalizer_sha256": original.sha256_file(normalizer_path),
                    "train": classification_metrics(train_p, target.numpy()),
                    "held_fold": classification_metrics(held_p, ht.numpy()),
                    "gradient_norm": float(grad_norm),
                    "elapsed_seconds": time.perf_counter() - started,
                }
                fold_records.append(record)
                with (output / "history.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record) + "\n")
            print(json.dumps({"fold_complete": fold, "seconds": time.perf_counter()-started}),
                  flush=True)
        targets = np.array([r["target"] for r in rows])
        curves = []
        with (output / "oof-predictions.jsonl").open("x", encoding="utf-8") as stream:
            for step in POINTS:
                if not np.isfinite(oof[step]).all():
                    raise ValueError("incomplete out-of-fold predictions")
                curves.append({"step": step, **classification_metrics(oof[step], targets)})
                for row, probs in zip(rows, oof[step], strict=True):
                    stream.write(json.dumps({**row, "fold": fold_map[row["pair_unit_id"]],
                                             "step": step, "probabilities": probs.tolist(),
                                             "prediction": int(probs.argmax())}) + "\n")
        if torch.cuda.is_initialized():
            raise RuntimeError("CUDA unexpectedly initialized")
        result = {
            **meta, "status": "COMPLETE", "finished_at": datetime.now(UTC).isoformat(),
            "elapsed_seconds": time.perf_counter() - started,
            "step_seconds_three_samples": seconds_per_step,
            "mean_step_seconds": float(np.mean(seconds_per_step)),
            "trainable_parameters": model.trainable_parameters,
            "per_checkpoint_bytes": path.stat().st_size,
            "curves": curves, "fold_records": fold_records,
            "checkpoint_hashes": checkpoint_hashes,
            "reload_exact": True,
            "predictions_sha256": original.sha256_file(output / "oof-predictions.jsonl"),
            "local_cost_usd": 0, "cloud_cost_usd": 0,
        }
        original._write_json(output / "run.json", result)
        print(json.dumps({"status": "COMPLETE", "curves": curves}), flush=True)
        return result
    except BaseException as error:
        original._write_json(output / "run.json", {
            **meta, "status": "INCOMPLETE", "error": repr(error),
            "elapsed_seconds": time.perf_counter()-started})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--score-scale", type=float, choices=(1.0, math.sqrt(32)),
                        default=math.sqrt(32))
    args = parser.parse_args()
    run(args.manifest.resolve(strict=True), args.cache.resolve(strict=True), args.output.resolve(),
        score_scale=args.score_scale)
