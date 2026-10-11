"""Extract frozen per-window STARSS23 features in separate pinned model environments."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from scripts.extract_frozen_videoprism_features import (
    CHECKPOINT_SHA256,
    MODEL_REVISION,
    SOURCE_COMMIT,
    sha256_file,
)

AUDIO_SHA = "49b0e63577021994257b3de2ac5f1686d71554b93c8149f9d2568ad5760b904e"
AUDIO_SOURCE = "a425fdce92572e602a1d5634799bd9f1f2efa806"


def validate_windows(report):
    if report.get("status") != "PASS" or not report.get("windows"):
        raise ValueError("window probe is not complete")
    rows = report["windows"]
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("duplicate window identity")
    if any(r["split"] != "dev-train" for r in rows):
        raise ValueError("only dev-train smoke windows permitted")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modality", choices=["audio", "video"], required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = validate_windows(json.loads(args.windows.read_text(encoding="utf-8")))
    expected_sha = AUDIO_SHA if args.modality == "audio" else CHECKPOINT_SHA256
    revision = AUDIO_SOURCE if args.modality == "audio" else SOURCE_COMMIT
    if sha256_file(args.checkpoint) != expected_sha:
        raise ValueError("checkpoint hash mismatch")
    actual_revision = subprocess.check_output(
        ["git", "-C", str(args.source_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_revision != revision:
        raise ValueError("model source revision mismatch")
    for row in rows:
        if sha256_file(Path(row["path"])) != row["sha256"]:
            raise ValueError("window hash mismatch")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.source_root.resolve()))
    import psutil

    process = psutil.Process()
    started = time.perf_counter()
    report = {
        "status": "RUNNING",
        "modality": args.modality,
        "checkpoint_sha256": expected_sha,
        "source_commit": revision,
        "windows_sha256": sha256_file(args.windows),
        "features": [],
        "runner_sha256": sha256_file(Path(__file__)),
        "python": sys.version,
        "platform": platform.platform(),
        "device": "cpu",
        "precision": "float32",
        "training": False,
    }
    try:
        if args.modality == "audio":
            import torch
            import torchaudio
            from models.dymn.model import get_model
            from models.preprocess import AugmentMelSTFT

            torch.set_num_threads(4)
            with contextlib.redirect_stdout(io.StringIO()):
                model = get_model(width_mult=0.4, pretrained_name=None, T_max=1.0)
                mel = AugmentMelSTFT().eval()
            model.load_state_dict(
                torch.load(args.checkpoint, map_location="cpu", weights_only=True), strict=True
            )
            model.eval().requires_grad_(False)
            report.update(
                parameters=sum(p.numel() for p in model.parameters()),
                preprocessing="channel 0; 24k->32k torchaudio sinc; official eval mel",
                torch=torch.__version__,
                torchaudio=torchaudio.__version__,
            )

            def encode(data):
                waveform = torch.from_numpy(data["audio"][:, 0].copy()).unsqueeze(0)
                waveform = torchaudio.functional.resample(waveform, 24000, 32000)
                with torch.inference_mode():
                    _, feature = model(mel(waveform).unsqueeze(1))
                return feature.squeeze(0).numpy()

            expected_shape = (384,)
        else:
            import jax
            import jax.numpy as jnp
            import tensorflow as tf
            from videoprism import models as vp

            model = vp.get_model("videoprism_public_v1_base")
            variables = vp.load_pretrained_weights(
                "videoprism_public_v1_base", checkpoint_path=str(args.checkpoint)
            )
            forward = jax.jit(lambda x: model.apply(variables, x, train=False))
            report.update(
                parameters=sum(x.size for x in jax.tree_util.tree_leaves(variables)),
                model_revision=MODEL_REVISION,
                jax=jax.__version__,
                tensorflow=tf.__version__,
                preprocessing="full equirectangular frame bilinear resize 288x288 /255",
            )

            def encode(data):
                frames = (
                    np.asarray(tf.image.resize(data["video"], (288, 288)), dtype=np.float32) / 255
                )
                result = forward(jnp.asarray(frames[None]))
                tokens = result[0] if isinstance(result, tuple) else result
                return np.asarray(tokens).mean(axis=1).squeeze(0).astype(np.float32)

            expected_shape = (768,)
        report["load_seconds"] = time.perf_counter() - started
        for row in rows:
            tick = time.perf_counter()
            with np.load(row["path"], allow_pickle=False) as data:
                feature = encode(data)
                if feature.shape != expected_shape or not np.isfinite(feature).all():
                    raise ValueError("invalid frozen feature")
                elapsed = time.perf_counter() - tick
                repeat = encode(data)
            delta = float(np.max(np.abs(feature - repeat)))
            if not np.allclose(feature, repeat, rtol=1e-5, atol=1e-6):
                raise ValueError("repeated extraction mismatch")
            path = args.output_dir / f"feature-{len(report['features']):03d}.npy"
            np.save(path, feature, allow_pickle=False)
            if not np.array_equal(feature, np.load(path, allow_pickle=False)):
                raise ValueError("feature reload mismatch")
            report["features"].append(
                {
                    "id": row["id"],
                    "window_sha256": row["sha256"],
                    "path": str(path.resolve()),
                    "sha256": sha256_file(path),
                    "shape": list(feature.shape),
                    "seconds": elapsed,
                    "repeat_max_abs_diff": delta,
                }
            )
        report["status"] = "PASS"
    except Exception as error:
        report["status"] = "FAIL"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["wall_seconds"] = time.perf_counter() - started
        report["peak_working_set_bytes"] = getattr(process.memory_info(), "peak_wset", None)
        (args.output_dir / "result.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(report))


if __name__ == "__main__":
    main()
