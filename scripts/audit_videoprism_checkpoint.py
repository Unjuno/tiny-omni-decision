"""Audit a pinned official VideoPrism checkpoint with the upstream loader."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MODEL_ID = "google/videoprism-base-f16r288"
MODEL_REVISION = "be719a406d563b66f0ac969e7c94bab8e997c81a"
SOURCE_COMMIT = "c6eba0ece3dd49a24ffc44582966997a4a59f1d8"
CHECKPOINT_SHA256 = "61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_checkpoint(path: Path, expected_sha256: str = CHECKPOINT_SHA256) -> str:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError("checkpoint SHA-256 does not match the pinned expected digest")
    return actual


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sha256", default=CHECKPOINT_SHA256)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--width", type=int, default=288)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = _args()
    run_started_utc = datetime.now(UTC).isoformat()
    checkpoint = args.checkpoint.resolve(strict=True)
    source_root = args.source_root.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {output}")
    if args.frames < 1 or args.height < 1 or args.width < 1:
        raise ValueError("frame and image dimensions must be positive")
    checkpoint_sha256 = validate_checkpoint(checkpoint, args.expected_sha256)
    source_commit = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if source_commit != SOURCE_COMMIT:
        raise ValueError(f"unexpected upstream source commit: {source_commit}")

    sys.path.insert(0, str(source_root))
    import jax
    import jax.numpy as jnp
    import numpy as np
    import psutil
    from videoprism import models as vp

    process = psutil.Process()
    memory_before = process.memory_info().rss
    started = time.perf_counter()
    model = vp.get_model("videoprism_public_v1_base")
    load_started = time.perf_counter()
    variables = vp.load_pretrained_weights(
        "videoprism_public_v1_base", checkpoint_path=str(checkpoint)
    )
    jax.tree_util.tree_map(
        lambda value: value.block_until_ready() if hasattr(value, "block_until_ready") else value,
        variables,
    )
    load_seconds = time.perf_counter() - load_started
    leaves = jax.tree_util.tree_leaves(variables)
    parameter_count = sum(int(leaf.size) for leaf in leaves)
    dtype_counts: dict[str, int] = {}
    for leaf in leaves:
        dtype = str(leaf.dtype)
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + int(leaf.size)

    video = np.random.default_rng(args.seed).random(
        (1, args.frames, args.height, args.width, 3), dtype=np.float32
    )
    forward = jax.jit(lambda value: model.apply(variables, value, train=False))
    forward_started = time.perf_counter()
    result = forward(jnp.asarray(video))
    jax.tree_util.tree_map(
        lambda value: value.block_until_ready() if hasattr(value, "block_until_ready") else value,
        result,
    )
    forward_seconds = time.perf_counter() - forward_started
    features = result[0] if isinstance(result, tuple) else result
    features_np = np.asarray(features)
    if not np.isfinite(features_np).all():
        raise ValueError("VideoPrism output contains non-finite values")
    memory_after = process.memory_info()
    report: dict[str, Any] = {
        "status": "pass",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "upstream_source_commit": source_commit,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "parameter_count": parameter_count,
        "parameter_dtype_counts": dtype_counts,
        "input_shape": list(video.shape),
        "input_dtype": str(video.dtype),
        "seed": args.seed,
        "output_shape": list(features_np.shape),
        "output_dtype": str(features_np.dtype),
        "output_finite": True,
        "output_sha256": hashlib.sha256(features_np.tobytes()).hexdigest(),
        "load_seconds": load_seconds,
        "first_forward_compile_and_run_seconds": forward_seconds,
        "rss_start_bytes": memory_before,
        "rss_end_bytes": memory_after.rss,
        "peak_working_set_bytes": getattr(memory_after, "peak_wset", None),
        "devices": [str(device) for device in jax.devices()],
        "jax_version": jax.__version__,
        "wall_seconds": time.perf_counter() - started,
        "run_started_utc": run_started_utc,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
