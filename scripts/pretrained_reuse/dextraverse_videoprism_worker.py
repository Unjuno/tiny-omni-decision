"""Dedicated pinned JAX worker for the Zarr reader's binary feature pipe."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

from scripts.extract_frozen_videoprism_features import (
    CHECKPOINT_SHA256,
    SOURCE_COMMIT,
    _preprocess,
    sha256_file,
)
from scripts.pretrained_reuse.dextraverse_features import (
    pool_videoprism_tokens,
    read_video_batch,
    write_video_features,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--device", choices=("gpu", "cpu"), required=True)
    parser.add_argument("--worker-report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _args()
    checkpoint = args.checkpoint.resolve(strict=True)
    source = args.source_root.resolve(strict=True)
    if sha256_file(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("VideoPrism checkpoint SHA-256 differs from the pinned checkpoint")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if commit != SOURCE_COMMIT:
        raise ValueError("VideoPrism source tree is not the pinned upstream commit")

    import jax
    import jax.numpy as jnp
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    devices = jax.devices("gpu" if args.device == "gpu" else "cpu")
    if not devices:
        raise RuntimeError(f"JAX has no {args.device} device")
    selected_device = devices[0]
    sys.path.insert(0, str(source))
    from videoprism import models as vp

    model = vp.get_model("videoprism_public_v1_base")
    variables = vp.load_pretrained_weights(
        "videoprism_public_v1_base", checkpoint_path=str(checkpoint)
    )
    encode = jax.jit(lambda video: model.apply(variables, video, train=False))
    args.worker_report.parent.mkdir(parents=True, exist_ok=True)
    args.worker_report.write_text(json.dumps({
        "status": "ready",
        "device": str(selected_device),
        "platform": selected_device.platform,
        "jax_version": jax.__version__,
        "tensorflow_version": tf.__version__,
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "source_commit": commit,
        "checkpoint_sha256": CHECKPOINT_SHA256,
    }, indent=2) + "\n", encoding="utf-8")
    sys.stderr.write(
        f"READY device={selected_device} checkpoint_sha256={CHECKPOINT_SHA256}\n"
    )
    sys.stderr.flush()
    source_stream = sys.stdin.buffer
    destination_stream = sys.stdout.buffer
    while True:
        batch = read_video_batch(source_stream)
        if batch is None:
            break
        preprocessed = np.stack([_preprocess(sample, tf) for sample in batch])
        with jax.default_device(selected_device):
            encoded = encode(jnp.asarray(preprocessed, dtype=jnp.float32))
            encoded = jax.tree_util.tree_map(
                lambda value: value.block_until_ready()
                if hasattr(value, "block_until_ready") else value,
                encoded,
            )
        tokens = encoded[0] if isinstance(encoded, tuple) else encoded
        features = np.stack([
            pool_videoprism_tokens(np.asarray(tokens[index:index + 1]))
            for index in range(batch.shape[0])
        ])
        write_video_features(destination_stream, features)


if __name__ == "__main__":
    main()
