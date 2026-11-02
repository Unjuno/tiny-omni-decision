"""Persistent CPU-only VideoPrism image feature worker for streaming extraction."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve(strict=True)
    if sha256_file(checkpoint) != args.expected_sha256:
        raise ValueError("VideoPrism checkpoint hash mismatch")
    import subprocess

    source_root = args.source_root.resolve(strict=True)
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source_root, text=True
    ).strip()
    if source_commit != "c6eba0ece3dd49a24ffc44582966997a4a59f1d8":
        raise ValueError("unexpected VideoPrism source revision")
    sys.path.insert(0, str(source_root))
    import jax
    import jax.numpy as jnp
    import numpy as np
    import tensorflow as tf
    from videoprism import models as vp

    model = vp.get_model("videoprism_public_v1_base")
    variables = vp.load_pretrained_weights(
        "videoprism_public_v1_base", checkpoint_path=str(checkpoint)
    )
    encode = jax.jit(lambda video: model.apply(variables, video, train=False))
    print(
        json.dumps(
            {
                "status": "ready",
                "device": [str(device) for device in jax.devices()],
                "jax_version": jax.__version__,
                "tensorflow_version": tf.__version__,
                "source_commit": source_commit,
                "checkpoint_sha256": args.expected_sha256,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    for line in sys.stdin:
        request = json.loads(line)
        if request.get("command") == "stop":
            break
        sample_id = str(request["sample_id"])
        path = Path(request["path"]).resolve(strict=True)
        image = tf.io.decode_png(tf.io.read_file(str(path)), channels=3).numpy()
        height, width = image.shape[:2]
        scale = max(288 / height, 288 / width)
        resized_height, resized_width = round(height * scale), round(width * scale)
        frames = np.repeat(image[None, ...], 8, axis=0)
        values = tf.image.resize(frames, (resized_height, resized_width), method="bilinear")
        values = tf.image.resize_with_crop_or_pad(values, 288, 288)
        video = jnp.asarray(np.asarray(values, dtype=np.float32)[None, ...] / 255.0)
        started = time.perf_counter()
        result = encode(video)
        jax.tree_util.tree_map(
            lambda value: (
                value.block_until_ready() if hasattr(value, "block_until_ready") else value
            ),
            result,
        )
        tokens = result[0] if isinstance(result, tuple) else result
        feature = np.asarray(tokens).mean(axis=1).squeeze(0).astype(np.float32)
        if feature.shape != (768,) or not np.isfinite(feature).all():
            raise ValueError(f"invalid feature for {sample_id}: {feature.shape}")
        print(
            json.dumps(
                {
                    "sample_id": sample_id,
                    "feature": feature.tolist(),
                    "forward_seconds": time.perf_counter() - started,
                },
                allow_nan=False,
                separators=(",", ":"),
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
