"""Persistent frozen VideoPrism feature worker for DCASE event-centered clips."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

CHECKPOINT_SHA256 = "61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29"
SOURCE_COMMIT = "c6eba0ece3dd49a24ffc44582966997a4a59f1d8"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class VideoPrismWorker:
    def __init__(self, source: Path, checkpoint: Path):
        if sha256_file(checkpoint) != CHECKPOINT_SHA256:
            raise ValueError("VideoPrism checkpoint SHA-256 mismatch")
        import subprocess

        commit = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip()
        if commit != SOURCE_COMMIT:
            raise ValueError("VideoPrism source commit mismatch")
        sys.path.insert(0, str(source.resolve()))
        from videoprism import models as vp

        self.model = vp.get_model("videoprism_public_v1_base")
        self.variables = vp.load_pretrained_weights(
            "videoprism_public_v1_base", checkpoint_path=str(checkpoint)
        )
        self.forward = jax.jit(lambda images: self.model.apply(self.variables, images, train=False))

    def process(self, request: dict[str, object]) -> dict[str, object]:
        started = time.perf_counter()
        media_path = Path(str(request["media_npz"]))
        output_path = Path(str(request["output_npy"]))
        if not output_path.parent.is_dir():
            raise ValueError("VideoPrism feature output directory must exist")
        with np.load(media_path, allow_pickle=False) as archive:
            video = archive["video"]
            valid = archive["valid"]
            sample_ids = archive["sample_ids"].tolist()
        if (
            video.ndim != 5
            or video.shape[1:] != (8, 360, 640, 3)
            or valid.shape != (len(video),)
            or len(sample_ids) != len(video)
        ):
            raise ValueError("unexpected DCASE media batch shape or row count")

        flat = video.reshape((-1, 360, 640, 3))
        scale = max(288 / 360, 288 / 640)
        height, width = round(360 * scale), round(640 * scale)
        resized = tf.image.resize(flat, (height, width), method="bilinear")
        cropped = tf.image.resize_with_crop_or_pad(resized, 288, 288)
        images = np.asarray(cropped, dtype=np.float32).reshape((-1, 8, 288, 288, 3)) / 255.0
        result = self.forward(jnp.asarray(images))
        jax.tree_util.tree_map(
            lambda value: value.block_until_ready()
            if hasattr(value, "block_until_ready")
            else value,
            result,
        )
        tokens = result[0] if isinstance(result, tuple) else result
        features = np.asarray(tokens).mean(axis=1).astype(np.float32)
        if features.shape != (len(video), 768) or not np.isfinite(features).all():
            raise ValueError("VideoPrism returned invalid pooled features")
        np.save(output_path, features, allow_pickle=False)
        return {
            "status": "PASS",
            "batch_id": request.get("batch_id"),
            "sample_ids": sample_ids,
            "feature_shape": list(features.shape),
            "device": [str(device) for device in jax.devices()],
            "valid_count": int(valid.sum()),
            "wall_seconds": time.perf_counter() - started,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    worker = VideoPrismWorker(args.source, args.checkpoint)
    print(
        json.dumps(
            {
                "status": "READY",
                "source_commit": SOURCE_COMMIT,
                "checkpoint_sha256": CHECKPOINT_SHA256,
                "device": [str(device) for device in jax.devices()],
                "jax": jax.__version__,
                "tensorflow": tf.__version__,
                "parameters": sum(
                    int(value.size) for value in jax.tree_util.tree_leaves(worker.variables)
                ),
            }
        ),
        flush=True,
    )
    for line in sys.stdin:
        try:
            response = worker.process(json.loads(line))
        except Exception as error:
            response = {"status": "FAIL", "error": f"{type(error).__name__}: {error}"}
        print(json.dumps(response), flush=True)


if __name__ == "__main__":
    main()
