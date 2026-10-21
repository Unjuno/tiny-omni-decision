"""Encode one local video to the pinned pooled VideoPrism Path A feature."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from scripts.extract_frozen_videoprism_features import (
    CHECKPOINT_SHA256,
    MODEL_REVISION,
    SOURCE_COMMIT,
    _decode_video,
    _preprocess,
    sha256_file,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-path", type=Path, required=True)
    parser.add_argument("--expected-media-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    video = args.video_path.resolve(strict=True)
    checkpoint = args.checkpoint.resolve(strict=True)
    source_root = args.source_root.resolve(strict=True)
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite VideoPrism feature output: {output}")
    if sha256_file(video) != args.expected_media_sha256:
        raise ValueError("video media hash does not match the frozen sample manifest")
    if sha256_file(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("VideoPrism checkpoint hash mismatch")
    source_commit = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if source_commit != SOURCE_COMMIT:
        raise ValueError(f"unexpected VideoPrism source revision: {source_commit}")

    sys.path.insert(0, str(source_root))
    import jax
    import jax.numpy as jnp
    import numpy as np
    import psutil
    import tensorflow as tf
    from videoprism import models as vp

    process = psutil.Process()
    rss_start = process.memory_info().rss
    started = time.perf_counter()
    frames = _decode_video(video)
    processed = _preprocess(frames, tf)
    model = vp.get_model("videoprism_public_v1_base")
    variables = vp.load_pretrained_weights(
        "videoprism_public_v1_base", checkpoint_path=str(checkpoint)
    )
    encode = jax.jit(lambda clip: model.apply(variables, clip, train=False))
    forward_started = time.perf_counter()
    result = encode(jnp.asarray(processed[None, ...]))
    jax.tree_util.tree_map(
        lambda value: value.block_until_ready() if hasattr(value, "block_until_ready") else value,
        result,
    )
    forward_seconds = time.perf_counter() - forward_started
    tokens = result[0] if isinstance(result, tuple) else result
    feature = np.asarray(tokens).mean(axis=1).squeeze(0).astype(np.float32)
    if feature.shape != (768,) or not np.isfinite(feature).all():
        raise ValueError("VideoPrism produced an invalid pooled video feature")

    output.mkdir(parents=True)
    feature_path = output / "video-feature.npy"
    np.save(feature_path, feature, allow_pickle=False)
    report = {
        "status": "complete_single_video_feature",
        "created_utc": datetime.now(UTC).isoformat(),
        "video_sha256": args.expected_media_sha256,
        "video_model_revision": MODEL_REVISION,
        "video_checkpoint_sha256": CHECKPOINT_SHA256,
        "videoprism_source_commit": source_commit,
        "frame_count": 8,
        "frame_size": 288,
        "preprocessing": (
            "uniform endpoint-inclusive frames, resize-short-side, center crop, "
            "RGB float32 [0,1]"
        ),
        "pooling": "mean across all spatial-temporal VideoPrism tokens",
        "feature_shape": list(feature.shape),
        "feature_sha256": sha256_file(feature_path),
        "feature_file_bytes": feature_path.stat().st_size,
        "device": [str(device) for device in jax.devices()],
        "forward_seconds_after_model_load": forward_seconds,
        "process_seconds_including_model_load": time.perf_counter() - started,
        "rss_start_bytes": rss_start,
        "rss_end_bytes": process.memory_info().rss,
        "peak_working_set_bytes": getattr(process.memory_info(), "peak_wset", None),
        "jax_version": jax.__version__,
        "tensorflow_version": tf.__version__,
        "sealed_audit_loaded": False,
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
