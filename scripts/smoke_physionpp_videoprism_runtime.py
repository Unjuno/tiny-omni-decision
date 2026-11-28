"""Run a CPU-only VideoPrism feature smoke on one deterministic TRAIN clip."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import numpy as np

MODEL_REVISION = "be719a406d563b66f0ac969e7c94bab8e997c81a"
SOURCE_COMMIT = "c6eba0ece3dd49a24ffc44582966997a4a59f1d8"
CHECKPOINT_SHA256 = "61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29"
FRAME_COUNT = 8
FRAME_SIZE = 288


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def uniform_indices(frame_count: int, requested: int = FRAME_COUNT) -> list[int]:
    if frame_count < 2 or requested < 2:
        raise ValueError("frame counts must be at least two")
    return sorted({round(i * (frame_count - 1) / (requested - 1)) for i in range(requested)})


def _safe_media_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value.replace("\\", "/"))
    if (
        path.is_absolute()
        or PureWindowsPath(value).is_absolute()
        or ".." in path.parts
        or not path.parts
    ):
        raise ValueError(f"unsafe media path: {value}")
    return path


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows or any(row.get("split") != "train" for row in rows):
        raise ValueError("smoke requires a non-empty TRAIN-only manifest")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("duplicate sample IDs in train manifest")
    return rows


def run_smoke(
    *,
    train_path: Path,
    corpus_root: Path,
    checkpoint: Path,
    source_root: Path,
    output: Path,
    sample_id: str | None = None,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    if sha256_file(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("VideoPrism checkpoint SHA-256 mismatch")
    actual_commit = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual_commit != SOURCE_COMMIT:
        raise ValueError(f"unexpected VideoPrism source revision: {actual_commit}")

    rows = _read_rows(train_path)
    row = next((item for item in rows if item["id"] == sample_id), None) if sample_id else rows[0]
    if row is None:
        raise ValueError(f"sample ID not found in train manifest: {sample_id}")
    if row.get("modality") != "video" or row.get("source") != "physionpp/readout":
        raise ValueError("selected record is not a Physion++ video Decision")
    if len(row.get("media", [])) != 1 or row["media"][0].get("kind") != "video":
        raise ValueError("selected Decision must contain exactly one video")
    media = row["media"][0]
    video_path = corpus_root.joinpath(*_safe_media_path(media["path"]).parts)
    if not video_path.is_file() or sha256_file(video_path) != media["sha256"]:
        raise ValueError("selected training video is missing or hash-mismatched")

    import av
    import jax
    import jax.numpy as jnp
    import psutil
    import tensorflow as tf
    from videoprism import models as vp

    frame_indices = set(uniform_indices(int(media["num_frames"])))
    selected = []
    decoded = 0
    with av.open(str(video_path)) as container:
        stream = next((item for item in container.streams if item.type == "video"), None)
        if stream is None:
            raise ValueError("selected video has no video stream")
        for index, frame in enumerate(container.decode(stream)):
            decoded = index + 1
            if index in frame_indices:
                selected.append(frame.to_ndarray(format="rgb24"))
    if decoded != int(media["num_frames"]) or len(selected) != len(frame_indices):
        raise ValueError("decoded video frame count differs from the train manifest")

    started = time.perf_counter()
    process = psutil.Process()
    rss_before_model = process.memory_info().rss
    model = vp.get_model("videoprism_public_v1_base")
    variables = vp.load_pretrained_weights(
        "videoprism_public_v1_base", checkpoint_path=str(checkpoint)
    )
    encode = jax.jit(lambda video: model.apply(variables, video, train=False))

    height, width = selected[0].shape[:2]
    scale = max(FRAME_SIZE / height, FRAME_SIZE / width)
    resized_shape = (round(height * scale), round(width * scale))
    resized = tf.image.resize(np.stack(selected), resized_shape, method="bilinear")
    frames = tf.image.resize_with_crop_or_pad(resized, FRAME_SIZE, FRAME_SIZE)
    video = jnp.asarray(np.asarray(frames, dtype=np.float32)[None, ...] / 255.0)
    result = encode(video)
    jax.tree_util.tree_map(
        lambda value: value.block_until_ready() if hasattr(value, "block_until_ready") else value,
        result,
    )
    tokens = result[0] if isinstance(result, tuple) else result
    pooled = np.asarray(tokens).mean(axis=1).squeeze(0).astype(np.float32)
    if pooled.shape != (768,) or not np.isfinite(pooled).all():
        raise ValueError(f"unexpected/non-finite pooled VideoPrism output: {pooled.shape}")
    elapsed = time.perf_counter() - started
    output.mkdir(parents=True, exist_ok=False)
    feature_path = output / "frozen-video-feature.npy"
    np.save(feature_path, pooled, allow_pickle=False)
    report = {
        "schema_version": 1,
        "status": "complete_single_train_clip_videoprism_cpu_smoke",
        "claim_scope": "single-clip CPU feature/runtime smoke only; not fitted or quality evidence",
        "created_utc": datetime.now(UTC).isoformat(),
        "sample_id": row["id"],
        "split": row["split"],
        "target": row["target"],
        "media_sha256": media["sha256"],
        "train_manifest_sha256": sha256_file(train_path),
        "model_id": "google/videoprism-base-f16r288",
        "model_revision": MODEL_REVISION,
        "source_commit": actual_commit,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "frame_count": FRAME_COUNT,
        "frame_indices": sorted(frame_indices),
        "frame_size": FRAME_SIZE,
        "pooled_feature_shape": list(pooled.shape),
        "feature_sha256": sha256_file(feature_path),
        "elapsed_seconds_including_model_load_and_forward": elapsed,
        "rss_before_model_load_bytes": rss_before_model,
        "rss_after_forward_bytes": process.memory_info().rss,
        "jax_version": jax.__version__,
        "tensorflow_version": tf.__version__,
        "pyav_version": av.__version__,
        "jax_devices": [str(device) for device in jax.devices()],
        "device_policy": "JAX_PLATFORMS=cpu; GPU not used",
    }
    (output / "smoke-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-id")
    args = parser.parse_args()
    report = run_smoke(
        train_path=args.train.resolve(strict=True),
        corpus_root=args.corpus_root.resolve(strict=True),
        checkpoint=args.checkpoint.resolve(strict=True),
        source_root=args.source_root.resolve(strict=True),
        output=args.output.resolve(),
        sample_id=args.sample_id,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
