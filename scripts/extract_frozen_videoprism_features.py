"""Extract scene-keyed frozen VideoPrism features for the bounded CLEVRER probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

MODEL_REVISION = "be719a406d563b66f0ac969e7c94bab8e997c81a"
SOURCE_COMMIT = "c6eba0ece3dd49a24ffc44582966997a4a59f1d8"
CHECKPOINT_SHA256 = "61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29"
FRAME_COUNT = 8
FRAME_SIZE = 288


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def uniform_indices(frame_count: int, requested: int = FRAME_COUNT) -> list[int]:
    if frame_count < 1 or requested < 1:
        raise ValueError("frame counts must be positive")
    if requested == 1:
        return [0]
    step = (frame_count - 1) / (requested - 1)
    return [round(index * step) for index in range(requested)]


def validate_split_rows(train: list[dict[str, Any]], validation: list[dict[str, Any]]) -> None:
    for name, rows in (("train", train), ("validation", validation)):
        if not rows or any(row.get("split") != name for row in rows):
            raise ValueError(f"empty or incorrectly marked {name} split")
        if len({row["id"] for row in rows}) != len(rows):
            raise ValueError(f"duplicate sample ID in {name} split")
        scenes_by_id: dict[int, str] = {}
        for row in rows:
            scene = int(row["scene_index"])
            media_hash = str(row["media_sha256"])
            previous = scenes_by_id.setdefault(scene, media_hash)
            if previous != media_hash:
                raise ValueError(f"scene {scene} maps to multiple media identities")
    train_scenes = {int(row["scene_index"]) for row in train}
    val_scenes = {int(row["scene_index"]) for row in validation}
    train_media = {str(row["media_sha256"]) for row in train}
    val_media = {str(row["media_sha256"]) for row in validation}
    if train_scenes & val_scenes or train_media & val_media:
        raise ValueError("CLEVRER scene or media identity overlaps between splits")
    train_content = {
        (row["question"].casefold().strip(), tuple(sorted(row["options"]))) for row in train
    }
    val_content = {
        (row["question"].casefold().strip(), tuple(sorted(row["options"]))) for row in validation
    }
    if train_content & val_content:
        raise ValueError("normalized question content overlaps between splits")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-sha256", default=CHECKPOINT_SHA256)
    parser.add_argument("--temporal-order", choices=("forward", "reverse"), default="forward")
    return parser.parse_args()


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _decode_video(path: Path) -> Any:
    import av
    import numpy as np

    with av.open(str(path)) as container:
        streams = [stream for stream in container.streams if stream.type == "video"]
        if not streams:
            raise ValueError(f"video has no video stream: {path}")
        stream = streams[0]
        total = int(stream.frames or 0)
        if total > 0:
            indices = set(uniform_indices(total))
            selected = []
            for index, frame in enumerate(container.decode(stream)):
                if index in indices:
                    selected.append(frame.to_ndarray(format="rgb24"))
                if index >= max(indices):
                    break
            if len(selected) != FRAME_COUNT:
                raise ValueError(f"decoder returned {len(selected)}/{FRAME_COUNT} frames: {path}")
        else:
            decoded = [frame.to_ndarray(format="rgb24") for frame in container.decode(stream)]
            if not decoded:
                raise ValueError(f"video contains no decodable frames: {path}")
            selected = [decoded[index] for index in uniform_indices(len(decoded))]
    return np.stack(selected)


def _preprocess(frames: Any, tf: Any) -> Any:
    """Use the upstream demo's resize-short-side and center-crop convention."""
    import numpy as np

    height, width = frames.shape[1:3]
    scale = max(FRAME_SIZE / height, FRAME_SIZE / width)
    new_height, new_width = round(height * scale), round(width * scale)
    resized = tf.image.resize(frames, (new_height, new_width), method="bilinear")
    cropped = tf.image.resize_with_crop_or_pad(resized, FRAME_SIZE, FRAME_SIZE)
    return np.asarray(cropped, dtype=np.float32) / 255.0


def _apply_temporal_order(frames: Any, temporal_order: str) -> Any:
    if temporal_order == "forward":
        return frames
    if temporal_order == "reverse":
        return frames[::-1].copy()
    raise ValueError(f"unsupported temporal order: {temporal_order}")


def main() -> None:
    args = _args()
    sample_dir = args.sample_dir.resolve(strict=True)
    checkpoint = args.checkpoint.resolve(strict=True)
    source_root = args.source_root.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    if sha256_file(checkpoint) != args.expected_sha256:
        raise ValueError("VideoPrism checkpoint SHA-256 does not match expected digest")
    upstream_commit = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if upstream_commit != SOURCE_COMMIT:
        raise ValueError(f"unexpected VideoPrism source commit: {upstream_commit}")
    train = _read_rows(sample_dir / "train.jsonl")
    validation = _read_rows(sample_dir / "validation.jsonl")
    validate_split_rows(train, validation)
    for rows in (train, validation):
        for row in rows:
            media = sample_dir / row["media_path"]
            if not media.is_file() or sha256_file(media) != row["media_sha256"]:
                raise ValueError(f"missing or hash-mismatched media: {media}")

    sys.path.insert(0, str(source_root))
    import jax
    import jax.numpy as jnp
    import numpy as np
    import psutil
    import tensorflow as tf
    from videoprism import models as vp

    model = vp.get_model("videoprism_public_v1_base")
    variables = vp.load_pretrained_weights(
        "videoprism_public_v1_base", checkpoint_path=str(checkpoint)
    )
    encode = jax.jit(lambda video: model.apply(variables, video, train=False))
    process = psutil.Process()
    start_rss = process.memory_info().rss
    started = time.perf_counter()
    features_by_media: dict[str, Any] = {}
    per_video_seconds = []
    for row in train + validation:
        media_hash = str(row["media_sha256"])
        if media_hash in features_by_media:
            continue
        frames = _apply_temporal_order(
            _decode_video(sample_dir / row["media_path"]), args.temporal_order
        )
        video = jnp.asarray(_preprocess(frames, tf)[None, ...])
        forward_started = time.perf_counter()
        result = encode(video)
        jax.tree_util.tree_map(
            lambda value: (
                value.block_until_ready() if hasattr(value, "block_until_ready") else value
            ),
            result,
        )
        per_video_seconds.append(time.perf_counter() - forward_started)
        tokens = result[0] if isinstance(result, tuple) else result
        pooled = np.asarray(tokens).mean(axis=1).squeeze(0).astype(np.float32)
        if pooled.shape != (768,) or not np.isfinite(pooled).all():
            raise ValueError(f"invalid pooled feature for {row['id']}: {pooled.shape}")
        features_by_media[media_hash] = pooled

    output_dir.mkdir(parents=True, exist_ok=False)
    cache_path = output_dir / "video-features.npz"
    media_hashes = sorted(features_by_media)
    np.savez_compressed(
        cache_path,
        media_sha256=np.asarray(media_hashes),
        features=np.stack([features_by_media[key] for key in media_hashes]),
    )
    cache_report = {
        "status": "complete_videoprism_frozen_feature_cache",
        "model_id": "google/videoprism-base-f16r288",
        "model_revision": MODEL_REVISION,
        "upstream_source_commit": upstream_commit,
        "checkpoint_sha256": args.expected_sha256,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "dataset_revision": "MIT-IBM/CLEVRER@98b842082ba4f7c18b6b9e3f39145871782a65ef",
        "sample_manifest_sha256": {
            "train": sha256_file(sample_dir / "train.jsonl"),
            "validation": sha256_file(sample_dir / "validation.jsonl"),
            "fetch_report": sha256_file(sample_dir / "fetch-report.json"),
        },
        "train_ids": [row["id"] for row in train],
        "validation_ids": [row["id"] for row in validation],
        "train_scene_ids": [row["scene_index"] for row in train],
        "validation_scene_ids": [row["scene_index"] for row in validation],
        "train_media_hashes": sorted({row["media_sha256"] for row in train}),
        "validation_media_hashes": sorted({row["media_sha256"] for row in validation}),
        "unique_video_count": len(features_by_media),
        "frame_count": FRAME_COUNT,
        "frame_size": FRAME_SIZE,
        "temporal_order": args.temporal_order,
        "preprocessing": (
            "uniform endpoint-inclusive frame indices; upstream resize-short-side "
            "then center crop; float32 RGB [0,1]"
        ),
        "pooling": "mean across all spatial-temporal output tokens to 768 dimensions",
        "feature_cache_file": cache_path.name,
        "feature_cache_sha256": sha256_file(cache_path),
        "feature_cache_bytes": cache_path.stat().st_size,
        "feature_dimension": 768,
        "per_unique_video_forward_seconds": per_video_seconds,
        "feature_extraction_seconds": time.perf_counter() - started,
        "peak_working_set_bytes": getattr(process.memory_info(), "peak_wset", None),
        "rss_start_bytes": start_rss,
        "rss_end_bytes": process.memory_info().rss,
        "jax_version": jax.__version__,
        "devices": [str(device) for device in jax.devices()],
        "validation_scene_overlap": 0,
    }
    (output_dir / "cache-report.json").write_text(
        json.dumps(cache_report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(cache_report, indent=2))


if __name__ == "__main__":
    main()
