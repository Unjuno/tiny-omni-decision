"""Extract frame-position-pooled frozen VideoPrism features for CLEVRER."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from scripts.extract_frozen_videoprism_features import (
    CHECKPOINT_SHA256,
    FRAME_COUNT,
    FRAME_SIZE,
    MODEL_REVISION,
    SOURCE_COMMIT,
    _apply_temporal_order,
    _decode_video,
    _preprocess,
    sha256_file,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-sample-dir", type=Path, required=True)
    parser.add_argument("--evaluation-sample-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-count", type=int, default=FRAME_COUNT)
    return parser.parse_args()


def pool_framewise_tokens(tokens: np.ndarray, frame_count: int = FRAME_COUNT) -> np.ndarray:
    """Return `[T,D]` by averaging spatial positions after temporal attention."""
    values = np.asarray(tokens, dtype=np.float32)
    if values.ndim != 3 or values.shape[0] != 1 or values.shape[2] != 768:
        raise ValueError("VideoPrism tokens must have shape [1, T*N, 768]")
    if frame_count < 1 or values.shape[1] % frame_count:
        raise ValueError("token sequence must contain an equal spatial grid per frame")
    framewise = values.reshape(1, frame_count, -1, 768).mean(axis=2).squeeze(0)
    if framewise.shape != (frame_count, 768) or not np.isfinite(framewise).all():
        raise ValueError("invalid framewise VideoPrism features")
    return framewise.astype(np.float32, copy=False)


def center_framewise_features(
    framewise: np.ndarray, reference_mean: np.ndarray, tolerance: float = 1e-5
) -> tuple[np.ndarray, float]:
    """Apply one common offset so uniform pooling matches a frozen reference."""
    values = np.asarray(framewise, dtype=np.float32)
    reference = np.asarray(reference_mean, dtype=np.float32)
    if values.shape != (FRAME_COUNT, 768) or reference.shape != (768,):
        raise ValueError("expected framewise [8,768] and reference [768] features")
    if not np.isfinite(values).all() or not np.isfinite(reference).all():
        raise ValueError("features must be finite")
    centered = values.copy()
    for _ in range(3):
        residual = reference - centered.mean(axis=0)
        centered += residual[None, :]
        max_delta = float(np.max(np.abs(centered.mean(axis=0) - reference)))
        if max_delta <= tolerance:
            return centered, max_delta
    raise ValueError(f"uniform mean cannot be centered within {tolerance}: {max_delta}")


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _validate_cross_sample(train: list[dict[str, Any]], evaluation: list[dict[str, Any]]) -> None:
    for name, rows, expected_split in (
        ("training", train, "train"),
        ("evaluation", evaluation, "validation"),
    ):
        if not rows or any(row.get("split") != expected_split for row in rows):
            raise ValueError(f"{name} rows must be non-empty and marked {expected_split}")
        if len({str(row["id"]) for row in rows}) != len(rows):
            raise ValueError(f"duplicate sample ID in {name} rows")
        scene_media: dict[int, str] = {}
        for row in rows:
            scene = int(row["scene_index"])
            media = str(row["media_sha256"])
            if scene in scene_media and scene_media[scene] != media:
                raise ValueError(f"scene {scene} maps to multiple media assets")
            scene_media[scene] = media
    train_scenes = {int(row["scene_index"]) for row in train}
    evaluation_scenes = {int(row["scene_index"]) for row in evaluation}
    train_media = {str(row["media_sha256"]) for row in train}
    evaluation_media = {str(row["media_sha256"]) for row in evaluation}
    if train_scenes & evaluation_scenes or train_media & evaluation_media:
        raise ValueError("train/evaluation scene or media identity overlap")


def _unique_asset_rows(
    splits: list[tuple[Path, list[dict[str, Any]]]],
) -> dict[str, tuple[Path, dict[str, Any]]]:
    unique_rows: dict[str, tuple[Path, dict[str, Any]]] = {}
    for sample_dir, rows in splits:
        for row in rows:
            media = sample_dir / str(row["media_path"])
            media_hash = str(row["media_sha256"])
            if not media.is_file() or sha256_file(media) != media_hash:
                raise ValueError(f"missing or hash-mismatched media: {media}")
            unique_rows.setdefault(media_hash, (media, row))
    return unique_rows


def main() -> None:
    args = _args()
    train_dir = args.train_sample_dir.resolve(strict=True)
    evaluation_dir = args.evaluation_sample_dir.resolve(strict=True)
    checkpoint = args.checkpoint.resolve(strict=True)
    source_root = args.source_root.resolve(strict=True)
    output = args.output_dir.resolve()
    if args.frame_count < 1:
        raise ValueError("frame count must be positive")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite framewise feature directory: {output}")
    if sha256_file(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("VideoPrism checkpoint SHA-256 mismatch")
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if source_commit != SOURCE_COMMIT:
        raise ValueError(f"unexpected VideoPrism source commit: {source_commit}")

    train_rows = _read_rows(train_dir / "train.jsonl")
    evaluation_rows = _read_rows(evaluation_dir / "validation.jsonl")
    if not train_rows or not evaluation_rows:
        raise ValueError("train and evaluation splits must be non-empty")
    _validate_cross_sample(train_rows, evaluation_rows)
    unique_rows = _unique_asset_rows([(train_dir, train_rows), (evaluation_dir, evaluation_rows)])

    sys.path.insert(0, str(source_root))
    import jax
    import jax.numpy as jnp
    import psutil
    import tensorflow as tf
    from videoprism import models as vp

    model = vp.get_model("videoprism_public_v1_base")
    variables = vp.load_pretrained_weights(
        "videoprism_public_v1_base", checkpoint_path=str(checkpoint)
    )
    encode = jax.jit(lambda video: model.apply(variables, video, train=False))
    process = psutil.Process()
    rss_start = process.memory_info().rss
    started = time.perf_counter()
    features_by_media: dict[str, np.ndarray] = {}
    per_video_seconds = []
    for media_hash, (media, row) in unique_rows.items():
        frames = _apply_temporal_order(_decode_video(media, requested=args.frame_count), "forward")
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
        features = pool_framewise_tokens(np.asarray(tokens), frame_count=args.frame_count)
        pooled = features.mean(axis=0)
        if pooled.shape != (768,) or not np.isfinite(pooled).all():
            raise ValueError(f"invalid pooled feature for {row['id']}")
        features_by_media[media_hash] = features

    output.mkdir(parents=True, exist_ok=False)
    cache_path = output / "framewise-features.npz"
    media_hashes = sorted(features_by_media)
    np.savez_compressed(
        cache_path,
        media_sha256=np.asarray(media_hashes),
        features=np.stack([features_by_media[key] for key in media_hashes]),
    )
    report = {
        "status": "complete_videoprism_framewise_feature_cache",
        "model_id": "google/videoprism-base-f16r288",
        "model_revision": MODEL_REVISION,
        "upstream_source_commit": source_commit,
        "checkpoint_sha256": sha256_file(checkpoint),
        "dataset_revision": "MIT-IBM/CLEVRER@98b842082ba4f7c18b6b9e3f39145871782a65ef",
        "frame_count": args.frame_count,
        "frame_size": FRAME_SIZE,
        "feature_dimension": 768,
        "feature_layout": "[unique_media, temporal_position, hidden_dimension]",
        "pooling": "mean over spatial token positions separately at each output frame position",
        "preprocessing": (
            "uniform endpoint-inclusive frame indices; upstream resize-short-side "
            "then center crop; float32 RGB [0,1]"
        ),
        "train_sample_dir": str(train_dir),
        "train_manifest_sha256": {
            name: sha256_file(train_dir / name)
            for name in ("train.jsonl", "validation.jsonl", "fetch-report.json")
        },
        "evaluation_sample_dir": str(evaluation_dir),
        "evaluation_manifest_sha256": {
            name: sha256_file(evaluation_dir / name)
            for name in ("train.jsonl", "validation.jsonl", "fetch-report.json")
        },
        "train_ids": [row["id"] for row in train_rows],
        "evaluation_ids": [row["id"] for row in evaluation_rows],
        "train_scene_ids": sorted({int(row["scene_index"]) for row in train_rows}),
        "evaluation_scene_ids": sorted({int(row["scene_index"]) for row in evaluation_rows}),
        "train_media_hashes": sorted({str(row["media_sha256"]) for row in train_rows}),
        "evaluation_media_hashes": sorted({str(row["media_sha256"]) for row in evaluation_rows}),
        "unique_video_count": len(features_by_media),
        "per_unique_video_forward_seconds": per_video_seconds,
        "extraction_seconds": time.perf_counter() - started,
        "peak_working_set_bytes": getattr(process.memory_info(), "peak_wset", None),
        "rss_start_bytes": rss_start,
        "rss_end_bytes": process.memory_info().rss,
        "jax_version": jax.__version__,
        "devices": [str(device) for device in jax.devices()],
        "feature_cache_file": cache_path.name,
        "feature_cache_sha256": sha256_file(cache_path),
        "feature_cache_bytes": cache_path.stat().st_size,
        "validation_scene_overlap": 0,
    }
    (output / "cache-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
