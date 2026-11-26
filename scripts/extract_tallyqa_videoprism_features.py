"""Extract pinned frozen VideoPrism features for the screened TallyQA VG task."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from scripts.extract_frozen_videoprism_clevr4_features import (
    CHECKPOINT_SHA256,
    FRAME_COUNT,
    FRAME_SIZE,
    MODEL_REVISION,
    _load_video_prism,
    _preprocess_rgb_image,
    sha256_file,
)


def _rows(task: dict[str, Any], split: str) -> list[dict[str, Any]]:
    by_image: dict[int, dict[str, Any]] = {}
    for row in task[split]:
        image_id = int(row["image_id"])
        entry = by_image.setdefault(
            image_id,
            {"id": str(image_id), "split": split, "media_sha256": row["media_sha256"]},
        )
        if entry["media_sha256"] != row["media_sha256"]:
            raise ValueError("one image identity maps to multiple media hashes")
    return list(by_image.values())


def extract_split(
    rows: list[dict[str, Any]], media_root: Path, encode: Any, jax: Any, batch_size: int
) -> tuple[list[str], list[str], np.ndarray, list[float]]:
    ids: list[str] = []
    hashes: list[str] = []
    chunks: list[np.ndarray] = []
    seconds: list[float] = []
    started = time.perf_counter()
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        videos = []
        for row in batch_rows:
            image_id = row["id"]
            image_path = media_root / row["split"] / f"{image_id}.jpg"
            digest = sha256_file(image_path)
            if digest != row["media_sha256"]:
                raise ValueError(f"image hash mismatch: {image_id}")
            image = tf.io.decode_jpeg(tf.io.read_file(str(image_path)), channels=3)
            videos.append(_preprocess_rgb_image(image, tf))
            ids.append(image_id)
            hashes.append(digest)
        batch = np.stack(videos)
        if len(batch_rows) < batch_size:
            batch = np.concatenate(
                [batch, np.repeat(batch[-1:], batch_size - len(batch_rows), axis=0)]
            )
        t0 = time.perf_counter()
        result = encode(jax.numpy.asarray(batch))
        jax.tree_util.tree_map(
            lambda value: (
                value.block_until_ready() if hasattr(value, "block_until_ready") else value
            ),
            result,
        )
        seconds.append(time.perf_counter() - t0)
        tokens = result[0] if isinstance(result, tuple) else result
        pooled = np.asarray(tokens).mean(axis=1).astype(np.float32)[: len(batch_rows)]
        if pooled.shape != (len(batch_rows), 768) or not np.isfinite(pooled).all():
            raise ValueError("invalid VideoPrism pooled feature batch")
        chunks.append(pooled)
        if (start // batch_size + 1) % 100 == 0 or start + len(batch_rows) == len(rows):
            elapsed = time.perf_counter() - started
            print(
                json.dumps(
                    {
                        "event": "progress",
                        "split": batch_rows[0]["split"],
                        "images": len(ids),
                        "total": len(rows),
                        "seconds": elapsed,
                    }
                ),
                flush=True,
            )
    return ids, hashes, np.concatenate(chunks), seconds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-manifest", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    task_path = args.task_manifest.resolve(strict=True)
    media_root = args.media_root.resolve(strict=True)
    checkpoint = args.checkpoint.resolve(strict=True)
    source_root = args.source_root.resolve(strict=True)
    output = args.output_dir.resolve()
    if output.exists() or args.batch_size != 4:
        raise ValueError("output must be new and batch size must remain four")
    task = json.loads(task_path.read_text(encoding="utf-8"))
    if task["task"]["official_test_opened"] is not False:
        raise ValueError("invalid audit isolation declaration")
    train_rows, val_rows = _rows(task, "train"), _rows(task, "validation")
    train_ids = {row["id"] for row in train_rows}
    val_ids = {row["id"] for row in val_rows}
    train_hashes = {row["media_sha256"] for row in train_rows}
    val_hashes = {row["media_sha256"] for row in val_rows}
    if (
        train_ids & val_ids
        or train_hashes & val_hashes
        or len(train_rows) != 2795
        or len(val_rows) != 699
    ):
        raise ValueError("frozen image split integrity/count check failed")
    output.mkdir(parents=True)
    import psutil

    process = psutil.Process()
    rss_start = process.memory_info().rss
    encode, jax, source_commit = _load_video_prism(checkpoint, source_root)
    start = time.perf_counter()
    train = extract_split(train_rows, media_root, encode, jax, args.batch_size)
    validation = extract_split(val_rows, media_root, encode, jax, args.batch_size)
    cache = output / "features.npz"
    np.savez_compressed(
        cache,
        train_ids=np.asarray(train[0]),
        train_media_sha256=np.asarray(train[1]),
        train_features=train[2],
        validation_ids=np.asarray(validation[0]),
        validation_media_sha256=np.asarray(validation[1]),
        validation_features=validation[2],
    )
    report = {
        "status": "complete_tallyqa_vg_videoprism_features",
        "task_manifest_sha256": sha256_file(task_path),
        "task_name": task["task"]["name"],
        "source_commit": source_commit,
        "model_revision": MODEL_REVISION,
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_expected_sha256": CHECKPOINT_SHA256,
        "parameters": 114365184,
        "frozen": True,
        "frame_count": FRAME_COUNT,
        "frame_size": FRAME_SIZE,
        "static_image_policy": "repeat identical processed image across all 8 temporal slots",
        "pooling": "mean across VideoPrism output tokens to 768 float32 dimensions",
        "train_images": len(train[0]),
        "validation_images": len(validation[0]),
        "train_id_overlap": 0,
        "train_validation_media_hash_overlap": len(train_hashes & val_hashes),
        "cache_path": str(cache),
        "cache_sha256": sha256_file(cache),
        "cache_bytes": cache.stat().st_size,
        "batch_size": args.batch_size,
        "elapsed_seconds": time.perf_counter() - start,
        "train_forward_seconds": train[3],
        "validation_forward_seconds": validation[3],
        "python": sys.version,
        "jax": jax.__version__,
        "tensorflow": tf.__version__,
        "devices": [str(device) for device in jax.devices()],
        "rss_start_bytes": rss_start,
        "rss_end_bytes": process.memory_info().rss,
        "sealed_audit_loaded": False,
    }
    report_path = output / "cache-report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["cache_report_sha256"] = sha256_file(report_path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
