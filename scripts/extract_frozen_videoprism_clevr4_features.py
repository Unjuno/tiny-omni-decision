"""Extract pinned VideoPrism-B features for the full CLEVR-4 train/validation corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

MODEL_REVISION = "be719a406d563b66f0ac969e7c94bab8e997c81a"
SOURCE_COMMIT = "c6eba0ece3dd49a24ffc44582966997a4a59f1d8"
CHECKPOINT_SHA256 = "61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29"
FRAME_COUNT = 8
FRAME_SIZE = 288
DATASET_REVISION = "sgvaze/clevr4@cddc78fb2a8359dc958987b2c750bfdd4bfd2c73"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--selection-sample-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ordered_ids_sha256(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def _validate_validation_partition(
    validation_ids: list[str], selection_ids: list[str]
) -> tuple[list[str], list[str]]:
    if len(validation_ids) != len(set(validation_ids)):
        raise ValueError("full validation IDs contain duplicates")
    if len(selection_ids) != len(set(selection_ids)):
        raise ValueError("checkpoint-selection IDs contain duplicates")
    missing = set(selection_ids) - set(validation_ids)
    if missing:
        raise ValueError(
            f"checkpoint-selection IDs are absent from full validation: {len(missing)}"
        )
    selection = set(selection_ids)
    ordered_selection = [image_id for image_id in validation_ids if image_id in selection]
    new_holdout = [image_id for image_id in validation_ids if image_id not in selection]
    if (set(ordered_selection) & set(new_holdout)) or (
        set(ordered_selection) | set(new_holdout) != set(validation_ids)
    ):
        raise AssertionError("validation selection/holdout partition is inconsistent")
    return ordered_selection, new_holdout


def _validate_full_splits(
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    train_assets: list[dict[str, Any]],
    validation_assets: list[dict[str, Any]],
) -> None:
    if [row["id"] for row in train_rows] != [row["id"] for row in train_assets]:
        raise ValueError("train labels and image manifest ID order differ")
    if [row["id"] for row in validation_rows] != [row["id"] for row in validation_assets]:
        raise ValueError("validation labels and image manifest ID order differ")
    if {row.get("split") for row in train_rows} != {"train"}:
        raise ValueError("train label file contains non-train rows")
    if {row.get("split") for row in validation_rows} != {"val"}:
        raise ValueError("validation label file contains non-validation rows")
    if {row["id"] for row in train_rows} & {row["id"] for row in validation_rows}:
        raise ValueError("train and validation image IDs overlap")
    if {row["sha256"] for row in train_assets} & {row["sha256"] for row in validation_assets}:
        raise ValueError("train and validation media hashes overlap")


def _preprocess_rgb_image(image: Any, tf: Any) -> np.ndarray:
    """Resize/crop one RGB image and repeat it over eight static video slots."""
    if len(image.shape) != 3 or image.shape[-1] != 3:
        raise ValueError("image must have HWC RGB shape")
    height, width = int(image.shape[0]), int(image.shape[1])
    if height < 1 or width < 1:
        raise ValueError("image dimensions must be positive")
    scale = max(FRAME_SIZE / height, FRAME_SIZE / width)
    new_height, new_width = round(height * scale), round(width * scale)
    resized = tf.image.resize(image, (new_height, new_width), method="bilinear")
    cropped = tf.image.resize_with_crop_or_pad(resized, FRAME_SIZE, FRAME_SIZE)
    frame = np.asarray(cropped, dtype=np.float32) / 255.0
    if frame.shape != (FRAME_SIZE, FRAME_SIZE, 3) or not np.isfinite(frame).all():
        raise ValueError("image preprocessing produced invalid pixels")
    return np.repeat(frame[None, ...], FRAME_COUNT, axis=0)


def _load_video_prism(checkpoint: Path, source_root: Path) -> tuple[Any, Any, str]:
    commit = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != SOURCE_COMMIT:
        raise ValueError(f"unexpected VideoPrism source commit: {commit}")
    if sha256_file(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("VideoPrism checkpoint SHA-256 does not match the pinned value")
    sys.path.insert(0, str(source_root))
    import jax
    from videoprism import models as vp

    model = vp.get_model("videoprism_public_v1_base")
    variables = vp.load_pretrained_weights(
        "videoprism_public_v1_base", checkpoint_path=str(checkpoint)
    )
    encode = jax.jit(lambda video: model.apply(variables, video, train=False))
    return encode, jax, commit


def _extract_split(
    rows: list[dict[str, Any]],
    split: str,
    sample_dir: Path,
    encode: Any,
    tf: Any,
    jax: Any,
    batch_size: int,
) -> tuple[list[str], list[str], np.ndarray, list[float], int]:
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    features: list[np.ndarray] = []
    ids: list[str] = []
    media_hashes: list[str] = []
    batch_seconds: list[float] = []
    padded_count = 0
    for start in range(0, len(rows), batch_size):
        current = rows[start : start + batch_size]
        videos = []
        for row in current:
            if row.get("split") != split:
                raise ValueError(f"row {row.get('id')} is not marked as {split}")
            image_id = str(row["id"])
            path = sample_dir / split / "images" / f"{image_id}.png"
            if not path.is_file():
                raise FileNotFoundError(path)
            actual_hash = sha256_file(path)
            if actual_hash != row["sha256"]:
                raise ValueError(f"CLEVR-4 media hash mismatch: {image_id}")
            image = tf.io.decode_png(tf.io.read_file(str(path)), channels=3)
            videos.append(_preprocess_rgb_image(image, tf))
            ids.append(image_id)
            media_hashes.append(actual_hash)
        batch = np.stack(videos)
        pad = batch_size - len(current)
        if pad:
            batch = np.concatenate([batch, np.repeat(batch[-1:], pad, axis=0)], axis=0)
            padded_count += pad
        started = time.perf_counter()
        result = encode(jax.numpy.asarray(batch))
        jax.tree_util.tree_map(
            lambda value: (
                value.block_until_ready() if hasattr(value, "block_until_ready") else value
            ),
            result,
        )
        batch_seconds.append(time.perf_counter() - started)
        tokens = result[0] if isinstance(result, tuple) else result
        pooled = np.asarray(tokens).mean(axis=1).astype(np.float32)
        pooled = pooled[: len(current)]
        if pooled.shape != (len(current), 768) or not np.isfinite(pooled).all():
            raise ValueError(f"invalid VideoPrism features for {split} batch at {start}")
        features.append(pooled)
    if len(ids) != len(rows):
        raise AssertionError("feature rows do not cover split IDs")
    return ids, media_hashes, np.concatenate(features), batch_seconds, padded_count


def main() -> None:
    args = _args()
    sample_dir = args.sample_dir.resolve(strict=True)
    selection_dir = args.selection_sample_dir.resolve(strict=True)
    checkpoint = args.checkpoint.resolve(strict=True)
    source_root = args.source_root.resolve(strict=True)
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite VideoPrism feature output: {output}")

    train_labels_path = sample_dir / "train-labels.jsonl"
    val_labels_path = sample_dir / "val-labels.jsonl"
    train_assets_path = sample_dir / "train-image-manifest.jsonl"
    val_assets_path = sample_dir / "val-image-manifest.jsonl"
    selection_manifest_path = selection_dir / "val-image-manifest.jsonl"
    train_rows = _read_jsonl(train_labels_path)
    validation_rows = _read_jsonl(val_labels_path)
    train_assets = _read_jsonl(train_assets_path)
    validation_assets = _read_jsonl(val_assets_path)
    selection_assets = _read_jsonl(selection_manifest_path)
    if len(train_rows) != 8424 or len(validation_rows) != 2107:
        raise ValueError("full CLEVR-4 input must contain 8,424 train and 2,107 validation images")
    _validate_full_splits(train_rows, validation_rows, train_assets, validation_assets)
    selection_order, new_holdout_order = _validate_validation_partition(
        [row["id"] for row in validation_rows], [row["id"] for row in selection_assets]
    )
    if len(selection_order) != 512 or len(new_holdout_order) != 1595:
        raise ValueError(
            "expected 512 prior checkpoint-selection and 1,595 unused validation images"
        )

    output.mkdir(parents=True, exist_ok=False)
    import psutil
    import tensorflow as tf

    process = psutil.Process()
    rss_start = process.memory_info().rss
    encode, jax, source_commit = _load_video_prism(checkpoint, source_root)
    extraction_started = time.perf_counter()
    train_ids, train_hashes, train_features, train_times, train_pad = _extract_split(
        train_assets,
        "train",
        sample_dir,
        encode,
        tf,
        jax,
        args.batch_size,
    )
    val_ids, val_hashes, val_features, val_times, val_pad = _extract_split(
        validation_assets,
        "val",
        sample_dir,
        encode,
        tf,
        jax,
        args.batch_size,
    )
    cache_path = output / "features.npz"
    np.savez_compressed(
        cache_path,
        train_ids=np.asarray(train_ids),
        train_media_sha256=np.asarray(train_hashes),
        train_features=train_features,
        validation_ids=np.asarray(val_ids),
        validation_media_sha256=np.asarray(val_hashes),
        validation_features=val_features,
        selection_validation_ids=np.asarray(selection_order),
        new_holdout_ids=np.asarray(new_holdout_order),
    )
    report = {
        "schema_version": 1,
        "status": "complete_videoprism_clevr4_full_feature_cache",
        "implementation_commit": subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "extractor_sha256": sha256_file(Path(__file__).resolve()),
        "claim_scope": "synthetic CLEVR-4 taxonomy decisions; image-only Path A feature candidate",
        "dataset_revision": DATASET_REVISION,
        "source_archive_sha512_expected": (
            "769465d90b6550a242f6d5940b52f2e0944c65af09e17e6f3f2cb65701af66057"
            "ab06485b2194fa9d8932e03a03e87b3fadece43049e1af1ddd9f8e4990ed19a"
        ),
        "source_archive_full_hash_verified": False,
        "sample_hashes": {
            "train_labels": sha256_file(train_labels_path),
            "validation_labels": sha256_file(val_labels_path),
            "train_image_manifest": sha256_file(train_assets_path),
            "validation_image_manifest": sha256_file(val_assets_path),
            "selection_validation_manifest": sha256_file(selection_manifest_path),
            "fetch_report": sha256_file(sample_dir / "fetch-report.json"),
        },
        "train_image_count": len(train_ids),
        "train_unique_media_hash_count": len(set(train_hashes)),
        "validation_image_count": len(val_ids),
        "validation_unique_media_hash_count": len(set(val_hashes)),
        "checkpoint_selection_validation_count": len(selection_order),
        "new_validation_holdout_count": len(new_holdout_order),
        "train_validation_id_overlap": 0,
        "train_validation_media_hash_overlap": 0,
        "selection_holdout_id_overlap": 0,
        "train_ids_order_sha256": _ordered_ids_sha256(train_ids),
        "validation_ids_order_sha256": _ordered_ids_sha256(val_ids),
        "selection_validation_ids_order_sha256": _ordered_ids_sha256(selection_order),
        "new_holdout_ids_order_sha256": _ordered_ids_sha256(new_holdout_order),
        "encoder": {
            "model_id": "google/videoprism-base-f16r288",
            "model_revision": MODEL_REVISION,
            "source_commit": source_commit,
            "checkpoint_sha256": sha256_file(checkpoint),
            "checkpoint_bytes": checkpoint.stat().st_size,
            "parameters": 114365184,
            "frozen": True,
        },
        "preprocessing": {
            "image_resize": "resize-short-side-to-288-bilinear-then-center-crop",
            "frame_count": FRAME_COUNT,
            "static_image_policy": "repeat the same processed image across all 8 temporal slots",
            "pixel_range": "float32 RGB [0,1]",
            "feature_pooling": "mean across all output tokens to 768 dimensions",
        },
        "feature_cache_path": str(cache_path),
        "feature_cache_sha256": sha256_file(cache_path),
        "feature_cache_bytes": cache_path.stat().st_size,
        "batch_size": args.batch_size,
        "train_padded_forward_examples": train_pad,
        "validation_padded_forward_examples": val_pad,
        "train_batch_forward_seconds": train_times,
        "validation_batch_forward_seconds": val_times,
        "feature_extraction_seconds": time.perf_counter() - extraction_started,
        "jax_version": jax.__version__,
        "devices": [str(device) for device in jax.devices()],
        "rss_start_bytes": rss_start,
        "rss_end_bytes": process.memory_info().rss,
        "peak_working_set_bytes": getattr(process.memory_info(), "peak_wset", None),
        "sealed_audit_loaded": False,
    }
    report_path = output / "cache-report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["cache_report_sha256"] = sha256_file(report_path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
