"""Extract frozen VideoPrism features on a new CLEVR-4 train/dev split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from zipfile import ZipFile

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from scripts.extract_frozen_videoprism_clevr4_features import (  # noqa: E402
    CHECKPOINT_SHA256,
    FRAME_COUNT,
    MODEL_REVISION,
    _load_video_prism,
    _preprocess_rgb_image,
    sha256_file,
)
from scripts.prepare_clevr4_spatial_count import read_jsonl  # noqa: E402


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-images-per-split", type=int)
    return parser.parse_args()


def _pooled_spatial_tokens(tokens: np.ndarray) -> np.ndarray:
    if tokens.ndim != 3 or tokens.shape[1:] != (FRAME_COUNT * 16 * 16, 768):
        raise ValueError(f"unexpected VideoPrism token shape: {tokens.shape}")
    batch = tokens.reshape(len(tokens), FRAME_COUNT, 16, 16, 768).mean(axis=1)
    pooled = batch.reshape(len(batch), 8, 2, 8, 2, 768).mean(axis=(2, 4))
    result = pooled.reshape(len(pooled), 64, 768).astype(np.float16)
    if not np.isfinite(result).all():
        raise ValueError("pooled VideoPrism features contain NaN or Inf")
    return result


def _manifest_hash(path: Path) -> str:
    return sha256_file(path)


def main() -> None:
    args = _args()
    archive_path = args.archive.resolve(strict=True)
    split_dir = args.split_dir.resolve(strict=True)
    checkpoint = args.checkpoint.resolve(strict=True)
    source_root = args.source_root.resolve(strict=True)
    output = args.output_dir.resolve()
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    if args.max_images_per_split is not None and args.max_images_per_split < 1:
        raise ValueError("max images per split must be positive when set")
    output.mkdir(parents=True, exist_ok=False)
    metadata_path = output / "feature-metadata.json"
    metadata: dict[str, object] = {
        "status": "starting",
        "started_utc": datetime.now(UTC).isoformat(),
        "input_split_dir": str(split_dir),
        "checkpoint_sha256_expected": CHECKPOINT_SHA256,
        "model_revision": MODEL_REVISION,
        "frame_count": FRAME_COUNT,
        "output_shape_per_image": [64, 768],
        "output_dtype": "float16",
        "backend_policy": "CPU only; does not acquire GPU memory",
        "full_corpus": args.max_images_per_split is None,
        "smoke_image_limit_per_split": args.max_images_per_split,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    train_manifest = split_dir / "train-image-manifest.jsonl"
    validation_manifest = split_dir / "validation-image-manifest.jsonl"
    manifests = {
        "train": read_jsonl(train_manifest),
        "validation": read_jsonl(validation_manifest),
    }
    if not manifests["train"] or not manifests["validation"]:
        raise ValueError("nonempty train and validation manifests are required")
    for split, rows in manifests.items():
        if any(row.get("split") != split for row in rows):
            raise ValueError(f"manifest rows have the wrong split label: {split}")
        if args.max_images_per_split is not None:
            manifests[split] = rows[: args.max_images_per_split]
    train_ids = {row["id"] for row in manifests["train"]}
    validation_ids = {row["id"] for row in manifests["validation"]}
    train_hashes = {row["sha256"] for row in manifests["train"]}
    validation_hashes = {row["sha256"] for row in manifests["validation"]}
    if train_ids & validation_ids or train_hashes & validation_hashes:
        raise ValueError("train/validation identity overlap")

    # JAX and TensorFlow are imported only after the CPU-only environment is fixed.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2"
    os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
    os.environ["TF_NUM_INTEROP_THREADS"] = "2"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    import tensorflow as tf

    encode, jax, source_commit = _load_video_prism(checkpoint, source_root)
    backend = jax.default_backend()
    if backend != "cpu":
        raise RuntimeError(f"feature extraction must use CPU, got {backend}")
    extraction_start = time.perf_counter()
    with ZipFile(archive_path) as archive:
        for split, rows in manifests.items():
            features_path = output / f"{split}-spatial-features.npy"
            features = np.lib.format.open_memmap(
                features_path,
                mode="w+",
                dtype=np.float16,
                shape=(len(rows), 64, 768),
            )
            split_start = time.perf_counter()
            for start in range(0, len(rows), args.batch_size):
                batch_rows = rows[start : start + args.batch_size]
                videos = []
                for row in batch_rows:
                    payload = archive.read(row["member"])
                    if hashlib.sha256(payload).hexdigest() != row["sha256"]:
                        raise ValueError(f"source image hash mismatch: {row['id']}")
                    image = tf.io.decode_png(payload, channels=3)
                    videos.append(_preprocess_rgb_image(image, tf))
                video_batch = np.stack(videos)
                result = encode(jax.numpy.asarray(video_batch))
                jax.tree_util.tree_map(
                    lambda value: (
                        value.block_until_ready() if hasattr(value, "block_until_ready") else value
                    ),
                    result,
                )
                tokens = result[0] if isinstance(result, tuple) else result
                pooled = _pooled_spatial_tokens(np.asarray(tokens))
                features[start : start + len(batch_rows)] = pooled
                completed = start + len(batch_rows)
                if completed == len(rows) or completed % 1000 == 0:
                    elapsed = time.perf_counter() - split_start
                    print(
                        json.dumps(
                            {
                                "event": "feature_progress",
                                "split": split,
                                "images": completed,
                                "total": len(rows),
                                "elapsed_seconds": elapsed,
                                "images_per_second": completed / elapsed if elapsed else None,
                            }
                        ),
                        flush=True,
                    )
            features.flush()
            del features

    metadata.update(
        {
            "status": "complete" if args.max_images_per_split is None else "smoke_complete",
            "completed_utc": datetime.now(UTC).isoformat(),
            "elapsed_seconds": time.perf_counter() - extraction_start,
            "source_commit": source_commit,
            "checkpoint_sha256": sha256_file(checkpoint),
            "train_count": len(manifests["train"]),
            "validation_count": len(manifests["validation"]),
            "train_manifest_sha256": _manifest_hash(train_manifest),
            "validation_manifest_sha256": _manifest_hash(validation_manifest),
            "train_id_order_sha256": hashlib.sha256(
                "\n".join(row["id"] for row in manifests["train"]).encode()
            ).hexdigest(),
            "validation_id_order_sha256": hashlib.sha256(
                "\n".join(row["id"] for row in manifests["validation"]).encode()
            ).hexdigest(),
            "feature_hashes": {
                split: sha256_file(output / f"{split}-spatial-features.npy") for split in manifests
            },
            "feature_bytes": {
                split: (output / f"{split}-spatial-features.npy").stat().st_size
                for split in manifests
            },
            "evaluation_backend": backend,
            "environment": {
                "python": platform.python_version(),
                "jax": jax.__version__,
                "tensorflow": tf.__version__,
                "numpy": np.__version__,
            },
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
