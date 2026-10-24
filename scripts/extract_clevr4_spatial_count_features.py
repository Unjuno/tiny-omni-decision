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

FROZEN_TRAIN_MANIFEST_SHA256 = "feec0f9de345ab432d7e486fd0d9a8d58e7c92db6a07be1a2df4183ce96c474a"
FROZEN_VALIDATION_MANIFEST_SHA256 = (
    "100237bc05991db7cc4d7410f613d602e94ab3fe9a313752f30619ff74cf2a4b"
)
FROZEN_TRAIN_ID_ORDER_SHA256 = "23e504526b0d9bbf1f8a901c1c22245aa765460dc7425b69fe3ace354be5018c"
FROZEN_VALIDATION_ID_ORDER_SHA256 = (
    "bc0434db01109ee1df5e135ea0989ac42d225cd756b638ad4c26a8b46323104c"
)

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
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-train-images", type=int)
    parser.add_argument("--resume-validation-images", type=int)
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


def _find_completed_prefix(features: np.ndarray, *, chunk_rows: int = 128) -> int:
    """Return the contiguous prefix of finite, nonzero feature rows.

    The extractor initializes unwritten memory-map rows to zero and assigns a
    full feature batch only after encoding succeeds. Chunking keeps validation
    bounded in memory for the multi-gigabyte full-corpus array.
    """
    if features.ndim < 2:
        raise ValueError("feature array must have a batch and feature dimension")
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")

    completed = 0
    found_gap = False
    for start in range(0, len(features), chunk_rows):
        block = np.asarray(features[start : start + chunk_rows])
        if not np.isfinite(block).all():
            raise ValueError("feature array contains NaN or Inf")
        nonzero_rows = np.any(block.reshape(len(block), -1) != 0, axis=1)
        for has_features in nonzero_rows:
            if bool(has_features):
                if found_gap:
                    raise ValueError("feature array has non-contiguous written rows")
                completed += 1
            else:
                found_gap = True
    return completed


def _validate_feature_prefix(
    features: np.ndarray, completed: int, *, chunk_rows: int = 128
) -> None:
    if completed < 0 or completed > len(features):
        raise ValueError("resume checkpoint is outside the feature array")
    for start in range(0, completed, chunk_rows):
        block = np.asarray(features[start : min(start + chunk_rows, completed)])
        if not np.isfinite(block).all():
            raise ValueError("resume prefix contains NaN or Inf")
        nonzero_rows = np.any(block.reshape(len(block), -1) != 0, axis=1)
        if not nonzero_rows.all():
            raise ValueError("resume prefix contains an unwritten feature row")


def _open_feature_store(
    path: Path,
    *,
    expected_shape: tuple[int, int, int],
    resume_start: int | None,
) -> tuple[np.ndarray, int]:
    if resume_start is None:
        return (
            np.lib.format.open_memmap(
                path,
                mode="w+",
                dtype=np.float16,
                shape=expected_shape,
            ),
            0,
        )

    if path.is_file():
        features = np.load(path, mmap_mode="r+")
        if features.shape != expected_shape or features.dtype != np.float16:
            raise ValueError(f"resume feature array has unexpected shape or dtype: {path}")
        _validate_feature_prefix(features, resume_start)
        return features, resume_start

    if resume_start != 0:
        raise ValueError(f"resume progress exists but feature array is missing: {path}")
    return (
        np.lib.format.open_memmap(
            path,
            mode="w+",
            dtype=np.float16,
            shape=expected_shape,
        ),
        0,
    )


def _write_json_atomically(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _validate_resume_metadata(metadata: dict[str, object], *, split_dir: Path) -> None:
    expected_metadata = {
        "input_split_dir": str(split_dir),
        "checkpoint_sha256_expected": CHECKPOINT_SHA256,
        "model_revision": MODEL_REVISION,
        "frame_count": FRAME_COUNT,
        "output_shape_per_image": [64, 768],
        "output_dtype": "float16",
        "backend_policy": "CPU only; does not acquire GPU memory",
        "full_corpus": True,
        "smoke_image_limit_per_split": None,
    }
    if metadata.get("status") not in {"starting", "resuming"}:
        raise ValueError("resume metadata must describe an incomplete full-corpus extraction")
    if any(metadata.get(key) != value for key, value in expected_metadata.items()):
        raise ValueError("resume metadata does not match the requested frozen extraction")


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
    if args.resume and args.max_images_per_split is not None:
        raise ValueError("full-corpus resume cannot be combined with a smoke image limit")
    if not args.resume and (
        args.resume_train_images is not None or args.resume_validation_images is not None
    ):
        raise ValueError("resume image counts require --resume")

    metadata_path = output / "feature-metadata.json"
    if args.resume:
        if not output.is_dir() or not metadata_path.is_file():
            raise ValueError("--resume requires an existing extraction output with metadata")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        _validate_resume_metadata(metadata, split_dir=split_dir)
    else:
        output.mkdir(parents=True, exist_ok=False)
        metadata = {
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
        _write_json_atomically(metadata_path, metadata)

    train_manifest = split_dir / "train-image-manifest.jsonl"
    validation_manifest = split_dir / "validation-image-manifest.jsonl"
    all_manifests = {
        "train": read_jsonl(train_manifest),
        "validation": read_jsonl(validation_manifest),
    }
    if not all_manifests["train"] or not all_manifests["validation"]:
        raise ValueError("nonempty train and validation manifests are required")
    manifests = {split: list(rows) for split, rows in all_manifests.items()}
    for split, rows in manifests.items():
        if any(row.get("split") != split for row in rows):
            raise ValueError(f"manifest rows have the wrong split label: {split}")
    manifest_hashes = {
        "train": _manifest_hash(train_manifest),
        "validation": _manifest_hash(validation_manifest),
    }
    id_order_hashes = {
        split: hashlib.sha256("\n".join(row["id"] for row in rows).encode()).hexdigest()
        for split, rows in all_manifests.items()
    }
    if args.max_images_per_split is None:
        if manifest_hashes["train"] != FROZEN_TRAIN_MANIFEST_SHA256:
            raise ValueError("train manifest hash differs from the frozen experiment corpus")
        if manifest_hashes["validation"] != FROZEN_VALIDATION_MANIFEST_SHA256:
            raise ValueError("validation manifest hash differs from the frozen experiment corpus")
        if id_order_hashes["train"] != FROZEN_TRAIN_ID_ORDER_SHA256:
            raise ValueError("train ID order differs from the frozen experiment corpus")
        if id_order_hashes["validation"] != FROZEN_VALIDATION_ID_ORDER_SHA256:
            raise ValueError("validation ID order differs from the frozen experiment corpus")
    if args.max_images_per_split is not None:
        for split, rows in manifests.items():
            manifests[split] = rows[: args.max_images_per_split]
        if args.max_images_per_split is not None:
            manifests[split] = rows[: args.max_images_per_split]
    train_ids = {row["id"] for row in manifests["train"]}
    validation_ids = {row["id"] for row in manifests["validation"]}
    train_hashes = {row["sha256"] for row in manifests["train"]}
    validation_hashes = {row["sha256"] for row in manifests["validation"]}
    if train_ids & validation_ids or train_hashes & validation_hashes:
        raise ValueError("train/validation identity overlap")

    progress_path = output / "resume-progress.json"
    explicit_resume_counts = {
        "train": args.resume_train_images,
        "validation": args.resume_validation_images,
    }
    has_durable_progress = progress_path.is_file()
    if args.resume and has_durable_progress:
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if set(progress) != {"train", "validation"}:
            raise ValueError("resume progress file has unexpected split keys")
        for split, explicit_count in explicit_resume_counts.items():
            if explicit_count is not None and explicit_count != progress[split]:
                raise ValueError(f"explicit {split} resume count disagrees with durable progress")
    elif args.resume:
        progress = {}
        for split in ("train", "validation"):
            feature_path = output / f"{split}-spatial-features.npy"
            count = explicit_resume_counts[split]
            if feature_path.exists() and count is None:
                raise ValueError(f"legacy partial {split} data requires an explicit resume count")
            progress[split] = 0 if count is None else count
            if feature_path.exists():
                existing = np.load(feature_path, mmap_mode="r")
                expected_shape = (len(manifests[split]), 64, 768)
                if existing.shape != expected_shape or existing.dtype != np.float16:
                    raise ValueError(f"legacy partial {split} array has unexpected shape or dtype")
                discovered = _find_completed_prefix(existing)
                if discovered < progress[split]:
                    raise ValueError(
                        f"legacy partial {split} array ends before the requested safe checkpoint"
                    )
        _write_json_atomically(progress_path, progress)
    else:
        progress = {"train": 0, "validation": 0}
        _write_json_atomically(progress_path, progress)

    for split, count in progress.items():
        if not isinstance(count, int) or not 0 <= count <= len(manifests[split]):
            raise ValueError(f"invalid durable {split} resume count: {count!r}")
    metadata.update(
        {
            "status": "resuming" if args.resume else "starting",
            "batch_size": args.batch_size,
            "train_manifest_sha256": manifest_hashes["train"],
            "validation_manifest_sha256": manifest_hashes["validation"],
            "train_id_order_sha256": id_order_hashes["train"],
            "validation_id_order_sha256": id_order_hashes["validation"],
            "resume_progress_path": str(progress_path),
        }
    )
    if args.resume:
        metadata["resume_started_utc"] = datetime.now(UTC).isoformat()
        metadata["resume_counts"] = dict(progress)
    _write_json_atomically(metadata_path, metadata)

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
            features, start_index = _open_feature_store(
                features_path,
                expected_shape=(len(rows), 64, 768),
                resume_start=progress[split] if args.resume else None,
            )
            split_start = time.perf_counter()
            for start in range(start_index, len(rows), args.batch_size):
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
                    features.flush()
                    progress[split] = completed
                    _write_json_atomically(progress_path, progress)
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
            progress[split] = len(rows)
            _write_json_atomically(progress_path, progress)
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
