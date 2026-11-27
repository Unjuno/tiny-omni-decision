"""Run a pinned CPU VideoPrism smoke on actual Physical Concepts train rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scripts.audit_physical_concepts_tfrecord import _feature_map, iter_tfrecord_examples

MODEL_REVISION = "be719a406d563b66f0ac969e7c94bab8e997c81a"
SOURCE_COMMIT = "c6eba0ece3dd49a24ffc44582966997a4a59f1d8"
CHECKPOINT_SHA256 = "61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29"
FRAME_COUNT = 8
FRAME_SIZE = 288
SOURCE_PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE_PACKAGE_ROOT))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def uniform_frame_indices(frame_count: int = 15, requested: int = FRAME_COUNT) -> list[int]:
    if frame_count < 1 or requested < 1 or requested > frame_count:
        raise ValueError("requested frame count must be in [1, frame_count]")
    if requested == 1:
        return [0]
    step = (frame_count - 1) / (requested - 1)
    result = [round(index * step) for index in range(requested)]
    if len(set(result)) != requested:
        raise ValueError("uniform frame sampling produced duplicate indices")
    return result


def preprocess_frames(frames: Any, tf: Any, np: Any) -> Any:
    if frames.shape != (FRAME_COUNT, 64, 64, 3):
        raise ValueError(f"expected [8,64,64,3] RGB frames, found {frames.shape}")
    scale = FRAME_SIZE / 64
    resized = tf.image.resize(frames, (round(64 * scale), round(64 * scale)), method="bilinear")
    cropped = tf.image.resize_with_crop_or_pad(resized, FRAME_SIZE, FRAME_SIZE)
    result = np.asarray(cropped, dtype=np.float32) / 255.0
    if result.shape != (FRAME_COUNT, FRAME_SIZE, FRAME_SIZE, 3) or not np.isfinite(result).all():
        raise ValueError("preprocessed frames have invalid shape or non-finite values")
    return result


def validate_source_audit(
    manifest: dict[str, Any], split: str, audit: dict[str, Any]
) -> dict[str, Any]:
    if split not in {"train", "validation"}:
        raise ValueError("only train and validation features may be extracted")
    source = manifest[split]
    if (
        audit.get("status") != "complete"
        or audit.get("object_name") != source.get("object_name")
        or audit.get("generation") != source.get("generation")
        or audit.get("source_sha256") != source.get("sha256")
        or audit.get("records") != source.get("records")
    ):
        raise ValueError(f"pinned {split} source audit does not match the split manifest")
    return source


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-videos", type=int)
    parser.add_argument("--minimum-available-memory-mib", type=int, default=768)
    return parser.parse_args()


def main() -> None:
    args = _args()
    if (args.max_videos is not None and args.max_videos < 1) or (
        args.minimum_available_memory_mib < 256
    ):
        raise ValueError(
            "max-videos must be positive when set; memory floor must be at least 256 MiB"
        )
    manifest_path = args.manifest.resolve(strict=True)
    audit_path = args.audit.resolve(strict=True)
    checkpoint = args.checkpoint.resolve(strict=True)
    source_root = args.source_root.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    os.environ["JAX_PLATFORM_NAME"] = "cpu"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    source = validate_source_audit(manifest, args.split, audit)
    if sha256_file(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("VideoPrism checkpoint validation failed")
    upstream = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if upstream != SOURCE_COMMIT:
        raise ValueError(f"unexpected VideoPrism upstream revision: {upstream}")

    import jax
    import jax.numpy as jnp
    import numpy as np
    import psutil
    import tensorflow as tf
    from videoprism import models as vp

    if jax.default_backend() != "cpu" or any(device.platform != "cpu" for device in jax.devices()):
        raise RuntimeError(f"CPU-only smoke found non-CPU JAX devices: {jax.devices()}")
    process = psutil.Process()
    memory_floor = args.minimum_available_memory_mib * 1024 * 1024
    memory_before_model = psutil.virtual_memory().available
    if memory_before_model < memory_floor:
        raise MemoryError(
            f"available RAM below configured floor before model load: {memory_before_model}"
        )
    model_started = time.perf_counter()
    model = vp.get_model("videoprism_public_v1_base")
    variables = vp.load_pretrained_weights(
        "videoprism_public_v1_base", checkpoint_path=str(checkpoint)
    )
    encode = jax.jit(lambda video: model.apply(variables, video, train=False))
    model_load_seconds = time.perf_counter() - model_started
    memory_after_model = psutil.virtual_memory().available
    if memory_after_model < memory_floor:
        raise MemoryError(
            f"available RAM below configured floor after model load: {memory_after_model}"
        )

    shard_path = Path(source["local_payload_path"]).resolve(strict=True)
    expected_rows = audit["record_sha256"]
    row_ids: list[str] = []
    count = (
        min(args.max_videos, len(expected_rows))
        if args.max_videos is not None
        else len(expected_rows)
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    feature_array = np.empty((count, FRAME_COUNT, 768), dtype=np.float32)
    features_written = 0
    forward_times: list[float] = []
    frame_indices = uniform_frame_indices()
    from tiny_omni_decision.temporal_tokens import pool_spatial_tokens_per_frame

    for index, serialized in enumerate(iter_tfrecord_examples(shard_path)):
        if index >= count:
            break
        if (
            index >= len(expected_rows)
            or hashlib.sha256(serialized).hexdigest() != expected_rows[index]
        ):
            raise ValueError(f"{args.split} row hash mismatch at index {index}")
        row_id = expected_rows[index]
        row = _feature_map(serialized)
        image_kind, images = row["image"]
        if image_kind != "bytes" or len(images) != 1 or len(images[0]) != 15 * 64 * 64 * 3:
            raise ValueError(f"invalid RGB sequence for row {row_id}")
        all_frames = np.frombuffer(images[0], dtype=np.uint8).reshape(15, 64, 64, 3)
        selected = all_frames[frame_indices]
        video = jnp.asarray(preprocess_frames(selected, tf, np)[None, ...])
        if psutil.virtual_memory().available < memory_floor:
            raise MemoryError("available RAM fell below configured floor before forward pass")
        started = time.perf_counter()
        result = encode(video)
        jax.tree_util.tree_map(
            lambda value: value.block_until_ready()
            if hasattr(value, "block_until_ready")
            else value,
            result,
        )
        forward_times.append(time.perf_counter() - started)
        tokens = result[0] if isinstance(result, tuple) else result
        per_frame = pool_spatial_tokens_per_frame(np.asarray(tokens), FRAME_COUNT).squeeze(0)
        if per_frame.shape != (FRAME_COUNT, 768) or not np.isfinite(per_frame).all():
            raise ValueError(f"invalid per-frame feature shape/value: {per_frame.shape}")
        feature_array[index] = per_frame.astype(np.float32)
        row_ids.append(row_id)
        features_written += 1
    if features_written != count:
        raise ValueError(f"expected {count} {args.split} rows, extracted {features_written}")
    feature_path = output_dir / "features.npz"
    np.savez_compressed(
        feature_path,
        features=feature_array,
        sample_ids=np.asarray(row_ids),
        group_ids=np.asarray(row_ids),
        frame_indices=np.asarray(frame_indices, dtype=np.int32),
    )
    mean_forward = float(np.mean(forward_times))
    report = {
        "status": "complete_actual_data_videoprism_feature_smoke",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "source_manifest_sha256": sha256_file(manifest_path),
        "train_audit_sha256": sha256_file(audit_path),
        "source_split": args.split,
        "source_object_name": source["object_name"],
        "source_rows_processed": features_written,
        "source_row_sha256": row_ids,
        "sampled_frame_indices_zero_based": frame_indices,
        "model_id": "google/videoprism-base-f16r288",
        "model_revision": MODEL_REVISION,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "upstream_source_commit": upstream,
        "feature_layout": "mean-pooled spatial tokens per frame; temporal order retained",
        "feature_shape": list(feature_array.shape),
        "feature_sha256": hashlib.sha256(np.ascontiguousarray(feature_array).tobytes()).hexdigest(),
        "feature_file": feature_path.name,
        "feature_file_sha256": sha256_file(feature_path),
        "jax_version": jax.__version__,
        "tensorflow_version": tf.__version__,
        "numpy_version": np.__version__,
        "backend": jax.default_backend(),
        "model_load_seconds": model_load_seconds,
        "per_video_forward_seconds": forward_times,
        "mean_forward_seconds_per_video": mean_forward,
        "projected_seconds_for_selected_shard": mean_forward * int(source["records"]),
        "peak_working_set_bytes": getattr(process.memory_info(), "peak_wset", None),
        "available_ram_before_model_bytes": memory_before_model,
        "available_ram_after_model_bytes": memory_after_model,
        "available_ram_after_forward_bytes": psutil.virtual_memory().available,
        "minimum_available_memory_mib": args.minimum_available_memory_mib,
        "max_videos_requested": args.max_videos,
        "quality_claim": "none; runtime/feature extraction smoke only",
        "probe_data_opened": False,
    }
    report_path = output_dir / "smoke-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
