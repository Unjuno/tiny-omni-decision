"""Run one pinned CPU VideoPrism forward for a requested IntPhys frame count."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tarfile
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from scripts.audit_intphys_temporal_sampling_coverage import uniform_frame_numbers
from scripts.audit_videoprism_checkpoint import CHECKPOINT_SHA256, MODEL_REVISION
from scripts.extract_frozen_videoprism_features import (
    FRAME_SIZE,
    _preprocess,
)
from scripts.extract_frozen_videoprism_features import (
    SOURCE_COMMIT as VIDEOPRISM_SOURCE_COMMIT,
)
from scripts.extract_intphys_dev_features import (
    EXPECTED_ARCHIVE_BYTES,
    EXPECTED_ARCHIVE_MD5,
    EXPECTED_ARCHIVE_SHA256,
    file_digests,
    parse_dev_member,
)
from tiny_omni_decision.temporal_tokens import pool_spatial_tokens_per_frame


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_sample_frames(
    archive_path: Path,
    frame_numbers: tuple[int, ...],
    *,
    sample_key: tuple[str, str, str] = ("O1", "01", "1"),
) -> list[bytes]:
    """Read only the selected scene-frame PNG bytes for one known dev sample."""
    selected: dict[int, bytes] = {}
    with tarfile.open(archive_path, mode="r|gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            block, scene, channel, frame_number = parse_dev_member(member.name)
            if (
                (block, scene, "1") != sample_key
                or channel != "scene"
                or frame_number not in frame_numbers
            ):
                continue
            payload = archive.extractfile(member)
            if payload is None:
                raise ValueError(f"cannot read pinned smoke frame: {member.name}")
            selected[frame_number] = payload.read()
    if set(selected) != set(frame_numbers):
        raise ValueError(
            f"sample {sample_key} provided {len(selected)}/{len(frame_numbers)} requested frames"
        )
    return [selected[number] for number in frame_numbers]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--frame-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite smoke receipt: {args.output}")
    if not 2 <= args.frame_count <= 100:
        raise ValueError("frame count must be between 2 and 100")
    archive = args.archive.resolve(strict=True)
    checkpoint = args.checkpoint.resolve(strict=True)
    source_root = args.source_root.resolve(strict=True)
    frame_numbers = uniform_frame_numbers(100, args.frame_count)
    started_at = datetime.now(UTC).isoformat()
    started = time.perf_counter()

    md5, archive_sha256, archive_bytes = file_digests(archive)
    if (archive_bytes, md5, archive_sha256) != (
        EXPECTED_ARCHIVE_BYTES,
        EXPECTED_ARCHIVE_MD5,
        EXPECTED_ARCHIVE_SHA256,
    ):
        raise ValueError("IntPhys dev archive differs from the pinned publisher bytes")
    checkpoint_sha256 = sha256_file(checkpoint)
    if checkpoint_sha256 != CHECKPOINT_SHA256:
        raise ValueError("VideoPrism checkpoint differs from pinned R1 weights")
    source_commit = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if source_commit != VIDEOPRISM_SOURCE_COMMIT:
        raise ValueError("VideoPrism source checkout differs from pinned R1 commit")

    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    sys.path.insert(0, str(source_root))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import jax
    import jax.numpy as jnp
    import psutil
    import tensorflow as tf
    from videoprism import models as vp

    devices = jax.devices()
    if not devices or any(device.platform != "cpu" for device in devices):
        raise RuntimeError(f"CPU-only smoke required; JAX devices are {devices}")
    process = psutil.Process()
    process_memory_start = process.memory_info()
    host_memory = psutil.virtual_memory()
    frame_bytes = load_sample_frames(archive, frame_numbers)
    frames = np.stack(
        [np.asarray(tf.io.decode_png(payload, channels=3)) for payload in frame_bytes]
    )
    normalized = _preprocess(frames, tf)

    model = vp.get_model("videoprism_public_v1_base")
    variables = vp.load_pretrained_weights(
        "videoprism_public_v1_base", checkpoint_path=str(checkpoint)
    )
    encode = jax.jit(lambda video: model.apply(variables, video, train=False))
    forward_started = time.perf_counter()
    result = encode(jnp.asarray(normalized[None, ...]))
    jax.tree_util.tree_map(
        lambda value: value.block_until_ready() if hasattr(value, "block_until_ready") else value,
        result,
    )
    forward_seconds = time.perf_counter() - forward_started
    tokens = result[0] if isinstance(result, tuple) else result
    token_array = np.asarray(tokens)
    pooled = pool_spatial_tokens_per_frame(token_array, args.frame_count)
    if (
        pooled.shape != (1, args.frame_count, 768)
        or not np.isfinite(pooled).all()
        or not np.isfinite(normalized).all()
    ):
        raise FloatingPointError(f"invalid 16-frame forward output: {pooled.shape}")
    process_memory_end = process.memory_info()
    report = {
        "status": "complete_single_video_frame_count_smoke",
        "started_at_utc": started_at,
        "ended_at_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "forward_seconds": forward_seconds,
        "sample_id": "O1/01/1",
        "label_loaded": False,
        "training_or_evaluation_performed": False,
        "hidden_test_accessed": False,
        "frame_count": args.frame_count,
        "selected_frame_numbers_1_based": list(frame_numbers),
        "input_shape": list(normalized[None, ...].shape),
        "upstream_token_shape": list(token_array.shape),
        "pooled_feature_shape": list(pooled.shape),
        "all_features_finite": bool(np.isfinite(pooled).all()),
        "input_frame_size": FRAME_SIZE,
        "model_revision": MODEL_REVISION,
        "videoprism_source_commit": source_commit,
        "checkpoint_sha256": checkpoint_sha256,
        "archive_bytes": archive_bytes,
        "archive_md5": md5,
        "archive_sha256": archive_sha256,
        "backend": "JAX CPU",
        "python_version": platform.python_version(),
        "jax_version": jax.__version__,
        "tensorflow_version": tf.__version__,
        "process_rss_at_start_bytes": process_memory_start.rss,
        "process_rss_at_end_bytes": process_memory_end.rss,
        "process_peak_working_set_bytes": getattr(
            process_memory_end, "peak_wset", process_memory_end.rss
        ),
        "host_ram_available_at_start_bytes": host_memory.available,
        "command": sys.argv,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
