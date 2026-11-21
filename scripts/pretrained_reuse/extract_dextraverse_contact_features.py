"""Extract pinned frozen VideoPrism and causal tactile features from DextraVerse."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from scripts.extract_frozen_videoprism_features import (
    CHECKPOINT_SHA256,
    _preprocess,
)
from scripts.extract_frozen_videoprism_features import (
    SOURCE_COMMIT as VIDEOPRISM_SOURCE_COMMIT,
)
from scripts.pretrained_reuse.dextraverse_contact import LABELS
from scripts.pretrained_reuse.dextraverse_features import (
    assemble_sensor_window,
    pool_videoprism_tokens,
)
from scripts.pretrained_reuse.inspect_dextraverse_contact_archive import (
    EXPECTED_ARCHIVE_BYTES,
    EXPECTED_ARCHIVE_SHA256,
)

DATASET_REVISION = "257731928e25cd068b02d80d18d8f3cb77075a14"
VIDEO_MODEL_REVISION = "be719a406d563b66f0ac969e7c94bab8e997c81a"
VIDEO_PARAMETERS = 114_365_184


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--videoprism-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--batch-size", type=int, default=2)
    return parser.parse_args()


def _load_preflight(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if (report.get("status") != "PREFLIGHT_PASS"
            or report.get("archive_bytes") != EXPECTED_ARCHIVE_BYTES
            or report.get("archive_sha256") != EXPECTED_ARCHIVE_SHA256
            or report.get("query_records_count") != len(report.get("query_records", []))
            or not report.get("query_records")):
        raise ValueError("real-data preflight did not pass the frozen source/support gates")
    return report


def _validate_episode_rows(report: dict[str, Any]) -> tuple[list[dict], dict[int, dict]]:
    rows = report["query_records"]
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("preflight contains duplicate sample IDs")
    episodes = {int(row["episode_id"]): row for row in report["episode_rows"]}
    for row in rows:
        episode = episodes.get(int(row["episode_id"]))
        if episode is None or int(row["outer_fold"]) not in range(5):
            raise ValueError("preflight query references unknown episode/fold")
        start, end = int(episode["start_index"]), int(episode["end_index_exclusive"])
        indices = [*row["video_frame_indices"], *row["sensor_frame_indices"]]
        if (not indices or any(not start <= int(index) < end for index in indices)
                or int(row["query_index"]) not in row["sensor_frame_indices"]
                or row["target_id"] not in LABELS):
            raise ValueError(f"invalid causal index or target in {row['sample_id']}")
    return rows, episodes


def extract_features(
    archive: Path,
    preflight_path: Path,
    checkpoint: Path,
    videoprism_source: Path,
    output: Path,
    *,
    device: str = "gpu",
    batch_size: int = 2,
) -> dict[str, Any]:
    """Verify inputs and stream one immutable feature cache outside the repo."""
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing feature output: {output}")
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    archive = archive.resolve(strict=True)
    checkpoint = checkpoint.resolve(strict=True)
    source = videoprism_source.resolve(strict=True)
    report_path = preflight_path.resolve(strict=True)
    if archive.stat().st_size != EXPECTED_ARCHIVE_BYTES:
        raise ValueError("archive has not reached the pinned complete byte length")
    if sha256_file(archive) != EXPECTED_ARCHIVE_SHA256:
        raise ValueError("archive SHA-256 differs from the pinned publisher checksum")
    if sha256_file(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("VideoPrism checkpoint SHA-256 differs from the pinned checkpoint")
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if source_commit != VIDEOPRISM_SOURCE_COMMIT:
        raise ValueError("VideoPrism source tree is not the pinned upstream commit")
    report = _load_preflight(report_path)
    rows, episode_rows = _validate_episode_rows(report)

    import jax
    import jax.numpy as jnp
    import tensorflow as tf
    import zarr
    from zarr.storage import ZipStore

    if device == "gpu":
        devices = jax.devices("gpu")
        if not devices:
            raise RuntimeError("GPU was requested but JAX reports no local GPU")
        chosen_device = devices[0]
    else:
        devices = jax.devices("cpu")
        chosen_device = devices[0]
    try:
        tf.config.set_visible_devices([], "GPU")
    except RuntimeError as exc:
        raise RuntimeError("TensorFlow initialized before CPU-only preprocessing setup") from exc

    sys.path.insert(0, str(source))
    from videoprism import models as vp

    model = vp.get_model("videoprism_public_v1_base")
    variables = vp.load_pretrained_weights(
        "videoprism_public_v1_base", checkpoint_path=str(checkpoint)
    )
    encode = jax.jit(lambda video: model.apply(variables, video, train=False))

    max_sensor = max(len(row["sensor_frame_indices"]) for row in rows)
    output.mkdir(parents=True, exist_ok=False)
    features = np.lib.format.open_memmap(
        output / "video_features.npy", mode="w+", dtype=np.float32,
        shape=(len(rows), 8, 768),
    )
    features[:] = np.nan
    video_times_out = np.lib.format.open_memmap(
        output / "video_timestamps_ms.npy", mode="w+", dtype=np.float32,
        shape=(len(rows), 8),
    )
    video_valid_out = np.lib.format.open_memmap(
        output / "video_valid.npy", mode="w+", dtype=np.bool_,
        shape=(len(rows), 8),
    )
    sensor_values_out = np.lib.format.open_memmap(
        output / "sensor_values.npy", mode="w+", dtype=np.float32,
        shape=(len(rows), max_sensor, 30),
    )
    sensor_times_out = np.lib.format.open_memmap(
        output / "sensor_timestamps_ms.npy", mode="w+", dtype=np.float32,
        shape=(len(rows), max_sensor),
    )
    sensor_valid_out = np.lib.format.open_memmap(
        output / "sensor_valid.npy", mode="w+", dtype=np.bool_,
        shape=(len(rows), max_sensor),
    )
    targets = np.asarray([LABELS.index(row["target_id"]) for row in rows], dtype=np.int64)
    episodes = np.asarray([int(row["episode_id"]) for row in rows], dtype=np.int64)
    folds = np.asarray([int(row["outer_fold"]) for row in rows], dtype=np.int64)
    sample_ids = np.asarray([row["sample_id"] for row in rows], dtype=np.str_)
    groups: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[int(row["episode_id"])].append(index)

    started = time.perf_counter()
    forward_seconds = 0.0
    try:
        with ZipStore(str(archive), mode="r") as store:
            root = zarr.open_group(store=store, mode="r")
            timestamps = root["data/timestamps"]
            left_force = root["data/left_tactile_data_fingerforce"]
            right_force = root["data/right_tactile_data_fingerforce"]
            camera = root["data/camera_ego_rgb"]
            if (tuple(left_force.shape[1:]) != (5, 3)
                    or tuple(right_force.shape) != tuple(left_force.shape)
                    or tuple(camera.shape[1:]) != (240, 320, 3)):
                raise ValueError("pinned media/sensor array schema changed after preflight")
            for episode_id, positions in groups.items():
                positions.sort()
                episode = episode_rows[episode_id]
                episode_start = int(episode["start_index"])
                episode_end = int(episode["end_index_exclusive"])
                local_times = np.asarray(timestamps[episode_start:episode_end], dtype=np.float64)
                local_left = np.asarray(
                    left_force[episode_start:episode_end], dtype=np.float32
                )
                local_right = np.asarray(
                    right_force[episode_start:episode_end], dtype=np.float32
                )
                batch_frames: list[np.ndarray] = []
                batch_positions: list[int] = []
                for position in positions:
                    row = rows[position]
                    frame_ids = [int(index) for index in row["video_frame_indices"]]
                    frames = np.stack([np.asarray(camera[index]) for index in frame_ids])
                    if frames.shape != (8, 240, 320, 3) or frames.dtype != np.uint8:
                        raise ValueError(f"invalid RGB frames for {row['sample_id']}")
                    query_index = int(row["query_index"])
                    query_time = float(timestamps[query_index])
                    video_time_values = np.asarray(
                        timestamps[frame_ids], dtype=np.float64
                    )
                    video_relative_ms = (
                        (video_time_values - query_time) * 1000
                    ).astype(np.float32)
                    if (np.any(np.diff(video_relative_ms) <= 0)
                            or np.any(video_time_values > query_time - 0.25 + 1e-9)):
                        raise ValueError(
                            f"video features violate causal guard for {row['sample_id']}"
                        )
                    video_times_out[position] = video_relative_ms
                    video_valid_out[position] = True
                    sensor_ids = [int(index) for index in row["sensor_frame_indices"]]
                    sensor_values, relative_ms = assemble_sensor_window(
                        local_left,
                        local_right,
                        local_times,
                        [index - episode_start for index in sensor_ids],
                        query_index - episode_start,
                    )
                    length = len(sensor_ids)
                    sensor_values_out[position] = 0
                    sensor_times_out[position] = 0
                    sensor_valid_out[position] = False
                    sensor_values_out[position, :length] = sensor_values
                    sensor_times_out[position, :length] = relative_ms
                    sensor_valid_out[position, :length] = True
                    batch_frames.append(_preprocess(frames, tf))
                    batch_positions.append(position)
                    if len(batch_frames) == batch_size:
                        features_batch = np.stack(batch_frames).astype(np.float32, copy=False)
                        forward_started = time.perf_counter()
                        with jax.default_device(chosen_device):
                            values = encode(jnp.asarray(features_batch))
                            values = jax.tree_util.tree_map(
                                lambda value: value.block_until_ready()
                                if hasattr(value, "block_until_ready") else value,
                                values,
                            )
                        forward_seconds += time.perf_counter() - forward_started
                        tokens = values[0] if isinstance(values, tuple) else values
                        pooled = np.stack([
                            pool_videoprism_tokens(np.asarray(tokens[i:i + 1]))
                            for i in range(len(batch_positions))
                        ])
                        features[batch_positions] = pooled
                        features.flush()
                        batch_frames.clear()
                        batch_positions.clear()
                if batch_frames:
                    features_batch = np.stack(batch_frames).astype(np.float32, copy=False)
                    forward_started = time.perf_counter()
                    with jax.default_device(chosen_device):
                        values = encode(jnp.asarray(features_batch))
                        values = jax.tree_util.tree_map(
                            lambda value: value.block_until_ready()
                            if hasattr(value, "block_until_ready") else value,
                            values,
                        )
                    forward_seconds += time.perf_counter() - forward_started
                    tokens = values[0] if isinstance(values, tuple) else values
                    features[batch_positions] = np.stack([
                        pool_videoprism_tokens(np.asarray(tokens[i:i + 1]))
                        for i in range(len(batch_positions))
                    ])
                    features.flush()
    finally:
        for array in (
            features, video_times_out, video_valid_out, sensor_values_out,
            sensor_times_out, sensor_valid_out,
        ):
            array.flush()

    if (not np.isfinite(features).all() or not np.isfinite(video_times_out).all()
            or not video_valid_out.all() or not sensor_valid_out.any(axis=1).all()):
        raise ValueError("feature extraction ended with incomplete/non-finite rows")
    np.savez_compressed(
        output / "examples.npz", sample_ids=sample_ids, episode_ids=episodes,
        outer_folds=folds, targets=targets,
    )
    artifact_hashes = {
        name: sha256_file(output / name)
        for name in (
            "video_features.npy", "video_timestamps_ms.npy", "video_valid.npy",
            "sensor_values.npy", "sensor_timestamps_ms.npy", "sensor_valid.npy",
            "examples.npz",
        )
    }
    metadata = {
        "status": "complete_frozen_dextraverse_contact_features",
        "dataset_revision": DATASET_REVISION,
        "archive_sha256": EXPECTED_ARCHIVE_SHA256,
        "preflight_sha256": sha256_file(report_path),
        "video_model_revision": VIDEO_MODEL_REVISION,
        "videoprism_source_commit": source_commit,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "video_encoder_parameters": VIDEO_PARAMETERS,
        "video_feature_shape": [len(rows), 8, 768],
        "video_timestamps_shape": [len(rows), 8],
        "sensor_value_shape": [len(rows), max_sensor, 30],
        "sensor_valid_lengths": [len(row["sensor_frame_indices"]) for row in rows],
        "sample_ids_sha256": hashlib.sha256("\n".join(sample_ids.tolist()).encode()).hexdigest(),
        "device": str(chosen_device),
        "batch_size": batch_size,
        "video_forward_seconds": forward_seconds,
        "artifact_sha256": artifact_hashes,
        "elapsed_seconds": time.perf_counter() - started,
        "sealed_audit_accessed": False,
        "model_fit_performed": False,
    }
    (output / "feature-report.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> None:
    args = _args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite feature output: {args.output}")
    result = extract_features(
        args.archive, args.preflight, args.checkpoint, args.videoprism_source,
        args.output, device=args.device, batch_size=args.batch_size,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
