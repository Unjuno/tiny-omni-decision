"""Extract pinned frozen VideoPrism and causal tactile features from DextraVerse."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from scripts.extract_frozen_videoprism_features import (
    CHECKPOINT_SHA256,
)
from scripts.extract_frozen_videoprism_features import (
    SOURCE_COMMIT as VIDEOPRISM_SOURCE_COMMIT,
)
from scripts.pretrained_reuse.dextraverse_contact import LABELS
from scripts.pretrained_reuse.dextraverse_features import (
    assemble_sensor_window,
    read_video_features,
    write_video_batch,
)
from scripts.pretrained_reuse.inspect_dextraverse_contact_archive import (
    EXPECTED_ARCHIVE_BYTES,
    EXPECTED_ARCHIVE_SHA256,
)

DATASET_REVISION = "257731928e25cd068b02d80d18d8f3cb77075a14"
VIDEO_MODEL_REVISION = "be719a406d563b66f0ac969e7c94bab8e997c81a"
VIDEO_PARAMETERS = 114_365_184
EXPECTED_ZARR_VERSION = "2.18.7"
EPISODE_FRAME_CACHE_MAX_BYTES = 1024**3


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
    parser.add_argument("--videoprism-python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("gpu", "cpu"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
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
    videoprism_python: Path,
    output: Path,
    *,
    device: str = "cpu",
    batch_size: int = 8,
) -> dict[str, Any]:
    """Verify inputs and stream one immutable feature cache outside the repo."""
    output = output.resolve()
    repository_root = Path(__file__).resolve().parents[2]
    if output.is_relative_to(repository_root):
        raise ValueError("feature cache output must stay outside the repository")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing feature output: {output}")
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    archive = archive.resolve(strict=True)
    checkpoint = checkpoint.resolve(strict=True)
    source = videoprism_source.resolve(strict=True)
    worker_python = videoprism_python.resolve(strict=True)
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

    import zarr
    from zarr.storage import ZipStore

    if zarr.__version__ != EXPECTED_ZARR_VERSION:
        raise ValueError(
            f"ZIP reader version must remain {EXPECTED_ZARR_VERSION}, got {zarr.__version__}"
        )

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
    query_times_s = np.asarray([float(row["query_time_s"]) for row in rows], dtype=np.float32)
    episode_durations_s = np.asarray([
        float(episode_rows[int(row["episode_id"])]["duration_s"]) for row in rows
    ], dtype=np.float32)
    groups: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[int(row["episode_id"])].append(index)

    started = time.perf_counter()
    encoder_pipe_roundtrip_seconds = 0.0
    episode_rgb_cache_bytes_total = 0
    episode_rgb_cache_bytes_peak = 0
    rgb_window_materialized_bytes = 0
    direct_frame_reads = 0
    worker = None
    worker_stderr_path = output / "videoprism-worker.stderr.log"
    worker_report_path = output / "videoprism-worker-report.json"
    repo_root = Path(__file__).resolve().parents[2]
    worker_environment = os.environ.copy()
    worker_environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    worker_environment["PYTHONPATH"] = os.pathsep.join(filter(None, (
        str(repo_root), worker_environment.get("PYTHONPATH", ""),
    )))
    worker_command = [
        str(worker_python),
        str(Path(__file__).with_name("dextraverse_videoprism_worker.py")),
        "--checkpoint", str(checkpoint),
        "--source-root", str(source),
        "--device", device,
        "--worker-report", str(worker_report_path),
    ]
    try:
        with worker_stderr_path.open("wb") as worker_stderr:
            worker = subprocess.Popen(
                worker_command,
                cwd=repo_root,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=worker_stderr,
                bufsize=0,
                env=worker_environment,
            )
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
                episode_frame_count = episode_end - episode_start
                expected_episode_rgb_bytes = episode_frame_count * 240 * 320 * 3
                episode_video = None
                if expected_episode_rgb_bytes <= EPISODE_FRAME_CACHE_MAX_BYTES:
                    episode_video = np.asarray(camera[episode_start:episode_end])
                    if (episode_video.shape != (episode_frame_count, 240, 320, 3)
                            or episode_video.dtype != np.uint8):
                        raise ValueError(f"invalid episode RGB cache for episode {episode_id}")
                    episode_rgb_cache_bytes_total += episode_video.nbytes
                    episode_rgb_cache_bytes_peak = max(
                        episode_rgb_cache_bytes_peak, episode_video.nbytes
                    )
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
                    if episode_video is None:
                        frames = np.stack([np.asarray(camera[index]) for index in frame_ids])
                        direct_frame_reads += len(frame_ids)
                    else:
                        local_frame_ids = np.asarray(
                            [index - episode_start for index in frame_ids], dtype=np.int64
                        )
                        frames = episode_video[local_frame_ids]
                    rgb_window_materialized_bytes += frames.nbytes
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
                    batch_frames.append(frames)
                    batch_positions.append(position)
                    if len(batch_frames) == batch_size:
                        features_batch = np.stack(batch_frames).astype(np.uint8, copy=False)
                        forward_started = time.perf_counter()
                        write_video_batch(worker.stdin, features_batch)
                        pooled = read_video_features(worker.stdout, len(batch_positions))
                        encoder_pipe_roundtrip_seconds += time.perf_counter() - forward_started
                        features[batch_positions] = pooled
                        features.flush()
                        batch_frames.clear()
                        batch_positions.clear()
                if batch_frames:
                    features_batch = np.stack(batch_frames).astype(np.uint8, copy=False)
                    forward_started = time.perf_counter()
                    write_video_batch(worker.stdin, features_batch)
                    pooled = read_video_features(worker.stdout, len(batch_positions))
                    encoder_pipe_roundtrip_seconds += time.perf_counter() - forward_started
                    features[batch_positions] = pooled
                    features.flush()
        worker.stdin.write(struct.pack("<I", 0))
        worker.stdin.flush()
        worker.stdin.close()
        worker.stdout.close()
        return_code = worker.wait()
        if return_code != 0:
            log_text = worker_stderr_path.read_text(encoding="utf-8", errors="replace")[-4000:]
            raise RuntimeError(f"VideoPrism worker exited {return_code}: {log_text}")
        worker_report = json.loads(worker_report_path.read_text(encoding="utf-8"))
        if (worker_report.get("status") != "ready"
                or worker_report.get("checkpoint_sha256") != CHECKPOINT_SHA256
                or worker_report.get("source_commit") != VIDEOPRISM_SOURCE_COMMIT
                or worker_report.get("platform") != device):
            raise ValueError("VideoPrism worker runtime identity/device failed its pinned gate")
    finally:
        if worker is not None and worker.poll() is None:
            worker.terminate()
            worker.wait(timeout=30)
        if worker is not None:
            for pipe in (worker.stdin, worker.stdout):
                if pipe is not None and not pipe.closed:
                    pipe.close()
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
        outer_folds=folds, targets=targets, query_times_s=query_times_s,
        episode_durations_s=episode_durations_s,
    )
    artifact_hashes = {
        name: sha256_file(output / name)
        for name in (
            "video_features.npy", "video_timestamps_ms.npy", "video_valid.npy",
            "sensor_values.npy", "sensor_timestamps_ms.npy", "sensor_valid.npy",
            "examples.npz",
        )
    }
    artifact_hashes[worker_report_path.name] = sha256_file(worker_report_path)
    artifact_hashes[worker_stderr_path.name] = sha256_file(worker_stderr_path)
    code_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    metadata = {
        "status": "complete_frozen_dextraverse_contact_features",
        "dataset_revision": DATASET_REVISION,
        "archive_sha256": EXPECTED_ARCHIVE_SHA256,
        "preflight_sha256": sha256_file(report_path),
        "video_model_revision": VIDEO_MODEL_REVISION,
        "videoprism_source_commit": source_commit,
        "execution_source_commit": code_revision,
        "extractor_sha256": sha256_file(Path(__file__).resolve()),
        "worker_sha256": sha256_file(
            Path(__file__).with_name("dextraverse_videoprism_worker.py")
        ),
        "preprocessor_sha256": sha256_file(
            repo_root / "scripts" / "extract_frozen_videoprism_features.py"
        ),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "video_encoder_parameters": VIDEO_PARAMETERS,
        "video_feature_shape": [len(rows), 8, 768],
        "video_timestamps_shape": [len(rows), 8],
        "sensor_value_shape": [len(rows), max_sensor, 30],
        "sensor_valid_lengths": [len(row["sensor_frame_indices"]) for row in rows],
        "sample_ids_sha256": hashlib.sha256("\n".join(sample_ids.tolist()).encode()).hexdigest(),
        "device": worker_report["device"],
        "zarr_version": zarr.__version__,
        "numpy_version": np.__version__,
        "batch_size": batch_size,
        "encoder_pipe_roundtrip_seconds": encoder_pipe_roundtrip_seconds,
        "episode_rgb_cache_max_bytes": EPISODE_FRAME_CACHE_MAX_BYTES,
        "episode_rgb_cache_total_bytes": episode_rgb_cache_bytes_total,
        "episode_rgb_cache_peak_bytes": episode_rgb_cache_bytes_peak,
        "rgb_window_materialized_bytes": rgb_window_materialized_bytes,
        "direct_frame_reads": direct_frame_reads,
        "videoprism_python": str(worker_python),
        "videoprism_worker_report": worker_report,
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
        args.videoprism_python, args.output, device=args.device,
        batch_size=args.batch_size,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
