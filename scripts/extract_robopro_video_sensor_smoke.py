"""Extract causal VideoPrism/state windows for the eight-episode RoboPro smoke.

The pinned VideoPrism encoder runs on CPU. This script reads only the fixed
train/development sample from one attempt directory and writes a separate
feature cache; it does not train, score, or inspect an official test/audit set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import av
import numpy as np
import psutil

from tiny_omni_decision.robopro_actions import (
    causal_window_indices,
    numeric_option_separation,
)

DATASET_REVISION = "90ec789bf4018eb9c0f75da9f69aab5c185f0fd0"
VIDEO_MODEL_ID = "google/videoprism-base-f16r288"
VIDEO_MODEL_REVISION = "be719a406d563b66f0ac969e7c94bab8e997c81a"
VIDEO_SOURCE_COMMIT = "c6eba0ece3dd49a24ffc44582966997a4a59f1d8"
VIDEO_CHECKPOINT_SHA256 = "61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29"
VIDEO_SAMPLE_IDS = {
    "train": [7801, 7802, 7805, 7807],
    "development": [5001, 5005, 5006, 5008],
}
QUERY_PROGRESS = (0.25, 0.5, 0.75)
FPS = 50
FRAME_COUNT = 8
FRAME_SIZE = 288
HISTORY_SECONDS = 2.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _decode_rgb_and_pts(path: Path) -> tuple[np.ndarray, list[float], float]:
    with av.open(str(path)) as container:
        streams = [stream for stream in container.streams if stream.type == "video"]
        if not streams:
            raise ValueError(f"no video stream: {path}")
        stream = streams[0]
        rate = float(stream.average_rate)
        frames = []
        pts = []
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError(f"video frame has no PTS: {path}")
            frames.append(frame.to_ndarray(format="rgb24"))
            pts.append(float(frame.pts * frame.time_base))
    if rate != FPS:
        raise ValueError(f"expected {FPS} fps, got {rate}: {path}")
    return np.stack(frames), pts, rate


def _load_episode(
    split: str,
    episode_id: int,
    prompt: str,
    expected_length: int,
    task_id_to_text: dict[int, str],
    data_root: Path,
    receipt: dict[tuple[str, int, str], dict[str, Any]],
    parquet_rows: dict[int, list[dict[str, Any]]],
) -> dict[str, Any]:
    parquet_path = data_root / split / "data" / f"episode_{episode_id:06d}.parquet"
    video_path = data_root / split / "video" / f"episode_{episode_id:06d}_cam_high.mp4"
    rows = parquet_rows[episode_id]
    if len(rows) != expected_length:
        raise ValueError(f"row count differs from episode metadata: {episode_id}")
    expected_frame_ids = list(range(expected_length))
    if [row["frame_index"] for row in rows] != expected_frame_ids:
        raise ValueError(f"frame index gap or reorder: {episode_id}")
    if {row["episode_index"] for row in rows} != {episode_id}:
        raise ValueError(f"episode index changes within Parquet: {episode_id}")
    states = np.asarray([row["observation.state"] for row in rows], dtype=np.float32)
    actions = np.asarray([row["action"] for row in rows], dtype=np.float32)
    timestamps = np.asarray([row["timestamp"] for row in rows], dtype=np.float64)
    if states.shape != (expected_length, 14) or actions.shape != (expected_length, 14):
        raise ValueError(f"state/action shape is not [length,14]: {episode_id}")
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError(f"non-finite state/action value: {episode_id}")
    deltas = np.diff(timestamps)
    if not len(deltas) or np.max(np.abs(deltas - 1 / FPS)) > 1e-4:
        raise ValueError(f"Parquet timestamps are not 50-Hz: {episode_id}")
    task_ids = {int(row["task_index"]) for row in rows}
    if len(task_ids) != 1 or task_id_to_text[next(iter(task_ids))] != prompt:
        raise ValueError(f"task-index text mismatch: {episode_id}")

    for kind, path in (("parquet", parquet_path), ("mp4", video_path)):
        file_receipt = receipt[(split, episode_id, kind)]
        if path.stat().st_size != file_receipt["bytes"]:
            raise ValueError(f"file size differs from frozen download receipt: {path}")
        if sha256_file(path) != file_receipt["sha256"]:
            raise ValueError(f"hash differs from frozen download receipt: {path}")

    rgb_frames, video_pts, rate = _decode_rgb_and_pts(video_path)
    if len(rgb_frames) != expected_length:
        raise ValueError(f"decoded frame count differs from Parquet: {episode_id}")
    max_pts_delta = float(np.max(np.abs(timestamps - np.asarray(video_pts))))
    if max_pts_delta > 0.001:
        raise ValueError(f"video PTS differs from Parquet timestamps: {episode_id}")
    return {
        "split": split,
        "episode_id": episode_id,
        "prompt": prompt,
        "states": states,
        "actions": actions,
        "timestamps": timestamps,
        "rgb_frames": rgb_frames,
        "video_pts": video_pts,
        "video_rate": rate,
        "max_pts_delta": max_pts_delta,
        "parquet_sha256": receipt[(split, episode_id, "parquet")]["sha256"],
        "video_sha256": receipt[(split, episode_id, "mp4")]["sha256"],
    }


def run(
    metadata_dir: Path,
    split_manifest_path: Path,
    data_root: Path,
    checkpoint: Path,
    source_root: Path,
    output_root: Path,
    parquet_rows_path: Path,
) -> dict[str, Any]:
    run_started = time.perf_counter()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_root}")
    if sha256_file(checkpoint) != VIDEO_CHECKPOINT_SHA256:
        raise ValueError("VideoPrism checkpoint SHA-256 mismatch")
    source_commit = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if source_commit != VIDEO_SOURCE_COMMIT:
        raise ValueError(f"unexpected VideoPrism source commit: {source_commit}")

    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    if split_manifest["revision"] != DATASET_REVISION:
        raise ValueError("RoboPro source revision mismatch")
    for name, expected in split_manifest["source_files"].items():
        if sha256_file(metadata_dir / name) != expected["sha256"]:
            raise ValueError(f"source metadata hash mismatch: {name}")
    task_rows = read_jsonl(metadata_dir / "tasks.jsonl")
    episode_rows = read_jsonl(metadata_dir / "episodes.jsonl")
    task_id_to_text = {int(row["task_index"]): row["task"] for row in task_rows}
    episode_by_id = {int(row["episode_index"]): row for row in episode_rows}
    receipt_rows = json.loads((data_root / "download-receipt.json").read_text(encoding="utf-8"))[
        "files"
    ]
    receipt = {(row["split"], int(row["episode_id"]), row["kind"]): row for row in receipt_rows}
    parquet_rows = {
        int(episode_id): rows
        for episode_id, rows in json.loads(parquet_rows_path.read_text(encoding="utf-8")).items()
    }

    episodes_by_split: dict[str, list[dict[str, Any]]] = {}
    for split, sample_ids in VIDEO_SAMPLE_IDS.items():
        if not set(sample_ids) <= set(split_manifest[split]["episode_ids"]):
            raise ValueError(f"selected sample does not belong to the frozen {split} split")
        prompts = {episode_by_id[episode_id]["tasks"][0] for episode_id in sample_ids}
        if len(prompts) != 1:
            raise ValueError(f"sample episodes do not share one {split} prompt")
        prompt = next(iter(prompts))
        episodes_by_split[split] = [
            _load_episode(
                split,
                episode_id,
                prompt,
                int(episode_by_id[episode_id]["length"]),
                task_id_to_text,
                data_root,
                receipt,
                parquet_rows,
            )
            for episode_id in sample_ids
        ]

    sys.path.insert(0, str(source_root))
    import jax
    import jax.numpy as jnp
    import tensorflow as tf
    from videoprism import models as vp

    if any(device.platform != "cpu" for device in jax.devices()):
        raise RuntimeError(f"expected CPU-only JAX devices, got {jax.devices()}")
    from scripts.extract_frozen_videoprism_features import (
        FRAME_COUNT as VP_FRAME_COUNT,
    )
    from scripts.extract_frozen_videoprism_features import (
        FRAME_SIZE as VP_FRAME_SIZE,
    )
    from scripts.extract_frozen_videoprism_features import (
        _preprocess,
    )
    from scripts.extract_videoprism_framewise_features import pool_framewise_tokens

    if VP_FRAME_COUNT != FRAME_COUNT or VP_FRAME_SIZE != FRAME_SIZE:
        raise ValueError("local VideoPrism preprocessing constants changed")
    model = vp.get_model("videoprism_public_v1_base")
    load_start = time.perf_counter()
    variables = vp.load_pretrained_weights(
        "videoprism_public_v1_base", checkpoint_path=str(checkpoint)
    )
    for leaf in jax.tree_util.tree_leaves(variables):
        if not np.isfinite(np.asarray(leaf)).all():
            raise ValueError("non-finite value in loaded pretrained weights")
    load_seconds = time.perf_counter() - load_start
    encode = jax.jit(lambda video: model.apply(variables, video, train=False))

    process = psutil.Process()
    rss_start = process.memory_info().rss
    feature_rows: list[dict[str, Any]] = []
    per_window_seconds: list[float] = []
    frame_feature_arrays = []
    pooled_feature_arrays = []
    sensor_windows = []
    targets = []
    query_indices = []
    sampled_indices = []
    action_options = []
    option_reports = {}
    for split in ("train", "development"):
        episodes = episodes_by_split[split]
        split_options = []
        split_option_reports = []
        for progress in QUERY_PROGRESS:
            actions_at_progress = []
            indices_at_progress = []
            for episode in episodes:
                query_index = round((len(episode["actions"]) - 1) * progress)
                actions_at_progress.append(episode["actions"][query_index].tolist())
                indices_at_progress.append(query_index)
            separation = numeric_option_separation(actions_at_progress)
            if separation["within_tolerance_pairs"]:
                raise ValueError(
                    f"duplicate action options in {split} at progress {progress}: {separation}"
                )
            split_options.append(actions_at_progress)
            split_option_reports.append(
                {
                    "relative_progress": progress,
                    "query_indices": indices_at_progress,
                    "option_source_episode_ids": [episode["episode_id"] for episode in episodes],
                    "separation": separation,
                }
            )
        action_options.append(split_options)
        option_reports[split] = split_option_reports

        for episode in episodes:
            episode_frame_features = []
            episode_pool_features = []
            episode_sensor_windows = []
            episode_targets = []
            episode_queries = []
            episode_sampled = []
            for progress in QUERY_PROGRESS:
                query_index = round((len(episode["actions"]) - 1) * progress)
                frame_ids = causal_window_indices(
                    len(episode["rgb_frames"]),
                    query_index,
                    history_seconds=HISTORY_SECONDS,
                    fps=FPS,
                    requested=FRAME_COUNT,
                )
                if any(frame_id > query_index for frame_id in frame_ids):
                    raise ValueError("causal frame window contains a future observation")
                frames = episode["rgb_frames"][frame_ids]
                processed = _preprocess(frames, tf)
                if processed.shape != (FRAME_COUNT, FRAME_SIZE, FRAME_SIZE, 3):
                    raise ValueError(f"unexpected preprocessed video shape: {processed.shape}")
                started = time.perf_counter()
                result = encode(jnp.asarray(processed[None, ...]))
                jax.tree_util.tree_map(
                    lambda value: (
                        value.block_until_ready() if hasattr(value, "block_until_ready") else value
                    ),
                    result,
                )
                per_window_seconds.append(time.perf_counter() - started)
                tokens = result[0] if isinstance(result, tuple) else result
                frame_features = pool_framewise_tokens(np.asarray(tokens))
                if (
                    frame_features.shape != (FRAME_COUNT, 768)
                    or not np.isfinite(frame_features).all()
                ):
                    raise ValueError("invalid VideoPrism temporal features")
                episode_frame_features.append(frame_features)
                episode_pool_features.append(frame_features.mean(axis=0))
                episode_sensor_windows.append(episode["states"][frame_ids])
                episode_targets.append(episode["actions"][query_index])
                episode_queries.append(query_index)
                episode_sampled.append(frame_ids)
                feature_rows.append(
                    {
                        "split": split,
                        "episode_id": episode["episode_id"],
                        "prompt": episode["prompt"],
                        "relative_progress": progress,
                        "query_index": query_index,
                        "frame_indices": frame_ids,
                        "history_seconds": float(
                            episode["timestamps"][query_index] - episode["timestamps"][frame_ids[0]]
                        ),
                        "max_video_pts_timestamp_delta_seconds": episode["max_pts_delta"],
                        "video_sha256": episode["video_sha256"],
                        "parquet_sha256": episode["parquet_sha256"],
                    }
                )
            frame_feature_arrays.append(episode_frame_features)
            pooled_feature_arrays.append(episode_pool_features)
            sensor_windows.append(episode_sensor_windows)
            targets.append(episode_targets)
            query_indices.append(episode_queries)
            sampled_indices.append(episode_sampled)

    # Keep temporal and pooled layouts separate and explicit in the NPZ archive.
    temporal_features = np.asarray(frame_feature_arrays, dtype=np.float32)
    pooled_features = np.asarray(pooled_feature_arrays, dtype=np.float32)
    sensor_array = np.asarray(sensor_windows, dtype=np.float32)
    target_array = np.asarray(targets, dtype=np.float32)
    options_array = np.asarray(action_options, dtype=np.float32)
    if temporal_features.shape != (8, 3, FRAME_COUNT, 768):
        raise ValueError(f"invalid feature cache shape: {temporal_features.shape}")
    if pooled_features.shape != (8, 3, 768) or sensor_array.shape != (8, 3, FRAME_COUNT, 14):
        raise ValueError("pooled video or sensor cache shape mismatch")
    if target_array.shape != (8, 3, 14) or options_array.shape != (2, 3, 4, 14):
        raise ValueError("action target or option cache shape mismatch")

    output_root.mkdir(parents=True, exist_ok=False)
    cache_path = output_root / "videoprism-video-sensor-features.npz"
    np.savez_compressed(
        cache_path,
        split=np.asarray(["train"] * 4 + ["development"] * 4),
        episode_ids=np.asarray(VIDEO_SAMPLE_IDS["train"] + VIDEO_SAMPLE_IDS["development"]),
        relative_progress=np.asarray(QUERY_PROGRESS, dtype=np.float32),
        query_indices=np.asarray(query_indices, dtype=np.int32),
        sampled_frame_indices=np.asarray(sampled_indices, dtype=np.int32),
        temporal_video_features=temporal_features,
        pooled_video_features=pooled_features,
        aligned_state_windows=sensor_array,
        target_action_vectors=target_array,
        action_options=options_array,
        candidate_target_indices=np.broadcast_to(np.arange(4, dtype=np.int32), (2, 3, 4)).copy(),
    )
    process_info = process.memory_info()
    report = {
        "schema_version": "robopro-videoprism-sensor-feature-smoke-v1",
        "status": "feature_smoke_pass",
        "dataset": "Hoshipu/RoboPro",
        "dataset_revision": DATASET_REVISION,
        "source_train_development_split_manifest_sha256": sha256_file(split_manifest_path),
        "model_id": VIDEO_MODEL_ID,
        "model_revision": VIDEO_MODEL_REVISION,
        "model_source_commit": source_commit,
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "encoder_frozen": True,
        "device": [str(device) for device in jax.devices()],
        "jax_version": jax.__version__,
        "flax_version": __import__("flax").__version__,
        "tensorflow_version": tf.__version__,
        "pyav_version": av.__version__,
        "parameters": 114365184,
        "input_frames": FRAME_COUNT,
        "input_resolution": [FRAME_SIZE, FRAME_SIZE],
        "history_window_seconds_max": HISTORY_SECONDS,
        "frame_preprocessing": (
            "uniform causal PTS frames; resize-short-side 288, center crop, RGB float32 [0,1]"
        ),
        "video_feature_layout": {
            "temporal": list(temporal_features.shape),
            "pooled": list(pooled_features.shape),
            "temporal_pooling": (
                "mean spatial token positions for 8 temporally contextualized outputs"
            ),
        },
        "sensor_feature_layout": list(sensor_array.shape),
        "action_target_layout": list(target_array.shape),
        "candidate_option_layout": list(options_array.shape),
        "causal_windows_verified": all(
            max(row["frame_indices"]) <= row["query_index"] for row in feature_rows
        ),
        "candidate_action_uniqueness": option_reports,
        "sample_receipt": str(data_root / "download-receipt.json"),
        "parquet_rows_sidecar": {
            "path": parquet_rows_path.name,
            "bytes": parquet_rows_path.stat().st_size,
            "sha256": sha256_file(parquet_rows_path),
        },
        "sample_episode_windows": feature_rows,
        "weights_load_seconds": load_seconds,
        "video_window_forward_seconds": per_window_seconds,
        "feature_extraction_wall_seconds": sum(per_window_seconds),
        "total_wall_seconds": time.perf_counter() - run_started,
        "rss_start_bytes": rss_start,
        "rss_end_bytes": process_info.rss,
        "peak_working_set_bytes": getattr(process_info, "peak_wset", None),
        "feature_archive": {
            "path": cache_path.name,
            "bytes": cache_path.stat().st_size,
            "sha256": sha256_file(cache_path),
        },
        "model_training_or_quality_evaluation": False,
        "official_test_or_sealed_audit_accessed": False,
        "gpu_used": False,
        "limitations": [
            "Eight episodes and two exact prompts are a runtime smoke, not an evaluation sample.",
            "Action options distinguish recorded commands only; alternate commands may be valid.",
            "Task success, scene identity, and scene-held-out generalization are unverified.",
            "The state stream is raw 14-D proprioception, not a pretrained sensor representation.",
        ],
    }
    report_path = output_root / "videoprism-feature-report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    report_sha256 = sha256_file(report_path)
    print(
        json.dumps(
            {
                "status": report["status"],
                "device": report["device"],
                "feature_archive": report["feature_archive"],
                "causal_windows_verified": report["causal_windows_verified"],
                "wall_seconds": report["feature_extraction_wall_seconds"],
                "total_wall_seconds": report["total_wall_seconds"],
                "report_sha256": report_sha256,
                "report": str(report_path),
            },
            indent=2,
        )
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--parquet-rows-json", type=Path, required=True)
    args = parser.parse_args()
    run(
        args.metadata_dir.resolve(strict=True),
        args.split_manifest.resolve(strict=True),
        args.data_root.resolve(strict=True),
        args.checkpoint.resolve(strict=True),
        args.source_root.resolve(strict=True),
        args.output_root.resolve(),
        args.parquet_rows_json.resolve(strict=True),
    )


if __name__ == "__main__":
    main()
