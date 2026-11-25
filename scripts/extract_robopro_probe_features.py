"""Extract causal frozen VideoPrism windows for a fixed RoboPro probe sample."""

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

from tiny_omni_decision.robopro_actions import causal_window_indices, numeric_option_separation

DATASET_REVISION = "90ec789bf4018eb9c0f75da9f69aab5c185f0fd0"
VIDEO_MODEL_ID = "google/videoprism-base-f16r288"
VIDEO_MODEL_REVISION = "be719a406d563b66f0ac969e7c94bab8e997c81a"
VIDEO_SOURCE_COMMIT = "c6eba0ece3dd49a24ffc44582966997a4a59f1d8"
VIDEO_CHECKPOINT_SHA256 = "61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29"
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


def _decode_selected(path: Path, indices: set[int], expected_count: int):
    selected = {}
    selected_pts = {}
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate)
        for index, frame in enumerate(container.decode(stream)):
            if index in indices:
                if frame.pts is None:
                    raise ValueError(f"selected video frame has no PTS: {path}")
                selected[index] = frame.to_ndarray(format="rgb24")
                selected_pts[index] = float(frame.pts * frame.time_base)
            if index >= max(indices):
                break
    if set(selected) != indices:
        raise ValueError(f"video decoding missed requested indices: {path}")
    return selected, selected_pts, fps, expected_count


def extract(
    sample_path: Path,
    audit_path: Path,
    data_root: Path,
    arrays_path: Path,
    checkpoint: Path,
    source_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite feature output: {output_root}")
    sample_bytes = sample_path.read_bytes()
    sample = json.loads(sample_bytes)
    audit_bytes = audit_path.read_bytes()
    audit_report = json.loads(audit_bytes)
    if audit_report["status"] != "audit_complete":
        raise ValueError("sample data audit did not complete")
    if audit_report["sample_manifest_sha256"] != hashlib.sha256(sample_bytes).hexdigest():
        raise ValueError("data audit belongs to another sample manifest")
    if sample["revision"] != DATASET_REVISION:
        raise ValueError("unexpected RoboPro source revision")
    if sha256_file(checkpoint) != VIDEO_CHECKPOINT_SHA256:
        raise ValueError("VideoPrism checkpoint hash mismatch")
    source_commit = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if source_commit != VIDEO_SOURCE_COMMIT:
        raise ValueError("unexpected VideoPrism source commit")
    if sha256_file(arrays_path) == "":
        raise ValueError("could not hash selected parquet array sidecar")
    with np.load(arrays_path, allow_pickle=False) as data:
        episode_ids = data["episode_ids"].astype(np.int32)
        splits = data["split"].astype(str)
        offsets = data["offsets"].astype(np.int64)
        states = data["states"].astype(np.float32)
        actions = data["actions"].astype(np.float32)
        timestamps = data["timestamps"].astype(np.float64)
    if len(episode_ids) != sample["train_episode_count"] + sample["development_episode_count"]:
        raise ValueError("array sidecar episode count differs from the frozen sample")
    if states.shape != actions.shape or states.shape[1] != 14:
        raise ValueError("invalid sensor/action array dimensions")
    if offsets.shape != (len(episode_ids) + 1,) or offsets[-1] != len(states):
        raise ValueError("invalid Parquet offsets in array sidecar")
    if len(timestamps) != len(states) or not np.isfinite(states).all():
        raise ValueError("invalid Parquet sensor array")

    sample_records = []
    options = []
    group_records = []
    expected_episode_id_order = []
    for split in ("train", "development"):
        for group_index, group in enumerate(sample[split]):
            records = group["episodes"]
            ids = [int(row["episode_id"]) for row in records]
            if len(ids) != 4:
                raise ValueError("each candidate group must have four episodes")
            indices = [int(np.flatnonzero(episode_ids == episode_id)[0]) for episode_id in ids]
            if any(splits[index] != split for index in indices):
                raise ValueError("Parquet sidecar partition does not match sample")
            group_actions = []
            query_rows = []
            for progress in QUERY_PROGRESS:
                progress_actions = []
                progress_queries = []
                for episode_index in indices:
                    start, end = offsets[episode_index : episode_index + 2]
                    count = int(end - start)
                    query = round((count - 1) * progress)
                    progress_actions.append(actions[start + query].tolist())
                    progress_queries.append(query)
                separation = numeric_option_separation(progress_actions)
                if separation["within_tolerance_pairs"]:
                    raise ValueError(
                        f"candidate actions collide for {split} group {group_index} "
                        f"at {progress}: {separation}"
                    )
                group_actions.append(progress_actions)
                query_rows.append(progress_queries)
            options.append(group_actions)
            group_records.append(
                {
                    "split": split,
                    "group_index": group_index,
                    "prompt": group["prompt"],
                    "prompt_sha256": group["prompt_sha256"],
                    "episode_ids": ids,
                    "target_indices": [0, 1, 2, 3],
                    "query_indices_by_progress": query_rows,
                }
            )
            for option_index, (record, episode_index) in enumerate(
                zip(records, indices, strict=True)
            ):
                expected_episode_id_order.append(int(record["episode_id"]))
                start, end = offsets[episode_index : episode_index + 2]
                sample_records.append(
                    {
                        "split": split,
                        "group_index": group_index,
                        "option_index": option_index,
                        "episode_id": int(record["episode_id"]),
                        "prompt": group["prompt"],
                        "prompt_sha256": group["prompt_sha256"],
                        "start": int(start),
                        "end": int(end),
                    }
                )
    if episode_ids.tolist() != expected_episode_id_order:
        raise ValueError("Parquet sidecar ordering differs from the frozen sample manifest")

    sys.path.insert(0, str(source_root))
    import jax
    import jax.numpy as jnp
    import tensorflow as tf
    from videoprism import models as vp

    if not jax.devices() or any(device.platform != "cpu" for device in jax.devices()):
        raise RuntimeError(f"expected CPU-only JAX devices, got {jax.devices()}")
    from scripts.extract_frozen_videoprism_features import _preprocess
    from scripts.extract_videoprism_framewise_features import pool_framewise_tokens

    model = vp.get_model("videoprism_public_v1_base")
    load_started = time.perf_counter()
    variables = vp.load_pretrained_weights(
        "videoprism_public_v1_base", checkpoint_path=str(checkpoint)
    )
    weight_load_seconds = time.perf_counter() - load_started
    encode = jax.jit(lambda video: model.apply(variables, video, train=False))

    receipt = json.loads((data_root / "download-receipt.json").read_text(encoding="utf-8"))
    receipt_by_key = {
        (row["split"], int(row["episode_id"]), row["kind"]): row for row in receipt["files"]
    }
    temporal = []
    pooled = []
    state_windows = []
    targets = []
    query_indices = []
    frame_indices = []
    clock_deltas = []
    forward_seconds = []
    for record in sample_records:
        episode_id = record["episode_id"]
        split = record["split"]
        start, end = record["start"], record["end"]
        count = end - start
        video_path = data_root / split / "video" / f"episode_{episode_id:06d}_cam_high.mp4"
        expected_hash = receipt_by_key[(split, episode_id, "mp4")]["sha256"]
        if sha256_file(video_path) != expected_hash:
            raise ValueError(f"video hash differs from receipt: {episode_id}")
        record_temporal = []
        record_pooled = []
        record_state = []
        record_target = []
        record_queries = []
        record_frame_ids = []
        for progress in QUERY_PROGRESS:
            query = round((count - 1) * progress)
            window = causal_window_indices(
                count,
                query,
                history_seconds=HISTORY_SECONDS,
                fps=FPS,
                requested=FRAME_COUNT,
            )
            raw_frames, pts, video_fps, _ = _decode_selected(video_path, set(window), count)
            if abs(video_fps - FPS) > 1e-6:
                raise ValueError(f"unexpected frame rate: {episode_id}")
            frame_batch = np.stack([raw_frames[index] for index in window])
            processed = _preprocess(frame_batch, tf)
            if processed.shape != (FRAME_COUNT, FRAME_SIZE, FRAME_SIZE, 3):
                raise ValueError(f"unexpected VideoPrism input shape: {processed.shape}")
            started_forward = time.perf_counter()
            result = encode(jnp.asarray(processed[None, ...]))
            jax.tree_util.tree_map(
                lambda value: (
                    value.block_until_ready() if hasattr(value, "block_until_ready") else value
                ),
                result,
            )
            forward_seconds.append(time.perf_counter() - started_forward)
            tokens = result[0] if isinstance(result, tuple) else result
            frame_features = pool_framewise_tokens(np.asarray(tokens))
            if frame_features.shape != (FRAME_COUNT, 768) or not np.isfinite(frame_features).all():
                raise ValueError(f"invalid VideoPrism feature: {episode_id}")
            selected_stamps = timestamps[start + np.asarray(window)]
            selected_pts = np.asarray([pts[index] for index in window])
            clock_deltas.append(float(np.max(np.abs(selected_stamps - selected_pts))))
            if clock_deltas[-1] > 0.001 or max(window) > query:
                raise ValueError("future frame or clock mismatch in causal feature window")
            record_temporal.append(frame_features)
            record_pooled.append(frame_features.mean(axis=0))
            record_state.append(states[start + np.asarray(window)])
            record_target.append(actions[start + query])
            record_queries.append(query)
            record_frame_ids.append(window)
        temporal.append(record_temporal)
        pooled.append(record_pooled)
        state_windows.append(record_state)
        targets.append(record_target)
        query_indices.append(record_queries)
        frame_indices.append(record_frame_ids)

    temporal_array = np.asarray(temporal, dtype=np.float32)
    pooled_array = np.asarray(pooled, dtype=np.float32)
    state_array = np.asarray(state_windows, dtype=np.float32)
    target_array = np.asarray(targets, dtype=np.float32)
    options_array = np.asarray(options, dtype=np.float32)
    if temporal_array.shape != (128, 3, 8, 768):
        raise ValueError(f"unexpected temporal feature shape: {temporal_array.shape}")
    if state_array.shape != (128, 3, 8, 14) or target_array.shape != (128, 3, 14):
        raise ValueError("sensor/action feature cache has an unexpected shape")
    if options_array.shape != (32, 3, 4, 14):
        raise ValueError(f"unexpected candidate action shape: {options_array.shape}")

    output_root.mkdir(parents=True, exist_ok=False)
    cache_path = output_root / "robopro-video-state-features.npz"
    np.savez_compressed(
        cache_path,
        episode_ids=np.asarray([record["episode_id"] for record in sample_records], dtype=np.int32),
        split=np.asarray([record["split"] for record in sample_records]),
        group_index=np.asarray(
            [record["group_index"] for record in sample_records], dtype=np.int32
        ),
        candidate_index=np.asarray(
            [record["option_index"] for record in sample_records], dtype=np.int32
        ),
        prompt=np.asarray([record["prompt"] for record in sample_records]),
        prompt_sha256=np.asarray([record["prompt_sha256"] for record in sample_records]),
        relative_progress=np.asarray(QUERY_PROGRESS, dtype=np.float32),
        query_indices=np.asarray(query_indices, dtype=np.int32),
        sampled_frame_indices=np.asarray(frame_indices, dtype=np.int32),
        temporal_video_features=temporal_array,
        pooled_video_features=pooled_array,
        aligned_state_windows=state_array,
        target_action_vectors=target_array,
        action_options=options_array,
    )
    report = {
        "schema_version": "robopro-action-probe-frozen-feature-cache-v1",
        "status": "feature_extraction_pass",
        "dataset_revision": DATASET_REVISION,
        "sample_manifest_sha256": hashlib.sha256(sample_bytes).hexdigest(),
        "data_audit_sha256": hashlib.sha256(audit_bytes).hexdigest(),
        "parquet_array_sidecar_sha256": sha256_file(arrays_path),
        "model_id": VIDEO_MODEL_ID,
        "model_revision": VIDEO_MODEL_REVISION,
        "model_source_commit": source_commit,
        "checkpoint_sha256": sha256_file(checkpoint),
        "encoder_frozen": True,
        "devices": [str(device) for device in jax.devices()],
        "jax_version": jax.__version__,
        "flax_version": __import__("flax").__version__,
        "tensorflow_version": tf.__version__,
        "pyav_version": av.__version__,
        "video_encoder_parameters": 114365184,
        "temporal_feature_shape": list(temporal_array.shape),
        "pooled_feature_shape": list(pooled_array.shape),
        "aligned_state_shape": list(state_array.shape),
        "target_action_shape": list(target_array.shape),
        "candidate_action_shape": list(options_array.shape),
        "window_policy": {
            "query_relative_progress": list(QUERY_PROGRESS),
            "frames": FRAME_COUNT,
            "maximum_history_seconds": HISTORY_SECONDS,
            "causal": True,
        },
        "maximum_query_window_clock_delta_seconds": max(clock_deltas),
        "weight_load_seconds": weight_load_seconds,
        "forward_seconds_total": sum(forward_seconds),
        "forward_seconds_per_window": forward_seconds,
        "total_wall_seconds": time.perf_counter() - started,
        "cache": {
            "path": cache_path.name,
            "bytes": cache_path.stat().st_size,
            "sha256": sha256_file(cache_path),
        },
        "training_or_quality_evaluation": False,
        "official_test_or_sealed_audit_accessed": False,
        "gpu_used": False,
        "limitations": [
            "The 128 episodes and 32 prompt groups are a bounded project sample.",
            "Prompt grouping does not establish physical scene independence.",
            "Recorded actions are not verified success or optimality labels.",
        ],
    }
    report_path = output_root / "feature-cache-report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["report_sha256"] = sha256_file(report_path)
    print(
        json.dumps(
            {
                "status": report["status"],
                "windows": len(forward_seconds),
                "total_wall_seconds": report["total_wall_seconds"],
                "forward_seconds_total": report["forward_seconds_total"],
                "cache": report["cache"],
                "report": str(report_path),
            },
            indent=2,
        )
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-manifest", type=Path, required=True)
    parser.add_argument("--data-audit", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--parquet-arrays", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    extract(
        args.sample_manifest.resolve(strict=True),
        args.data_audit.resolve(strict=True),
        args.data_root.resolve(strict=True),
        args.parquet_arrays.resolve(strict=True),
        args.checkpoint.resolve(strict=True),
        args.source_root.resolve(strict=True),
        args.output_root.resolve(),
    )


if __name__ == "__main__":
    main()
