"""Prepare and encode the frozen train/validation HABIT transition pilot."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

SOURCE_REVISION = "bae97d251907743f8fb6db7e70c61d361df4dc1e"
AUDIT_SHA256 = "9b7a9d6c0efbc0be055df55277ea107d9be0cd6a45dcbcb5e0571d0257796672"
VIDEO_REVISION = "be719a406d563b66f0ac969e7c94bab8e997c81a"
VIDEO_COMMIT = "c6eba0ece3dd49a24ffc44582966997a4a59f1d8"
CHECKPOINT_SHA256 = "61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29"
STATE_COLUMNS = (
    "robot0.observation.state.cartesian_position",
    "robot0.observation.state.joint_positions",
    "robot0.observation.state.gripper_position",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selected_records(audit: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    for episode in audit["episodes"]:
        split = episode["split"]
        if split not in {"train", "validation"}:
            raise ValueError("only frozen train/validation records are allowed")
        for ordinal, transition in enumerate(episode["transition_inputs"]):
            records.append({
                "id": f"{split}:{episode['episode_index']}:{ordinal}",
                "split": split,
                "episode_index": episode["episode_index"],
                "sid": episode["sid"],
                "task_index": episode["task_index"],
                "video_sha256": episode["video_sha256"],
                "target_subtask_id": transition["target_subtask_id"],
                "target_option_index": transition["target_option_index"],
                "option_count": episode["unique_option_count"],
                "context_end_ms": transition["context_end_ms"],
                "onset_ms": transition["onset_ms"],
                "video_frame_indices": transition["video_frame_indices"],
                "state_row_index": transition["state_row_index"],
                "state_age_ms": transition["state_age_ms"],
            })
    if not records or len({row["id"] for row in records}) != len(records):
        raise ValueError("transition IDs must be nonempty and unique")
    if any(not 0 <= row["target_option_index"] < row["option_count"] for row in records):
        raise ValueError("target option index is outside the recorded option count")
    train_groups = {row["sid"] for row in records if row["split"] == "train"}
    validation_groups = {row["sid"] for row in records if row["split"] == "validation"}
    train_media = {row["video_sha256"] for row in records if row["split"] == "train"}
    validation_media = {row["video_sha256"] for row in records if row["split"] == "validation"}
    if train_groups & validation_groups or train_media & validation_media:
        raise ValueError("task group or media identity overlaps train and validation")
    return records


def _prepare_inputs(audit_path: Path, payload_root: Path, output_dir: Path) -> Path:
    import av
    import pyarrow.parquet as pq

    audit_bytes = audit_path.read_bytes()
    if hashlib.sha256(audit_bytes).hexdigest() != AUDIT_SHA256:
        raise ValueError("audit receipt hash differs from the pinned payload audit")
    audit = json.loads(audit_bytes)
    if (audit.get("source_revision") != SOURCE_REVISION
            or audit.get("sealed_audit_payload_accessed") is not False
            or audit.get("sealed_audit_metrics_generated") is not False
            or audit.get("action_columns_read") is not False):
        raise ValueError("audit receipt is not the expected train/validation-only receipt")
    records = _selected_records(audit)
    episodes = {
        (row["split"], row["episode_index"]): row for row in audit["episodes"]
    }
    rows_by_episode: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        rows_by_episode[(record["split"], record["episode_index"])].append(record)

    for (split, episode_id), transitions in rows_by_episode.items():
        episode = episodes[(split, episode_id)]
        video = payload_root / split / "video" / f"episode_{episode_id:06d}.mp4"
        parquet = payload_root / split / "parquet" / f"episode_{episode_id:06d}.parquet"
        if (sha256_file(video) != episode["video_sha256"]
                or sha256_file(parquet) != episode["parquet_sha256"]):
            raise ValueError(f"payload SHA-256 mismatch for {split} episode {episode_id}")
        for row in transitions:
            row["video_path"] = str(video.resolve())

        table = pq.read_table(parquet, columns=["timestamp", *STATE_COLUMNS])
        timestamps_s = table.column("timestamp").to_pylist()
        states = []
        for index in range(len(timestamps_s)):
            state = sum((table.column(name)[index].as_py() for name in STATE_COLUMNS), [])
            if len(state) != 14 or not all(math.isfinite(value) for value in state):
                raise ValueError(f"invalid 14-D state at episode {episode_id}, row {index}")
            states.append(state)
        needed = {i for row in transitions for i in row["video_frame_indices"]}
        frame_times_ms: dict[int, float] = {}
        with av.open(str(video)) as container:
            stream = container.streams.video[0]
            for index, frame in enumerate(container.decode(stream)):
                if index in needed:
                    if frame.pts is None:
                        raise ValueError("selected video frame has no timestamp")
                    frame_times_ms[index] = float(frame.pts * stream.time_base) * 1000
                if index > max(needed):
                    break
        if set(frame_times_ms) != needed:
            raise ValueError(f"selected frame index missing for episode {episode_id}")
        for row in transitions:
            input_times = [frame_times_ms[index] for index in row["video_frame_indices"]]
            if max(input_times) > row["context_end_ms"] + 1e-6:
                raise ValueError(f"future video frame in {row['id']}")
            sensor_rows, sensor_times = [], []
            for frame_ms in input_times:
                state_index = bisect.bisect_right(timestamps_s, frame_ms / 1000) - 1
                if state_index < 0:
                    raise ValueError(f"no causal state row for {row['id']}")
                age_ms = frame_ms - timestamps_s[state_index] * 1000
                if age_ms < -1e-5 or age_ms > 110.0:
                    raise ValueError(f"sensor/frame alignment {age_ms:.4f} ms exceeds bound")
                sensor_rows.append(states[state_index])
                sensor_times.append(timestamps_s[state_index] * 1000)
            row["video_frame_times_ms"] = input_times
            row["sensor_frame_times_ms"] = sensor_times
            row["sensor_sequence"] = sensor_rows

    manifest = {
        "source_revision": SOURCE_REVISION,
        "audit_receipt_sha256": AUDIT_SHA256,
        "sealed_audit_accessed": False,
        "action_columns_read": False,
        "records": records,
    }
    path = output_dir / "transition-inputs.json"
    path.write_text(json.dumps(manifest, separators=(",", ":")), encoding="utf-8")
    return path


def _worker(input_path: Path, output_dir: Path, checkpoint: Path, source_root: Path) -> None:
    import av
    import jax
    import jax.numpy as jnp
    import numpy as np
    import psutil
    import tensorflow as tf

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.extract_frozen_videoprism_features import _preprocess

    if any(device.platform != "cpu" for device in jax.devices()):
        raise ValueError("this feature extraction is required to run on CPU only")
    if tf.config.list_physical_devices("GPU"):
        raise ValueError("TensorFlow found a GPU despite CPU-only execution policy")
    if sha256_file(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("VideoPrism checkpoint hash mismatch")
    source_commit = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if source_commit != VIDEO_COMMIT:
        raise ValueError("VideoPrism source commit mismatch")
    sys.path.insert(0, str(source_root))
    from videoprism import models as vp

    inputs = json.loads(input_path.read_text(encoding="utf-8"))
    if (inputs.get("source_revision") != SOURCE_REVISION
            or inputs.get("audit_receipt_sha256") != AUDIT_SHA256
            or inputs.get("sealed_audit_accessed") is not False):
        raise ValueError("worker received unexpected input manifest")
    records = inputs["records"]
    grouped: dict[tuple[str, int], list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for position, row in enumerate(records):
        grouped[(row["split"], row["episode_index"])].append((position, row))
    model = vp.get_model("videoprism_public_v1_base")
    variables = vp.load_pretrained_weights(
        "videoprism_public_v1_base", checkpoint_path=str(checkpoint)
    )
    encode = jax.jit(lambda video: model.apply(variables, video, train=False))
    features = np.empty((len(records), 8, 768), dtype=np.float32)
    started = time.perf_counter()
    forward_seconds = 0.0
    decode_seconds = 0.0
    process = psutil.Process()

    for (split, episode_id), indexed_rows in grouped.items():
        media = Path(indexed_rows[0][1]["video_path"]).resolve(strict=True)
        episode_hash = indexed_rows[0][1]["video_sha256"]
        if sha256_file(media) != episode_hash:
            raise ValueError(f"worker media hash mismatch for {split} episode {episode_id}")
        wanted = {i for _, row in indexed_rows for i in row["video_frame_indices"]}
        frames: dict[int, Any] = {}
        decode_start = time.perf_counter()
        with av.open(str(media)) as container:
            stream = container.streams.video[0]
            for index, frame in enumerate(container.decode(stream)):
                if index in wanted:
                    pts_ms = float(frame.pts * stream.time_base) * 1000
                    if pts_ms > min(
                        row["context_end_ms"] for _, row in indexed_rows
                        if index in row["video_frame_indices"]
                    ) + 1e-6:
                        raise ValueError("selected input includes a future frame")
                    frames[index] = frame.to_ndarray(format="rgb24")
                if index > max(wanted):
                    break
        decode_seconds += time.perf_counter() - decode_start
        if set(frames) != wanted:
            raise ValueError(f"incomplete frame decode for {episode_id}")
        for position, row in indexed_rows:
            selected = np.stack([frames[index] for index in row["video_frame_indices"]])
            video = jnp.asarray(_preprocess(selected, tf)[None, ...])
            forward_start = time.perf_counter()
            result = encode(video)
            result = jax.tree_util.tree_map(
                lambda value: value.block_until_ready()
                if hasattr(value, "block_until_ready") else value,
                result,
            )
            forward_seconds += time.perf_counter() - forward_start
            tokens = result[0] if isinstance(result, tuple) else result
            tokens = np.asarray(tokens)
            if tokens.shape != (1, 2048, 768) or not np.isfinite(tokens).all():
                raise ValueError(f"invalid VideoPrism output for {row['id']}: {tokens.shape}")
            features[position] = tokens.reshape(1, 8, 256, 768).mean(axis=2)[0]
        del frames

    np.savez_compressed(
        output_dir / "habit-transition-features.npz",
        ids=np.asarray([row["id"] for row in records]),
        splits=np.asarray([row["split"] for row in records]),
        video_features=features,
        sensor_sequence=np.asarray([row["sensor_sequence"] for row in records], dtype=np.float32),
        target_index=np.asarray([row["target_option_index"] for row in records], dtype=np.int64),
        option_count=np.asarray([row["option_count"] for row in records], dtype=np.int64),
    )
    report = {
        "status": "complete_frozen_habit_transition_features",
        "records": len(records),
        "train_records": sum(row["split"] == "train" for row in records),
        "validation_records": sum(row["split"] == "validation" for row in records),
        "unique_train_videos": len({
            row["video_sha256"] for row in records if row["split"] == "train"
        }),
        "unique_validation_videos": len({
            row["video_sha256"] for row in records if row["split"] == "validation"
        }),
        "train_task_groups": len({row["sid"] for row in records if row["split"] == "train"}),
        "validation_task_groups": len({
            row["sid"] for row in records if row["split"] == "validation"
        }),
        "video_feature_shape": list(features.shape),
        "sensor_sequence_shape": [len(records), 8, 14],
        "source_revision": SOURCE_REVISION,
        "audit_receipt_sha256": AUDIT_SHA256,
        "videoprism_revision": VIDEO_REVISION,
        "videoprism_source_commit": source_commit,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "jax_devices": [str(device) for device in jax.devices()],
        "wall_seconds": time.perf_counter() - started,
        "video_decode_seconds": decode_seconds,
        "model_forward_seconds": forward_seconds,
        "working_set_bytes_at_exit": process.memory_info().rss,
        "training_performed": False,
        "sealed_audit_payload_accessed": False,
    }
    (output_dir / "worker-result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-receipt", type=Path)
    parser.add_argument("--payload-root", type=Path)
    parser.add_argument("--worker-python", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if args.worker:
        if args.checkpoint is None or args.source_root is None:
            parser.error("--worker requires --checkpoint and --source-root")
        _worker(output_dir / "transition-inputs.json", output_dir,
                args.checkpoint.resolve(strict=True), args.source_root.resolve(strict=True))
        return
    required = (args.audit_receipt, args.payload_root, args.worker_python,
                args.checkpoint, args.source_root)
    if any(value is None for value in required):
        parser.error("preparation requires audit, payload, worker, checkpoint, and source paths")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    output_dir.mkdir(parents=True)
    input_path = _prepare_inputs(
        args.audit_receipt.resolve(strict=True), args.payload_root.resolve(strict=True), output_dir
    )
    subprocess.run(
        [
            str(args.worker_python.resolve(strict=True)), str(Path(__file__).resolve()),
            "--worker", "--output-dir", str(output_dir), "--checkpoint",
            str(args.checkpoint.resolve(strict=True)), "--source-root",
            str(args.source_root.resolve(strict=True)),
        ],
        check=True,
        env={**os.environ, "JAX_PLATFORMS": "cpu", "CUDA_VISIBLE_DEVICES": ""},
    )
    final = json.loads((output_dir / "worker-result.json").read_text(encoding="utf-8"))
    final.update({
        "input_manifest_sha256": sha256_file(input_path),
        "feature_file_sha256": sha256_file(output_dir / "habit-transition-features.npz"),
        "extractor_sha256": sha256_file(Path(__file__).resolve()),
    })
    (output_dir / "result.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
