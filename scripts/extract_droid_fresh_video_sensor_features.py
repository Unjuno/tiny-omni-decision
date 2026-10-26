"""Extract frozen VideoPrism and proprioception features for a fresh DROID task."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.audit_droid100_groups import (  # noqa: E402
    episode_example_bytes_values,
    episode_example_context,
    episode_example_float_values,
    iter_tfrecord,
)
from scripts.build_droid_group_split import _canonical_path  # noqa: E402
from tiny_omni_decision.droid_next_action_readout import ACTION_OPTIONS  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(task_path: Path) -> list[dict[str, Any]]:
    task = json.loads(task_path.read_text(encoding="utf-8"))
    rows = [
        row
        for name in ("train", "validation")
        for row in task["splits"][name]["rows"]
    ]
    ids = [row["id"] for row in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError("task must have non-empty unique rows")
    if task.get("sealed_audit_loaded") is not False:
        raise ValueError("feature extraction must not load a sealed audit")
    return rows


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite feature output: {output}")
    if output.is_relative_to(REPOSITORY_ROOT.resolve()):
        raise ValueError("feature cache must stay outside the repository")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["JAX_PLATFORMS"] = "cpu"

    import jax
    import jax.numpy as jnp
    import tensorflow as tf
    from videoprism import models as vp

    devices = jax.devices()
    if not devices or any(device.platform != "cpu" for device in devices):
        raise RuntimeError(f"this extraction is CPU-only; found JAX devices: {devices}")
    manifest_raw = args.manifest.read_bytes()
    manifest = json.loads(manifest_raw)
    config_raw = args.config.read_bytes()
    config = json.loads(config_raw)
    task_raw = args.task_manifest.read_bytes()
    rows = read_rows(args.task_manifest)
    row_by_id = {row["id"]: index for index, row in enumerate(rows)}
    labels_by_id = {row["id"]: row["target_option_id"] for row in rows}
    data_root = args.dataset_root.resolve(strict=True) / manifest["prefix"]
    model_source = args.model_source.resolve(strict=True)
    sys.path.insert(0, str(model_source))
    checkpoint = args.checkpoint.resolve(strict=True)
    checkpoint_hash = sha256_file(checkpoint)
    if config["source"]["selection_manifest_sha256"] != hashlib.sha256(
        manifest_raw
    ).hexdigest():
        raise ValueError("experiment config and shard manifest hashes differ")
    if config["representation"]["video_checkpoint_sha256"] != checkpoint_hash:
        raise ValueError("VideoPrism checkpoint does not match experiment config")

    output.mkdir(parents=True, exist_ok=False)
    video = np.lib.format.open_memmap(
        output / "video_features.npy", mode="w+", dtype="float32", shape=(len(rows), 3, 8, 768)
    )
    sensor = np.lib.format.open_memmap(
        output / "sensor_sequences.npy", mode="w+", dtype="float32", shape=(len(rows), 8, 14)
    )
    labels = np.lib.format.open_memmap(
        output / "labels.npy", mode="w+", dtype="uint8", shape=(len(rows),)
    )
    views = {
        "exterior_image_1_left": "steps/observation/exterior_image_1_left",
        "exterior_image_2_left": "steps/observation/exterior_image_2_left",
        "wrist_image_left": "steps/observation/wrist_image_left",
    }
    sensor_fields = (
        ("observation/joint_position", "steps/observation/joint_position", 7),
        ("observation/cartesian_position", "steps/observation/cartesian_position", 6),
        ("observation/gripper_position", "steps/observation/gripper_position", 1),
    )
    model = vp.get_model("videoprism_public_v1_base")
    variables = vp.load_pretrained_weights(
        "videoprism_public_v1_base", checkpoint_path=str(checkpoint)
    )
    forward = jax.jit(lambda batch: model.apply(variables, batch, train=False))
    seen: set[str] = set()
    started = time.perf_counter()
    progress_path = output / "progress.jsonl"
    with progress_path.open("x", encoding="utf-8", buffering=1) as progress:
        for shard in manifest["selected_shards"]:
            path = data_root / Path(shard["object_name"]).name
            shard_count = 0
            for record in iter_tfrecord(path):
                context = episode_example_context(record)
                identity = hashlib.sha256(
                    _canonical_path(context["episode_metadata/file_path"]).encode()
                ).hexdigest()
                shard_count += 1
                if identity not in row_by_id:
                    continue
                if identity in seen:
                    raise ValueError(f"duplicate selected episode identity: {identity}")
                index = row_by_id[identity]
                row = rows[index]
                indices = [int(value) for value in row["input_indices"]]
                if len(indices) != 8 or indices != sorted(indices):
                    raise ValueError("task input indices are not a fixed ordered eight-step window")
                step_count = int(row["source_step_count"])
                decoded_views = []
                for key in views.values():
                    raw_frames = episode_example_bytes_values(record, key)
                    if len(raw_frames) != step_count:
                        raise ValueError(f"frame count mismatch for {identity}: {key}")
                    decoded = []
                    for step in indices:
                        frame = tf.io.decode_jpeg(raw_frames[step], channels=3)
                        if tuple(frame.shape) != (180, 320, 3):
                            raise ValueError(f"unexpected frame geometry for {identity}")
                        decoded.append(frame.numpy())
                    decoded_views.append(np.stack(decoded))
                frames = np.stack(decoded_views)
                height, width = frames.shape[2:4]
                scale = max(288 / height, 288 / width)
                new_height, new_width = round(height * scale), round(width * scale)
                flat = frames.reshape(24, height, width, 3)
                resized = tf.image.resize(flat, (new_height, new_width), method="bilinear")
                cropped = tf.image.resize_with_crop_or_pad(resized, 288, 288)
                batch = cropped.numpy().reshape(3, 8, 288, 288, 3).astype(np.float32) / 255.0
                result = forward(jnp.asarray(batch))
                jax.tree_util.tree_map(
                    lambda value: value.block_until_ready()
                    if hasattr(value, "block_until_ready")
                    else value,
                    result,
                )
                tokens = result[0] if isinstance(result, tuple) else result
                token_array = np.asarray(tokens)
                if token_array.shape != (3, 2048, 768) or not np.isfinite(token_array).all():
                    raise ValueError(
                        f"invalid VideoPrism output for {identity}: {token_array.shape}"
                    )
                temporal = token_array.reshape(3, 8, 256, 768).mean(axis=2).astype(np.float32)
                video[index] = temporal
                for offset, (_, key, width_value) in zip(
                    (0, 7, 13), sensor_fields, strict=True
                ):
                    values = episode_example_float_values(record, key)
                    if len(values) != step_count * width_value:
                        raise ValueError(f"sensor shape mismatch for {identity}: {key}")
                    for time_index, step in enumerate(indices):
                        block = np.asarray(
                            values[step * width_value : (step + 1) * width_value],
                            dtype=np.float32,
                        )
                        if not np.isfinite(block).all():
                            raise ValueError(f"non-finite sensor feature for {identity}")
                        sensor[index, time_index, offset : offset + width_value] = block
                labels[index] = ACTION_OPTIONS.index(labels_by_id[identity])
                seen.add(identity)
                progress.write(
                    json.dumps(
                        {
                            "index": index,
                            "id": identity,
                            "split": row["split"],
                            "label": int(labels[index]),
                            "video_feature_sha256": hashlib.sha256(temporal.tobytes()).hexdigest(),
                            "elapsed_seconds": time.perf_counter() - started,
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                if len(seen) % 10 == 0:
                    video.flush()
                    sensor.flush()
                    labels.flush()
                    print(f"extracted {len(seen)}/{len(rows)}", flush=True)
            if shard_count != int(shard["episodes"]):
                raise ValueError(f"source shard {shard['index']} episode count mismatch")
    if seen != set(row_by_id):
        raise ValueError(f"feature cache missing {len(set(row_by_id) - seen)} task rows")
    video.flush()
    sensor.flush()
    labels.flush()
    del video, sensor, labels
    summary = {
        "status": "PASS",
        "count": len(rows),
        "train_count": sum(row["split"] == "train" for row in rows),
        "validation_count": sum(row["split"] == "validation" for row in rows),
        "video_shape": [len(rows), 3, 8, 768],
        "sensor_shape": [len(rows), 8, 14],
        "target_options": list(ACTION_OPTIONS),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "experiment_config_sha256": hashlib.sha256(config_raw).hexdigest(),
        "model_id": "google/videoprism-base-f16r288",
        "model_revision": "be719a406d563b66f0ac969e7c94bab8e997c81a",
        "jax_devices": [str(device) for device in devices],
        "elapsed_seconds": time.perf_counter() - started,
        "sealed_audit_loaded": False,
        "model_training": False,
        "task_manifest_sha256": hashlib.sha256(task_raw).hexdigest(),
        "shard_manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
    }
    for filename in ("video_features.npy", "sensor_sequences.npy", "labels.npy", "progress.jsonl"):
        summary[f"{filename}_sha256"] = sha256_file(output / filename)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
