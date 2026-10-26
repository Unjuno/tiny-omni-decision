"""Decode only the frozen early-window frames and robot-state inputs on CPU."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import struct
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image

from scripts.audit_droid100_groups import (
    episode_example_bytes_values,
    episode_example_context,
    episode_example_float_values,
    iter_tfrecord,
)
from scripts.build_droid_group_split import _canonical_path
from scripts.download_droid_selected_shards import validate_manifest, verify_metadata

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_HASHES = {
    "manifest": "2bfb5626d203344bf2bd976f711c0ed5f7b4c1979932b7b48565329099207e64",
    "receipt": "2bfb6322e9c0c4649f875d8ad37fb3e990f786324fe2fdbd9cd6b2872af50667",
    "task_manifest": "b882b66f8be2dd38aa7d69badd3d00b93f58769a9daa2be5f38088fbe084ec80",
}
CAMERA_FEATURES = {
    "exterior_image_1_left": "steps/observation/exterior_image_1_left",
    "exterior_image_2_left": "steps/observation/exterior_image_2_left",
    "wrist_image_left": "steps/observation/wrist_image_left",
}
SENSOR_FEATURES = {
    "observation/joint_position": ("steps/observation/joint_position", 7),
    "observation/cartesian_position": ("steps/observation/cartesian_position", 6),
    "observation/gripper_position": ("steps/observation/gripper_position", 1),
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_json(path: Path, expected_sha256: str | None = None) -> tuple[dict[str, Any], str]:
    data = path.read_bytes()
    digest = sha256(data)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"SHA-256 mismatch for {path.name}")
    return json.loads(data), digest


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPOSITORY_ROOT
        / "manifests/pretrained_reuse/droid-selected-shards-seed17-20261026.json",
    )
    parser.add_argument("--receipt-json", type=Path, required=True)
    parser.add_argument("--task-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _input_fingerprint(record: bytes, row: dict[str, Any]) -> dict[str, Any]:
    indices = [int(index) for index in row["observation_indices"]]
    if len(indices) != 8 or len(set(indices)) != 8:
        raise ValueError("each episode must have eight unique input step indices")
    if indices != sorted(indices) or indices[-1] >= int(row["observed_step_count"]):
        raise ValueError("input step indices are unordered or exceed the observed window")

    video_hashes = {}
    media_bytes = 0
    decoded_images = 0
    for view, feature_key in CAMERA_FEATURES.items():
        frame_bytes = episode_example_bytes_values(record, feature_key)
        if len(frame_bytes) != int(row["source_step_count"]):
            raise ValueError(f"{view} frame count differs from source step count")
        raw_digest = hashlib.sha256()
        rgb_digest = hashlib.sha256()
        for index in indices:
            jpeg = frame_bytes[index]
            if not jpeg:
                raise ValueError(f"empty JPEG at {view} step {index}")
            with Image.open(io.BytesIO(jpeg)) as image:
                if image.format != "JPEG" or image.size != (320, 180):
                    raise ValueError(f"unexpected image format or dimensions: {view} step {index}")
                rgb = image.convert("RGB")
                rgb.load()
                if rgb.size != (320, 180):
                    raise ValueError(
                        f"decoded image size changed unexpectedly: {view} step {index}"
                    )
                rgb_digest.update(rgb.tobytes())
            raw_digest.update(len(jpeg).to_bytes(8, "little"))
            raw_digest.update(jpeg)
            media_bytes += len(jpeg)
            decoded_images += 1
        video_hashes[view] = {
            "ordered_jpeg_sha256": raw_digest.hexdigest(),
            "ordered_rgb_sha256": rgb_digest.hexdigest(),
        }

    sensor_hashes = {}
    sensor_steps = {}
    for name, (feature_key, width) in SENSOR_FEATURES.items():
        values = episode_example_float_values(record, feature_key)
        expected_count = int(row["source_step_count"]) * width
        if len(values) != expected_count:
            raise ValueError(f"{name} value count differs from expected dimension and steps")
        digest = hashlib.sha256()
        selected = []
        for index in indices:
            step_values = values[index * width : (index + 1) * width]
            if len(step_values) != width:
                raise ValueError(f"{name} state width mismatch at step {index}")
            selected.extend(step_values)
            for value in step_values:
                digest.update(struct.pack("<f", value))
        sensor_hashes[name] = digest.hexdigest()
        sensor_steps[name] = {"steps": len(indices), "values_per_step": width}

    return {
        "id": row["id"],
        "group_id": row["group_id"],
        "split": row["split"],
        "target_option_id": row["target_option_id"],
        "input_indices": indices,
        "video_hashes": video_hashes,
        "sensor_hashes": sensor_hashes,
        "sensor_shapes": sensor_steps,
        "decoded_images": decoded_images,
        "selected_jpeg_bytes": media_bytes,
    }


def main() -> None:
    args = _args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite input audit: {output}")
    if output.is_relative_to(REPOSITORY_ROOT.resolve()):
        raise ValueError("input audit output must be outside the repository")

    manifest_bytes = args.manifest.resolve(strict=True).read_bytes()
    manifest_sha = sha256(manifest_bytes)
    if manifest_sha != EXPECTED_HASHES["manifest"]:
        raise ValueError("selected-shard manifest SHA-256 differs from the fixed protocol")
    manifest = json.loads(manifest_bytes)
    validate_manifest(manifest)
    receipt, receipt_sha = _load_json(
        args.receipt_json.resolve(strict=True), EXPECTED_HASHES["receipt"]
    )
    task, task_sha = _load_json(
        args.task_manifest.resolve(strict=True), EXPECTED_HASHES["task_manifest"]
    )
    if receipt.get("manifest_sha256") != manifest_sha:
        raise ValueError("download receipt references a different selected-shard manifest")
    if receipt.get("episode_count") != manifest.get("expected_episode_count"):
        raise ValueError("download receipt episode count differs from the source manifest")
    verify_metadata(args.data_dir.resolve(strict=True).parents[2], manifest)

    rows_by_id: dict[str, dict[str, Any]] = {}
    for split_name in ("train", "validation"):
        for row in task["splits"][split_name]["rows"]:
            if row["split"] != split_name or row["id"] in rows_by_id:
                raise ValueError("task rows contain a split mismatch or repeated identity")
            rows_by_id[row["id"]] = row
    groups = {
        split_name: {row["group_id"] for row in task["splits"][split_name]["rows"]}
        for split_name in ("train", "validation")
    }
    if groups["train"] & groups["validation"]:
        raise ValueError("task input audit found train/validation group overlap")

    fingerprints = []
    found_ids = set()
    episodes_per_shard = {}
    for shard_row in manifest["selected_shards"]:
        shard_path = args.data_dir / Path(shard_row["object_name"]).name
        shard_episodes = 0
        for record in iter_tfrecord(shard_path):
            context = episode_example_context(record)
            episode_id = hashlib.sha256(
                _canonical_path(context["episode_metadata/file_path"]).encode("utf-8")
            ).hexdigest()
            row = rows_by_id.get(episode_id)
            if row is None:
                continue
            if episode_id in found_ids:
                raise ValueError("selected episode appears in multiple source records")
            found_ids.add(episode_id)
            shard_episodes += 1
            fingerprints.append(_input_fingerprint(record, row))
        if shard_episodes > int(shard_row["episodes"]):
            raise ValueError("selected input count exceeds the pinned shard episode count")
        episodes_per_shard[str(shard_row["index"])] = shard_episodes
    if found_ids != set(rows_by_id):
        raise ValueError("one or more task input rows cannot be found in selected source shards")

    split_summaries = {}
    for split_name in ("train", "validation"):
        rows = sorted(
            (row for row in fingerprints if row["split"] == split_name),
            key=lambda row: row["id"],
        )
        split_summaries[split_name] = {
            "episode_count": len(rows),
            "recording_groups": len({row["group_id"] for row in rows}),
            "target_counts": {
                option_id: sum(row["target_option_id"] == option_id for row in rows)
                for option_id in ("success", "failure")
            },
            "decoded_images": sum(row["decoded_images"] for row in rows),
            "selected_jpeg_bytes": sum(row["selected_jpeg_bytes"] for row in rows),
            "input_fingerprints": rows,
        }
    result = {
        "schema_version": 1,
        "status": "CPU_INPUT_SMOKE_PASS",
        "created_utc": datetime.now(UTC).isoformat(),
        "claim_scope": "raw input decoding and index integrity only; no model features or metrics",
        "selection_manifest_sha256": manifest_sha,
        "download_receipt_sha256": receipt_sha,
        "task_manifest_sha256": task_sha,
        "selected_shards_episode_counts": episodes_per_shard,
        "sealed_audit_loaded": False,
        "model_inference_or_training": False,
        "video_preprocessing_applied": False,
        "splits": split_summaries,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": result["status"],
                "episodes": {name: row["episode_count"] for name, row in split_summaries.items()},
                "decoded_images": sum(row["decoded_images"] for row in split_summaries.values()),
                "sealed_audit_loaded": False,
                "output": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
