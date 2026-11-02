"""Extract frozen VideoPrism features for the pinned HABIT transition corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from tiny_omni_decision.habit_transition import (
    resolve_episode_options,
    transition_window_for_boundary,
)

DATASET_REVISION = "bae97d251907743f8fb6db7e70c61d361df4dc1e"
PLAN_SHA256 = "fe9d8012131db25578d25a6501a9c56c50dd2d4ddef33cfd2145d3aa81cb8f1b"
VIDEO_MODEL_REVISION = "be719a406d563b66f0ac969e7c94bab8e997c81a"
VIDEO_SOURCE_COMMIT = "c6eba0ece3dd49a24ffc44582966997a4a59f1d8"
VIDEO_CHECKPOINT_SHA256 = "61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29"
OPTION_MODEL_REVISION = "4ca70771034acceecb2e72475f72050fcdde4ddc"
OPTION_WEIGHT_SHA256 = "cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433"
AUDIT_REPORT_SHA256 = "61e7f7617de732df3c84146d1409de4bf7b337fde3c3f3295db9cf321adb3cd4"
OPTION_RECEIPT_SHA256 = "3be4e27d0668a75773bcb74e03c68b87957c626d620951f3646b828e0659e6bc"
EXPECTED_WINDOWS = {"train": 936, "validation": 144}
STATE_FIELDS = tuple(
    f"robot{robot}.observation.state.{field}"
    for robot in (0, 1)
    for field in ("cartesian_position", "joint_positions", "joint_velocities", "gripper_position")
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _episode_samples(
    episode_id: int,
    split: str,
    parent_group: str,
    labels: list[int],
    state_rows: np.ndarray,
    option_ids: tuple[int, ...],
) -> list[dict[str, Any]]:
    records = []
    for boundary in range(1, len(labels)):
        if labels[boundary] == labels[boundary - 1]:
            continue
        window = transition_window_for_boundary(labels, boundary=boundary, option_ids=option_ids)
        if window is None:
            continue
        records.append(
            {
                "sample_id": f"{episode_id}:{boundary}",
                "episode_id": episode_id,
                "split": split,
                "parent_group": parent_group,
                "history_indices": window.history_indices,
                "target_index": window.target_index,
                "target_option_id": window.target_option_id,
                "option_ids": option_ids,
                "target_position": option_ids.index(window.target_option_id),
                "state": state_rows[list(window.history_indices)],
            }
        )
    return records


def _build_records(attempt_root: Path, metadata_root: Path, audit_path: Path):
    import pyarrow.parquet as pq

    plan_path = attempt_root / "metadata-plan.json"
    if sha256_file(plan_path) != PLAN_SHA256:
        raise ValueError("metadata-plan hash differs from frozen preflight")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        audit.get("format") != "habit_transition_dataset_audit_v1"
        or audit.get("limited_smoke")
        or audit.get("reserved_media_loaded")
        or audit.get("dataset") != "configinc/HABIT"
        or audit.get("revision") != DATASET_REVISION
    ):
        raise ValueError("full audit receipt is missing or incompatible")
    if sha256_file(audit_path) != AUDIT_REPORT_SHA256:
        raise ValueError("full audit receipt SHA-256 mismatch")
    episodes = {
        int(row["episode_index"]): row
        for row in read_jsonl(metadata_root / "full__meta__episodes.jsonl")
    }
    subtasks = {
        int(row["task_index"]): row["task"]
        for row in read_jsonl(metadata_root / "full__meta__subtasks.jsonl")
    }
    option_receipt_path = OPTION_CACHE_DIR / "receipt.json"
    if sha256_file(option_receipt_path) != OPTION_RECEIPT_SHA256:
        raise ValueError("pinned option feature receipt SHA-256 mismatch")
    with np.load(OPTION_CACHE_DIR / "option-vectors.npz", allow_pickle=False) as cache:
        cached_ids = cache["option_ids"].astype(np.int64)
        cached_vectors = cache["vectors"].astype(np.float32)
    option_vectors = {int(key): cached_vectors[index] for index, key in enumerate(cached_ids)}

    entries: dict[str, dict[int, dict[str, dict]]] = {
        "train": defaultdict(dict),
        "validation": defaultdict(dict),
    }
    for item in plan["file_tree_files"]:
        if item["split"] not in entries:
            raise ValueError("reserved or unknown file in frozen plan")
        filename = Path(item["path"]).name
        episode_id = int(filename.split("episode_")[1].split(".")[0])
        kind = (
            "video"
            if filename.endswith(".mp4")
            else "parquet"
            if filename.endswith(".parquet")
            else None
        )
        if kind is None or kind in entries[item["split"]][episode_id]:
            raise ValueError(f"unexpected/duplicate plan file: {item['path']}")
        entries[item["split"]][episode_id][kind] = item

    records_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    option_texts: dict[int, str] = {}
    episode_items: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    for split in ("train", "validation"):
        for plan_item in plan["sampled_episode_ids"][split]:
            episode_id = int(plan_item["episode_index"])
            row = episodes[episode_id]
            if row.get("sid") != plan_item["sid"]:
                raise ValueError(f"episode SID mismatch: {episode_id}")
            if plan_item["sid"] not in plan_item["parent_group"].split("_"):
                raise ValueError(f"episode outside parent group: {episode_id}")
            options = resolve_episode_options(row["low_level_tasks"], subtasks)
            if len(options) < 2:
                continue
            candidate_ids = tuple(option_id for option_id, _ in options)
            for option_id, text in options:
                if option_id not in option_vectors:
                    raise ValueError(f"missing pinned text vector for option {option_id}")
                previous = option_texts.setdefault(option_id, text)
                if previous != text:
                    raise ValueError(f"global option ID changes text: {option_id}")

            pair = entries[split][episode_id]
            if set(pair) != {"video", "parquet"}:
                raise ValueError(f"incomplete media pair: {episode_id}")
            media = {
                kind: attempt_root / "selected-media" / split / Path(item["path"]).name
                for kind, item in pair.items()
            }
            for kind, path in media.items():
                expected = pair[kind]
                if not path.is_file() or path.stat().st_size != expected["bytes"]:
                    raise ValueError(f"media missing or wrong size: {path}")
                if sha256_file(path) != expected["sha256"]:
                    raise ValueError(f"media SHA-256 mismatch: {path}")
            columns = ["frame_index", "low_level_task_index", *STATE_FIELDS]
            table = pq.read_table(media["parquet"], columns=columns)
            frame_indices = table["frame_index"].to_pylist()
            labels = table["low_level_task_index"].to_pylist()
            if frame_indices != list(range(len(labels))):
                raise ValueError(f"nonconsecutive Parquet frames: {episode_id}")
            if any(label not in candidate_ids for label in labels):
                raise ValueError(f"label outside supplied options: {episode_id}")
            fields = [
                np.asarray(table[name].to_pylist(), dtype=np.float32) for name in STATE_FIELDS
            ]
            state_rows = np.concatenate(fields, axis=1)
            if state_rows.shape != (len(labels), 42) or not np.isfinite(state_rows).all():
                raise ValueError(f"invalid proprioceptive state rows: {episode_id}")
            episode_records = _episode_samples(
                episode_id, split, plan_item["parent_group"], labels, state_rows, candidate_ids
            )
            expected_windows = len(episode_records)
            reported_group = audit["split_reports"][split]["task_groups"].get(
                plan_item["parent_group"], {}
            )
            if expected_windows < 0 or reported_group is None:
                raise ValueError(f"episode missing from full audit report: {episode_id}")
            records_by_split[split].extend(episode_records)
            if episode_records:
                episode_items[split].append(
                    {
                        "episode_id": episode_id,
                        "parent_group": plan_item["parent_group"],
                        "media": media,
                        "records": episode_records,
                    }
                )
    for split in ("train", "validation"):
        if len(records_by_split[split]) != EXPECTED_WINDOWS[split]:
            raise ValueError(f"window count mismatch for {split}: {len(records_by_split[split])}")
        sample_ids = [record["sample_id"] for record in records_by_split[split]]
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError(f"duplicate transition sample IDs in {split}")
    return plan, audit, records_by_split, episode_items, option_vectors


def _decode_requested_frames(path: Path, requested: set[int], av: Any) -> dict[int, np.ndarray]:
    frames: dict[int, np.ndarray] = {}
    with av.open(str(path)) as container:
        streams = [stream for stream in container.streams if stream.type == "video"]
        if len(streams) != 1:
            raise ValueError(f"expected one video stream: {path}")
        for index, frame in enumerate(container.decode(streams[0])):
            if index in requested:
                frames[index] = frame.to_ndarray(format="rgb24")
    if frames.keys() != requested:
        raise ValueError(f"video did not return all requested frames: {path}")
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-root", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--option-cache-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    global OPTION_CACHE_DIR
    OPTION_CACHE_DIR = args.option_cache_dir.resolve(strict=True)
    if sha256_file(args.audit_report) != AUDIT_REPORT_SHA256:
        raise ValueError("pinned full audit receipt SHA-256 mismatch")
    if sha256_file(OPTION_CACHE_DIR / "receipt.json") != OPTION_RECEIPT_SHA256:
        raise ValueError("pinned option feature receipt SHA-256 mismatch")
    plan, audit, records_by_split, episode_items, option_vectors = _build_records(
        args.attempt_root.resolve(strict=True),
        args.metadata_root.resolve(strict=True),
        args.audit_report.resolve(strict=True),
    )

    checkpoint = args.checkpoint.resolve(strict=True)
    if sha256_file(checkpoint) != VIDEO_CHECKPOINT_SHA256:
        raise ValueError("VideoPrism checkpoint SHA-256 mismatch")
    source_root = args.source_root.resolve(strict=True)
    import subprocess

    source_commit = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if source_commit != VIDEO_SOURCE_COMMIT:
        raise ValueError(f"unexpected VideoPrism source commit: {source_commit}")

    config = {
        "dataset_revision": DATASET_REVISION,
        "metadata_plan_sha256": sha256_file(args.attempt_root / "metadata-plan.json"),
        "audit_report_sha256": AUDIT_REPORT_SHA256,
        "option_cache_sha256": sha256_file(OPTION_CACHE_DIR / "option-vectors.npz"),
        "option_model_revision": OPTION_MODEL_REVISION,
        "option_model_weight_sha256": OPTION_WEIGHT_SHA256,
        "video_model_revision": VIDEO_MODEL_REVISION,
        "video_source_commit": source_commit,
        "video_checkpoint_sha256": VIDEO_CHECKPOINT_SHA256,
        "video_layout": "8-frame consecutive windows; mean spatial tokens per frame",
        "proprioception_fields": list(STATE_FIELDS),
        "train_samples": len(records_by_split["train"]),
        "validation_samples": len(records_by_split["validation"]),
        "reserved_loaded": False,
    }
    config_hash = hashlib.sha256(
        (json.dumps(config, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()
    output = args.output_dir.resolve()
    config_path = output / "run-config.json"
    if output.exists():
        if not args.resume:
            raise FileExistsError("output exists; pass --resume only for the same frozen config")
        if not config_path.is_file() or json.loads(config_path.read_text()) != {
            "config": config,
            "config_sha256": config_hash,
        }:
            raise ValueError("resume config mismatch; refusing to reuse feature shards")
        if (output / "receipt.json").exists():
            raise FileExistsError("completed feature cache already exists")
    else:
        if args.resume:
            raise FileNotFoundError("--resume requested but output directory does not exist")
        output.mkdir(parents=True)
        config_path.write_text(
            json.dumps({"config": config, "config_sha256": config_hash}, indent=2) + "\n",
            encoding="utf-8",
        )

    import sys

    import av
    import jax
    import jax.numpy as jnp
    import tensorflow as tf
    from videoprism import models as vp

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.extract_frozen_videoprism_features import _preprocess
    from tiny_omni_decision.temporal_tokens import pool_spatial_tokens_per_frame

    model = vp.get_model("videoprism_public_v1_base")
    variables = vp.load_pretrained_weights(
        "videoprism_public_v1_base", checkpoint_path=str(checkpoint)
    )
    encode = jax.jit(lambda video: model.apply(variables, video, train=False))
    data_dir = output / "shards"
    data_dir.mkdir(exist_ok=True)
    progress_path = output / "progress.jsonl"
    progress = progress_path.open("a", encoding="utf-8", buffering=1)
    started = time.perf_counter()
    completed_samples = 0
    completed_episodes = 0
    for split in ("train", "validation"):
        for episode in episode_items[split]:
            episode_id = episode["episode_id"]
            shard_path = data_dir / split / f"episode_{episode_id:06d}.npz"
            shard_path.parent.mkdir(exist_ok=True)
            expected_ids = [record["sample_id"] for record in episode["records"]]
            if shard_path.exists():
                if not args.resume:
                    raise FileExistsError(f"existing feature shard without --resume: {shard_path}")
                with np.load(shard_path, allow_pickle=False) as shard:
                    if shard["sample_ids"].tolist() != expected_ids:
                        raise ValueError(f"resume shard sample IDs mismatch: {shard_path}")
                    if shard["video_features"].shape != (len(expected_ids), 8, 768):
                        raise ValueError(f"resume shard shape mismatch: {shard_path}")
                    if not np.isfinite(shard["video_features"]).all():
                        raise ValueError(f"resume shard contains non-finite features: {shard_path}")
                completed_samples += len(expected_ids)
                completed_episodes += 1
                continue
            if shard_path.with_suffix(".tmp.npz").exists():
                raise FileExistsError(f"interrupted temporary shard needs inspection: {shard_path}")

            history_indices = {
                index for record in episode["records"] for index in record["history_indices"]
            }
            frames = _decode_requested_frames(episode["media"]["video"], history_indices, av)
            video_features = []
            state_features = []
            target_positions = []
            target_ids = []
            candidate_ids = []
            candidate_vectors = []
            for record in episode["records"]:
                frame_batch = np.stack([frames[index] for index in record["history_indices"]])
                input_video = jnp.asarray(_preprocess(frame_batch, tf)[None, ...])
                tokens = encode(input_video)
                jax.tree_util.tree_map(
                    lambda value: (
                        value.block_until_ready() if hasattr(value, "block_until_ready") else value
                    ),
                    tokens,
                )
                token_values = tokens[0] if isinstance(tokens, tuple) else tokens
                pooled = pool_spatial_tokens_per_frame(np.asarray(token_values), 8).squeeze(0)
                if pooled.shape != (8, 768) or not np.isfinite(pooled).all():
                    raise FloatingPointError(f"invalid VideoPrism feature: {record['sample_id']}")
                if not np.isfinite(record["state"]).all():
                    raise FloatingPointError(f"non-finite state: {record['sample_id']}")
                ids = list(record["option_ids"])
                candidate_ids.append(ids)
                candidate_vectors.append(np.stack([option_vectors[item] for item in ids]))
                video_features.append(pooled.astype(np.float32, copy=False))
                state_features.append(record["state"].astype(np.float32, copy=False))
                target_positions.append(record["target_position"])
                target_ids.append(record["target_option_id"])

            max_options = max(len(values) for values in candidate_ids)
            padded_ids = np.full((len(expected_ids), max_options), -1, dtype=np.int64)
            padded_vectors = np.zeros((len(expected_ids), max_options, 384), dtype=np.float32)
            option_mask = np.zeros((len(expected_ids), max_options), dtype=np.bool_)
            for index, values in enumerate(candidate_ids):
                count = len(values)
                padded_ids[index, :count] = values
                padded_vectors[index, :count] = candidate_vectors[index]
                option_mask[index, :count] = True
            shard = {
                "sample_ids": np.asarray(expected_ids),
                "group_ids": np.asarray([episode["parent_group"]] * len(expected_ids)),
                "video_features": np.stack(video_features),
                "state_features": np.stack(state_features),
                "candidate_ids": padded_ids,
                "candidate_features": padded_vectors,
                "option_mask": option_mask,
                "target_positions": np.asarray(target_positions, dtype=np.int64),
                "target_ids": np.asarray(target_ids, dtype=np.int64),
                "config_sha256": np.asarray(config_hash),
            }
            temporary = shard_path.with_suffix(".tmp.npz")
            np.savez_compressed(temporary, **shard)
            temporary.replace(shard_path)
            shard_hash = sha256_file(shard_path)
            completed_samples += len(expected_ids)
            completed_episodes += 1
            elapsed = time.perf_counter() - started
            total = len(records_by_split["train"]) + len(records_by_split["validation"])
            progress.write(
                json.dumps(
                    {
                        "event": "episode_complete",
                        "split": split,
                        "episode_id": episode_id,
                        "samples": len(expected_ids),
                        "completed_samples": completed_samples,
                        "total_samples": total,
                        "completed_episodes": completed_episodes,
                        "elapsed_seconds": round(elapsed, 2),
                        "estimated_remaining_seconds": round(
                            elapsed / completed_samples * (total - completed_samples), 2
                        ),
                        "shard_sha256": shard_hash,
                    }
                )
                + "\n"
            )
    progress.close()

    for split in ("train", "validation"):
        actual_ids: list[str] = []
        for episode in episode_items[split]:
            shard_path = data_dir / split / f"episode_{episode['episode_id']:06d}.npz"
            with np.load(shard_path, allow_pickle=False) as shard:
                if shard["config_sha256"].item() != config_hash:
                    raise ValueError(f"feature shard config mismatch: {shard_path}")
                actual_ids.extend(shard["sample_ids"].tolist())
        expected_ids = [record["sample_id"] for record in records_by_split[split]]
        if actual_ids != expected_ids:
            raise ValueError(f"completed feature sample order mismatch for {split}")

    receipt = {
        "status": "complete_habit_frozen_videoprism_feature_cache",
        "config_sha256": config_hash,
        "dataset_revision": f"configinc/HABIT@{DATASET_REVISION}",
        "video_model_revision": VIDEO_MODEL_REVISION,
        "video_checkpoint_sha256": VIDEO_CHECKPOINT_SHA256,
        "video_source_commit": source_commit,
        "option_model_revision": OPTION_MODEL_REVISION,
        "option_model_weight_sha256": OPTION_WEIGHT_SHA256,
        "devices": [str(device) for device in jax.devices()],
        "feature_shape_per_example": [8, 768],
        "state_shape_per_example": [8, 42],
        "train_samples": len(records_by_split["train"]),
        "validation_samples": len(records_by_split["validation"]),
        "feature_shards": completed_episodes,
        "reserved_loaded": False,
        "wall_seconds": round(time.perf_counter() - started, 3),
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
