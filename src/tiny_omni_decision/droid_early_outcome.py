"""Build a group-safe, early DROID success/failure decision benchmark manifest."""

from __future__ import annotations

import hashlib
import math
from pathlib import PurePosixPath
from typing import Any

SOURCE_FILE_KEY = "episode_metadata/file_path"
RECORDING_GROUP_KEY = "episode_metadata/recording_folderpath"
OPTION_IDS = ("success", "failure")


def canonical_path(value: str) -> str:
    return "/".join(part for part in value.replace("\\", "/").casefold().split("/") if part)


def stable_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def source_outcome(source_file: str) -> str:
    parts = PurePosixPath(canonical_path(source_file)).parts
    labels = [part for part in parts if part in OPTION_IDS]
    if len(labels) != 1:
        raise ValueError("source path must contain exactly one success/failure label segment")
    return labels[0]


def uniform_indices(step_count: int, requested: int) -> list[int]:
    if step_count < 1 or requested < 1 or step_count < requested:
        raise ValueError("step count must be positive and cover all requested observations")
    if requested == 1:
        return [0]
    return [round(index * (step_count - 1) / (requested - 1)) for index in range(requested)]


def build_early_outcome_rows(
    audit: dict[str, Any],
    split: dict[str, Any],
    *,
    observation_count: int = 8,
    observed_fraction: float = 0.25,
) -> dict[str, Any]:
    if observation_count < 1 or not 0 < observed_fraction <= 1:
        raise ValueError("invalid observation window configuration")
    if split.get("sealed_audit") is not None:
        raise ValueError("this development task builder must not load a sealed audit")

    episode_by_id: dict[str, dict[str, Any]] = {}
    for episode in audit.get("episodes", []):
        source_path = str(episode[SOURCE_FILE_KEY])
        episode_id = stable_id(canonical_path(source_path))
        if episode_id in episode_by_id:
            raise ValueError("duplicate normalized source-file identity in audit")
        episode_by_id[episode_id] = episode

    split_ids: dict[str, list[str]] = {}
    for name in ("train", "validation"):
        part = split.get(name, {})
        ids = list(part.get("episode_ids", []))
        if not ids or len(ids) != len(set(ids)):
            raise ValueError(f"{name} split must have unique non-empty episode IDs")
        if len(ids) != int(part.get("episode_count", -1)):
            raise ValueError(f"{name} split episode count does not match its IDs")
        split_ids[name] = ids

    if set(split_ids["train"]) & set(split_ids["validation"]):
        raise ValueError("train/validation source identity overlap")
    if set(split_ids["train"]) | set(split_ids["validation"]) != set(episode_by_id):
        raise ValueError("development split must assign every audited episode exactly once")

    group_sets = {
        name: {
            canonical_path(str(episode_by_id[episode_id][RECORDING_GROUP_KEY]))
            for episode_id in ids
        }
        for name, ids in split_ids.items()
    }
    if group_sets["train"] & group_sets["validation"]:
        raise ValueError("train/validation recording-group overlap")

    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    excluded: dict[str, dict[str, int]] = {"train": {}, "validation": {}}
    for split_name, ids in split_ids.items():
        for episode_id in ids:
            episode = episode_by_id[episode_id]
            step_count = int(episode["step_count"])
            early_step_count = math.floor(step_count * observed_fraction)
            if early_step_count < observation_count:
                label = source_outcome(str(episode[SOURCE_FILE_KEY]))
                excluded[split_name][label] = excluded[split_name].get(label, 0) + 1
                continue
            indices = uniform_indices(early_step_count, observation_count)
            label = source_outcome(str(episode[SOURCE_FILE_KEY]))
            rows_by_split[split_name].append(
                {
                    "id": episode_id,
                    "group_id": stable_id(canonical_path(str(episode[RECORDING_GROUP_KEY]))),
                    "split": split_name,
                    "shard_index": int(episode["shard_index"]),
                    "source_step_count": step_count,
                    "observed_step_count": early_step_count,
                    "observation_indices": indices,
                    "video_views": [
                        "exterior_image_1_left",
                        "exterior_image_2_left",
                        "wrist_image_left",
                    ],
                    "sensor_keys": [
                        "observation/joint_position",
                        "observation/cartesian_position",
                        "observation/gripper_position",
                    ],
                    "target_option_id": label,
                    "options": list(OPTION_IDS),
                }
            )

    for name, rows in rows_by_split.items():
        group_ids = [row["group_id"] for row in rows]
        row_ids = [row["id"] for row in rows]
        if len(group_ids) != len(set(group_ids)) or len(row_ids) != len(set(row_ids)):
            raise ValueError(f"{name} contains repeated episode or recording identities")
    train_groups = {row["group_id"] for row in rows_by_split["train"]}
    validation_groups = {row["group_id"] for row in rows_by_split["validation"]}
    if train_groups & validation_groups:
        raise ValueError("windowed task has train/validation group overlap")

    return {
        "schema_version": 1,
        "task_id": "droid-early-outcome-video-sensor-v0",
        "claim_scope": "source-folder early outcome prediction; development only",
        "options": list(OPTION_IDS),
        "target_provenance": "success/failure path segment; no path string is a model input",
        "input_contract": {
            "modalities": ["video", "sensor"],
            "video_views": [
                "exterior_image_1_left",
                "exterior_image_2_left",
                "wrist_image_left",
            ],
            "sensor_fields": [
                "observation/joint_position",
                "observation/cartesian_position",
                "observation/gripper_position",
            ],
            "text": "excluded",
            "audio": "not present in DROID",
            "observed_fraction": observed_fraction,
            "observation_count": observation_count,
            "timebase": "step index only; no explicit per-step timestamps in pinned schema",
            "future_observations_in_input": False,
        },
        "splits": {
            name: {
                "episode_count": len(rows),
                "recording_groups": len({row["group_id"] for row in rows}),
                "target_counts": {
                    option_id: sum(row["target_option_id"] == option_id for row in rows)
                    for option_id in OPTION_IDS
                },
                "excluded_short_episode_counts": excluded[name],
                "rows": rows,
            }
            for name, rows in rows_by_split.items()
        },
        "sealed_audit_loaded": False,
    }
