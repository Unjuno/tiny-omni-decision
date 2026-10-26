"""Task manifest builder for a fresh, group-safe DROID next-action cohort."""

from __future__ import annotations

import hashlib
import math
from typing import Any

from tiny_omni_decision.droid_next_action_readout import (
    ACTION_OPTIONS,
    next_cartesian_direction_label,
)


def canonical_path(value: str) -> str:
    return "/".join(part for part in value.replace("\\", "/").casefold().split("/") if part)


def stable_id(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def uniform_indices(length: int, count: int) -> list[int]:
    if length < count or count < 1:
        raise ValueError("input window must cover every requested observation")
    if count == 1:
        return [0]
    return [round(index * (length - 1) / (count - 1)) for index in range(count)]


def build_task_manifest(
    audit: dict[str, Any],
    split: dict[str, Any],
    target_labels: dict[str, str],
    *,
    observation_count: int = 8,
    observed_fraction: float = 0.25,
) -> dict[str, Any]:
    if (
        observation_count < 1
        or not math.isfinite(observed_fraction)
        or not 0 < observed_fraction <= 1
    ):
        raise ValueError("invalid observation window configuration")
    if split.get("sealed_audit") is not None:
        raise ValueError("development task builder must not load a sealed audit")
    episodes: dict[str, dict[str, Any]] = {}
    for row in audit.get("episodes", []):
        identity = stable_id(canonical_path(str(row["episode_metadata/file_path"])))
        if identity in episodes:
            raise ValueError("duplicate source identity in audited episodes")
        episodes[identity] = row
    assigned: dict[str, str] = {}
    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    group_sets: dict[str, set[str]] = {"train": set(), "validation": set()}
    for name in ("train", "validation"):
        part = split.get(name, {})
        ids = list(part.get("episode_ids", []))
        if not ids or len(ids) != len(set(ids)):
            raise ValueError(f"{name} split must have unique non-empty episode IDs")
        if len(ids) != int(part.get("episode_count", -1)):
            raise ValueError(f"{name} split count does not match episode IDs")
        for identity in ids:
            if identity not in episodes or identity in assigned:
                raise ValueError("split has an unknown or repeated source identity")
            assigned[identity] = name
            episode = episodes[identity]
            group_path = canonical_path(
                str(episode["episode_metadata/recording_folderpath"])
            )
            group_id = stable_id(group_path)
            group_sets[name].add(group_id)
            steps = int(episode["step_count"])
            observed_steps = math.floor(steps * observed_fraction)
            if observed_steps < observation_count:
                continue
            if identity not in target_labels or target_labels[identity] not in ACTION_OPTIONS:
                raise ValueError("eligible episode is missing a valid next-action target")
            indices = uniform_indices(observed_steps, observation_count)
            target_index = indices[-1] + 1
            if target_index >= steps:
                raise ValueError("next-action target lies outside the source episode")
            rows_by_split[name].append(
                {
                    "id": identity,
                    "group_id": group_id,
                    "split": name,
                    "shard_index": int(episode["shard_index"]),
                    "source_step_count": steps,
                    "observed_step_count": observed_steps,
                    "input_indices": indices,
                    "target_index": target_index,
                    "target_option_id": target_labels[identity],
                    "options": list(ACTION_OPTIONS),
                }
            )
    if set(assigned) != set(episodes):
        raise ValueError("train/validation must assign every audited episode exactly once")
    if group_sets["train"] & group_sets["validation"]:
        raise ValueError("recording-group overlap across train and validation")
    if not set(target_labels) <= set(episodes):
        raise ValueError("target extraction references an unaudited source episode")
    return {
        "schema_version": 1,
        "task_id": "droid-fresh-next-cartesian-direction-seed17-v1",
        "claim_scope": "fresh-shard, development-only one-step robot action-direction readout",
        "source_partition": "official DROID train shards; local group-safe development split",
        "options": list(ACTION_OPTIONS),
        "target_definition": (
            "Dominant signed xyz axis of commanded Cartesian velocity at the step "
            "immediately after the last observed state; stop when max |xyz| <= 1e-4."
        ),
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
            "observation_count": observation_count,
            "observed_fraction": observed_fraction,
            "timebase": "uniform step indices; no explicit per-step timestamps in pinned schema",
            "future_observations_in_input": False,
            "action_target_index": "last input index + 1",
        },
        "splits": {
            name: {
                "episode_count": len(rows),
                "group_count": len({row["group_id"] for row in rows}),
                "target_counts": {
                    option: sum(row["target_option_id"] == option for row in rows)
                    for option in ACTION_OPTIONS
                },
                "rows": rows,
            }
            for name, rows in rows_by_split.items()
        },
        "excluded_short_episode_count": len(episodes)
        - sum(len(rows) for rows in rows_by_split.values()),
        "sealed_audit_loaded": False,
    }


def label_from_velocity(values: list[float], step_count: int, target_index: int) -> str:
    if len(values) != step_count * 6 or not 0 <= target_index < step_count:
        raise ValueError("commanded velocity shape or target index is invalid")
    velocity = [float(value) for value in values[target_index * 6 : (target_index + 1) * 6]]
    import numpy as np

    return ACTION_OPTIONS[next_cartesian_direction_label(np.asarray(velocity, dtype=np.float64))]
