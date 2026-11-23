"""Metadata-only, prompt-grouped RoboPro development split builder.

This module does not read Parquet/video payloads or produce a sealed audit.
It creates a project development split from the publisher's TRAIN episodes.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

ROBOPRO_REVISION = "90ec789bf4018eb9c0f75da9f69aab5c185f0fd0"
DUPLICATE_EPISODE_ID = 15999
DUPLICATE_OF_EPISODE_ID = 12000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number} must contain a JSON object")
                rows.append(value)
    return rows


def _split_range(value: str) -> tuple[int, int]:
    try:
        start, stop = (int(part) for part in value.split(":"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unsupported RoboPro split range: {value!r}") from exc
    if start < 0 or stop <= start:
        raise ValueError(f"invalid RoboPro split range: {value!r}")
    return start, stop


def _id_hash(ids: list[int]) -> str:
    payload = "".join(f"{episode_id}\n" for episode_id in sorted(ids)).encode()
    return hashlib.sha256(payload).hexdigest()


def build_prompt_group_split(
    info: dict[str, Any],
    task_rows: list[dict[str, Any]],
    episode_rows: list[dict[str, Any]],
    *,
    seed: int = 17,
    development_fraction: float = 0.2,
) -> dict[str, Any]:
    """Split whole exact-prompt groups; exclude the publisher-declared duplicate.

    Groups are assigned as units, so an exact task string cannot occur in both
    training and development. This does not establish semantic parent-task or
    physical-scene disjointness; those identifiers are absent from metadata.
    """
    if not 0.0 < development_fraction < 1.0:
        raise ValueError("development_fraction must be between 0 and 1")
    if info.get("total_episodes") != len(episode_rows):
        raise ValueError("episode metadata count does not match info.json")
    if info.get("total_tasks") != len(task_rows):
        raise ValueError("task metadata count does not match info.json")
    splits = info.get("splits")
    if not isinstance(splits, dict) or set(splits) != {"train"}:
        raise ValueError("expected the pinned RoboPro release to contain TRAIN only")
    start, stop = _split_range(splits["train"])
    if stop - start != len(episode_rows):
        raise ValueError("TRAIN split range does not match episode metadata count")

    task_by_id: dict[int, str] = {}
    for row in task_rows:
        task_id, task = row.get("task_index"), row.get("task")
        if not isinstance(task_id, int) or not isinstance(task, str) or not task:
            raise ValueError("each task requires an integer index and non-empty text")
        if task_id in task_by_id or task in task_by_id.values():
            raise ValueError("task indices and exact task strings must be unique")
        task_by_id[task_id] = task
    if set(task_by_id) != set(range(len(task_rows))):
        raise ValueError("task indices must match the complete zero-based metadata range")

    by_id: dict[int, dict[str, Any]] = {}
    for row in episode_rows:
        episode_id, length, tasks = row.get("episode_index"), row.get("length"), row.get("tasks")
        if not isinstance(episode_id, int) or not isinstance(length, int) or length < 1:
            raise ValueError("episodes require an integer id and positive length")
        if episode_id in by_id:
            raise ValueError("episode ids must be unique")
        if not isinstance(tasks, list) or len(tasks) != 1 or tasks[0] not in task_by_id.values():
            raise ValueError("each episode must map to exactly one known task prompt")
        by_id[episode_id] = row
    if set(by_id) != set(range(start, stop)):
        raise ValueError("episode ids must match the publisher's complete TRAIN range")

    duplicate = by_id.get(DUPLICATE_EPISODE_ID)
    canonical = by_id.get(DUPLICATE_OF_EPISODE_ID)
    if duplicate is None or canonical is None:
        raise ValueError("publisher-declared duplicate pair is missing")
    if duplicate["tasks"] != canonical["tasks"] or duplicate["length"] != canonical["length"]:
        raise ValueError("publisher-declared duplicate metadata no longer matches")

    groups: dict[str, list[int]] = defaultdict(list)
    for episode_id, row in by_id.items():
        if episode_id == DUPLICATE_EPISODE_ID:
            continue
        groups[row["tasks"][0]].append(episode_id)

    prompts = sorted(groups)
    random.Random(seed).shuffle(prompts)
    n_development_groups = round(len(prompts) * development_fraction)
    n_development_groups = min(max(n_development_groups, 1), len(prompts) - 1)
    development_prompts = set(prompts[:n_development_groups])
    train_ids = sorted(
        episode_id
        for prompt, ids in groups.items()
        if prompt not in development_prompts
        for episode_id in ids
    )
    development_ids = sorted(
        episode_id
        for prompt, ids in groups.items()
        if prompt in development_prompts
        for episode_id in ids
    )
    train_prompts = sorted(set(groups) - development_prompts)
    development_prompts_sorted = sorted(development_prompts)
    eligible_minimum = 4  # One target episode plus three distinct donor episodes.

    return {
        "schema_version": "robopro-prompt-grouped-development-split-v1",
        "dataset": "Hoshipu/RoboPro",
        "revision": ROBOPRO_REVISION,
        "seed": seed,
        "split_policy": "whole exact task-prompt groups; deterministic seeded assignment",
        "development_fraction_of_prompt_groups": development_fraction,
        "source_has_publisher_train_only": True,
        "duplicate_excluded": {
            "episode_id": DUPLICATE_EPISODE_ID,
            "duplicate_of_episode_id": DUPLICATE_OF_EPISODE_ID,
            "basis": "publisher-declared duplicate; metadata prompt and length also match",
        },
        "limitations": [
            "Exact prompt groups are not verified semantic parent-task groups.",
            "Physical scene identity is not exposed in the inspected metadata.",
            "Only metadata was read; no action choices, features, or model scores were generated.",
            "Development data comes from publisher TRAIN; it is not a sealed or official test set.",
            "The proposed target is the recorded action, not task success or optimality.",
        ],
        "action_candidate_group_minimum_episodes": eligible_minimum,
        "train": {
            "episode_ids": train_ids,
            "prompt_groups": train_prompts,
            "episode_count": len(train_ids),
            "prompt_group_count": len(train_prompts),
            "eligible_prompt_groups_for_4_choice": sum(
                len(groups[prompt]) >= eligible_minimum for prompt in train_prompts
            ),
            "episode_id_sha256_newline_sorted": _id_hash(train_ids),
            "prompt_sha256_newline_sorted": hashlib.sha256(
                "".join(f"{prompt}\n" for prompt in train_prompts).encode()
            ).hexdigest(),
        },
        "development": {
            "episode_ids": development_ids,
            "prompt_groups": development_prompts_sorted,
            "episode_count": len(development_ids),
            "prompt_group_count": len(development_prompts_sorted),
            "eligible_prompt_groups_for_4_choice": sum(
                len(groups[prompt]) >= eligible_minimum for prompt in development_prompts_sorted
            ),
            "episode_id_sha256_newline_sorted": _id_hash(development_ids),
            "prompt_sha256_newline_sorted": hashlib.sha256(
                "".join(f"{prompt}\n" for prompt in development_prompts_sorted).encode()
            ).hexdigest(),
        },
        "excluded_duplicate_episode_ids": [DUPLICATE_EPISODE_ID],
        "counts": {
            "source_episodes": len(episode_rows),
            "included_episodes": len(train_ids) + len(development_ids),
            "exact_prompt_groups": len(groups),
            "prompt_groups_with_at_least_4_episodes": sum(len(ids) >= 4 for ids in groups.values()),
            "prompt_groups_with_fewer_than_4_episodes": sum(
                len(ids) < 4 for ids in groups.values()
            ),
            "train_development_episode_overlap": 0,
            "train_development_prompt_overlap": 0,
        },
    }


def build_from_paths(
    info_path: Path,
    tasks_path: Path,
    episodes_path: Path,
    **kwargs: Any,
) -> dict[str, Any]:
    info = json.loads(info_path.read_text(encoding="utf-8"))
    task_rows = _read_jsonl(tasks_path)
    episode_rows = _read_jsonl(episodes_path)
    result = build_prompt_group_split(info, task_rows, episode_rows, **kwargs)
    result["source_files"] = {
        name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for name, path in (
            ("info.json", info_path),
            ("tasks.jsonl", tasks_path),
            ("episodes.jsonl", episodes_path),
        )
    }
    return result
