"""Deterministic, metadata-only prompt-group sample selection for RoboPro probes."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any


def _order_key(seed: int, split: str, prompt: str, episode_id: int | None = None) -> str:
    suffix = "prompt" if episode_id is None else f"episode:{episode_id}"
    value = f"robopro-action-probe-v1|{seed}|{split}|{prompt}|{suffix}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_probe_sample(
    episodes: list[dict[str, Any]],
    split_manifest: dict[str, Any],
    *,
    seed: int = 17,
    prompts_per_split: int = 16,
    episodes_per_prompt: int = 4,
    excluded_episode_ids: set[int] | None = None,
) -> dict[str, Any]:
    """Select whole exact-prompt groups without reading labels or media bytes."""
    if seed < 0 or prompts_per_split < 1 or episodes_per_prompt < 2:
        raise ValueError("seed and sample sizes must be non-negative and positive")
    if split_manifest.get("seed") != seed:
        raise ValueError("sample seed must match the frozen split manifest")
    by_id = {int(row["episode_index"]): row for row in episodes}
    if len(by_id) != len(episodes):
        raise ValueError("episode metadata contains duplicate IDs")
    excluded = excluded_episode_ids or set()
    rows: dict[str, Any] = {}
    chosen_prompt_sets: dict[str, set[str]] = {}

    for split in ("train", "development"):
        split_ids = [int(value) for value in split_manifest[split]["episode_ids"]]
        if not set(split_ids) <= by_id.keys():
            raise ValueError(f"{split} split references unknown episode IDs")
        grouped: dict[str, list[int]] = defaultdict(list)
        for episode_id in split_ids:
            row = by_id[episode_id]
            tasks = row.get("tasks")
            if not isinstance(tasks, list) or len(tasks) != 1 or not tasks[0]:
                raise ValueError(f"episode {episode_id} has an invalid task prompt")
            grouped[str(tasks[0])].append(episode_id)

        excluded_prompts = {
            str(by_id[episode_id]["tasks"][0])
            for episode_id in excluded
            if episode_id in by_id and episode_id in split_ids
        }
        candidates = [
            prompt
            for prompt, ids in grouped.items()
            if len(ids) >= episodes_per_prompt and prompt not in excluded_prompts
        ]
        candidates.sort(key=lambda prompt: _order_key(seed, split, prompt))
        if len(candidates) < prompts_per_split:
            raise ValueError(
                f"{split} has only {len(candidates)} eligible prompts; need {prompts_per_split}"
            )
        selected = candidates[:prompts_per_split]
        selected_records = []
        for prompt in selected:
            prompt_ids = sorted(
                grouped[prompt], key=lambda episode_id: _order_key(seed, split, prompt, episode_id)
            )[:episodes_per_prompt]
            selected_records.append(
                {
                    "prompt": prompt,
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    "episodes": [
                        {
                            "episode_id": episode_id,
                            "length": int(by_id[episode_id]["length"]),
                        }
                        for episode_id in prompt_ids
                    ],
                }
            )
        rows[split] = selected_records
        chosen_prompt_sets[split] = set(selected)

    if chosen_prompt_sets["train"] & chosen_prompt_sets["development"]:
        raise ValueError("exact prompt overlap in the frozen split manifest")
    if excluded & {
        int(record["episode_id"])
        for split_rows in rows.values()
        for group in split_rows
        for record in group["episodes"]
    }:
        raise ValueError("selected sample overlaps the excluded smoke episodes")

    return {
        "schema_version": "robopro-action-probe-sample-v1",
        "dataset": split_manifest["dataset"],
        "revision": split_manifest["revision"],
        "seed": seed,
        "selection_method": "sha256 order by partition/prompt/episode; metadata only",
        "prompts_per_split": prompts_per_split,
        "episodes_per_prompt": episodes_per_prompt,
        "excludes_existing_smoke_episodes": sorted(excluded),
        "train": rows["train"],
        "development": rows["development"],
        "train_episode_count": sum(len(group["episodes"]) for group in rows["train"]),
        "development_episode_count": sum(len(group["episodes"]) for group in rows["development"]),
        "train_prompt_groups": len(rows["train"]),
        "development_prompt_groups": len(rows["development"]),
        "limitations": [
            "Exact prompts are not verified semantic parent tasks.",
            "Physical scene identity is absent from the inspected metadata.",
            "Development is project-created from publisher TRAIN, not a sealed test set.",
            "Selection uses metadata only and does not establish valid action labels.",
        ],
    }
