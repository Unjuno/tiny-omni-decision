"""Audit pinned RoboPro metadata without loading media or action payloads."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_metadata(
    input_dir: Path,
    *,
    revision: str,
    declared_duplicate_index: int,
    declared_duplicate_of: int,
) -> dict[str, Any]:
    """Summarize counts and prompt groups while refusing to imply split validity."""
    info_path = input_dir / "info.json"
    episodes_path = input_dir / "episodes.jsonl"
    tasks_path = input_dir / "tasks.jsonl"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    episodes = _read_jsonl(episodes_path)
    tasks = _read_jsonl(tasks_path)

    indices = [row.get("episode_index") for row in episodes]
    if any(not isinstance(index, int) for index in indices):
        raise ValueError("every episode must have an integer episode_index")
    if len(set(indices)) != len(indices):
        raise ValueError("episode_index values must be unique")
    expected_indices = list(range(len(episodes)))
    if sorted(indices) != expected_indices:
        raise ValueError("episode_index values must be contiguous from zero")
    if len(episodes) != info.get("total_episodes"):
        raise ValueError("episode row count does not match info.json")
    if len(tasks) != info.get("total_tasks"):
        raise ValueError("task row count does not match info.json")

    task_indices = [row.get("task_index") for row in tasks]
    if any(not isinstance(index, int) for index in task_indices):
        raise ValueError("every task must have an integer task_index")
    if len(set(task_indices)) != len(task_indices):
        raise ValueError("task_index values must be unique")

    prompt_counts: Counter[str] = Counter()
    for row in episodes:
        prompts = row.get("tasks")
        if not isinstance(prompts, list) or not prompts or not all(
            isinstance(prompt, str) and prompt.strip() for prompt in prompts
        ):
            raise ValueError("every episode must have at least one non-empty task prompt")
        prompt_counts[" | ".join(prompt.strip().casefold() for prompt in prompts)] += 1

    by_index = {row["episode_index"]: row for row in episodes}
    duplicate = by_index.get(declared_duplicate_index)
    source = by_index.get(declared_duplicate_of)
    if duplicate is None or source is None:
        raise ValueError("declared duplicate and source episode must exist")

    split_names = sorted(info.get("splits", {}))
    has_eval_split = any(
        name.casefold() in {"validation", "valid", "dev", "evaluation", "eval", "test"}
        for name in split_names
    )
    group_sizes = sorted(prompt_counts.values())
    return {
        "schema_version": 1,
        "source_revision": revision,
        "source_files_sha256": {
            name: _sha256(path)
            for name, path in (
                ("info.json", info_path),
                ("episodes.jsonl", episodes_path),
                ("tasks.jsonl", tasks_path),
            )
        },
        "declared_totals": {
            "episodes": info.get("total_episodes"),
            "frames": info.get("total_frames"),
            "tasks": info.get("total_tasks"),
            "videos": info.get("total_videos"),
            "fps": info.get("fps"),
        },
        "observed": {
            "episode_rows": len(episodes),
            "unique_episode_indices": len(set(indices)),
            "episode_index_min": min(indices),
            "episode_index_max": max(indices),
            "task_rows": len(tasks),
            "unique_task_indices": len(set(task_indices)),
            "unique_exact_prompt_groups": len(prompt_counts),
            "episodes_per_exact_prompt": {
                "min": min(group_sizes),
                "median": median(group_sizes),
                "max": max(group_sizes),
                "groups_with_one_episode": sum(size == 1 for size in group_sizes),
            },
        },
        "declared_duplicate": {
            "episode_index": declared_duplicate_index,
            "duplicate_of_episode_index": declared_duplicate_of,
            "source_declared_duplicate": True,
            "metadata_rows_equal": {
                "task_prompts": duplicate.get("tasks") == source.get("tasks"),
                "length": duplicate.get("length") == source.get("length"),
            },
            "media_bytes_compared": False,
        },
        "split_audit": {
            "declared_splits": info.get("splits", {}),
            "independent_validation_split_present": has_eval_split,
            "quality_evaluation_admitted": False,
            "note": "Exact prompt groups do not identify underlying scene groups.",
        },
        "scope": {
            "media_downloaded": False,
            "parquet_payloads_downloaded": False,
            "action_values_read": False,
            "training_or_inference_run": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--declared-duplicate-index", type=int, default=15999)
    parser.add_argument("--declared-duplicate-of", type=int, default=12000)
    args = parser.parse_args()
    result = audit_metadata(
        args.input_dir,
        revision=args.revision,
        declared_duplicate_index=args.declared_duplicate_index,
        declared_duplicate_of=args.declared_duplicate_of,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
