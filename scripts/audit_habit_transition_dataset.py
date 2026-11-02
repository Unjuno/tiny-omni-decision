"""Verify the frozen HABIT train/validation transition corpus before encoding."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from tiny_omni_decision.habit_transition import (
    resolve_episode_options,
    transition_window_for_boundary,
)

EXPECTED_REVISION = "bae97d251907743f8fb6db7e70c61d361df4dc1e"
EXPECTED_FILES = 816


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def audit(
    attempt_root: Path,
    metadata_root: Path,
    *,
    limit_episodes: int | None = None,
    selected_splits: tuple[str, ...] = ("train", "validation"),
) -> dict:
    import av
    import pyarrow.parquet as pq

    plan_path = attempt_root / "metadata-plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("dataset") != "configinc/HABIT" or plan.get("revision") != EXPECTED_REVISION:
        raise ValueError("dataset/revision differs from the frozen preflight")
    entries = plan.get("file_tree_files")
    if not isinstance(entries, list) or len(entries) != EXPECTED_FILES:
        raise ValueError("frozen media list must contain exactly 816 train/validation files")
    if any(item.get("split") not in {"train", "validation"} for item in entries):
        raise ValueError("reserved or unknown split entry present; refusing to load it")

    split_groups = plan["split_groups"]
    if set(split_groups["train"]) & set(split_groups["validation"]):
        raise ValueError("train/validation parent task-group overlap")
    if set(split_groups["train"]) & set(split_groups["reserved"]):
        raise ValueError("train/reserved parent task-group overlap")
    if set(split_groups["validation"]) & set(split_groups["reserved"]):
        raise ValueError("validation/reserved parent task-group overlap")

    episodes = {
        int(row["episode_index"]): row
        for row in read_jsonl(metadata_root / "full__meta__episodes.jsonl")
    }
    subtasks = {
        int(row["task_index"]): row["task"]
        for row in read_jsonl(metadata_root / "full__meta__subtasks.jsonl")
    }
    tasks = {
        int(row["task_index"]): row["task"]
        for row in read_jsonl(metadata_root / "full__meta__tasks.jsonl")
    }
    entry_by_split_episode: dict[str, dict[int, dict[str, dict]]] = {
        "train": defaultdict(dict),
        "validation": defaultdict(dict),
    }
    for item in entries:
        name = Path(item["path"]).name
        episode_id = int(name.split("episode_")[1].split(".")[0])
        if name.endswith(".mp4"):
            kind = "video"
        elif name.endswith(".parquet"):
            kind = "parquet"
        else:
            kind = None
        if kind is None or kind in entry_by_split_episode[item["split"]][episode_id]:
            raise ValueError(f"unexpected/duplicate media entry: {item['path']}")
        entry_by_split_episode[item["split"]][episode_id][kind] = item

    declared_episode_ids = {
        split: [int(item["episode_index"]) for item in plan["sampled_episode_ids"][split]]
        for split in ("train", "validation")
    }
    if set(declared_episode_ids["train"]) & set(declared_episode_ids["validation"]):
        raise ValueError("train/validation episode ID overlap")
    reports: dict[str, dict] = {}
    media_hashes: dict[str, set[str]] = {"train": set(), "validation": set()}
    normalized_content: dict[str, set[tuple[str, ...]]] = {
        "train": set(),
        "validation": set(),
    }

    if not selected_splits or any(
        split not in {"train", "validation"} for split in selected_splits
    ):
        raise ValueError("selected_splits must contain train and/or validation")
    for split in selected_splits:
        declared = declared_episode_ids[split]
        if limit_episodes is not None:
            declared = declared[:limit_episodes]
        split_entries = entry_by_split_episode[split]
        if limit_episodes is None and set(split_entries) != set(declared_episode_ids[split]):
            raise ValueError(f"{split} media inventory does not match frozen episode IDs")
        event_counts: Counter[str] = Counter()
        task_group_events: dict[str, Counter[str]] = defaultdict(Counter)
        target_counts: Counter[int] = Counter()
        sample_ids: set[str] = set()
        decoded_frames = 0
        checked_files = 0
        for episode_id in declared:
            if (
                episode_id not in split_entries
                or set(split_entries[episode_id]) != {"video", "parquet"}
            ):
                raise ValueError(f"incomplete {split} media pair for episode {episode_id}")
            row = episodes.get(episode_id)
            plan_row = next(
                item for item in plan["sampled_episode_ids"][split]
                if int(item["episode_index"]) == episode_id
            )
            if row is None or row.get("sid") != plan_row["sid"]:
                raise ValueError(f"episode metadata/group mismatch for {episode_id}")
            parent_group = plan_row["parent_group"]
            if row["sid"] not in parent_group.split("_"):
                raise ValueError(f"episode SID is outside parent group for {episode_id}")
            high_level_id = int(row["task_index"])
            if tasks.get(high_level_id) != row["high_level_instruction"]:
                raise ValueError(f"high-level task mapping mismatch for {episode_id}")
            options = resolve_episode_options(row["low_level_tasks"], subtasks)
            option_ids = tuple(option_id for option_id, _ in options)
            normalized_options = tuple(
                sorted(text.casefold().strip() for _, text in options)
            )
            normalized_content[split].add(normalized_options)

            pair = split_entries[episode_id]
            local_paths: dict[str, Path] = {}
            for kind, item in pair.items():
                path = attempt_root / "selected-media" / split / Path(item["path"]).name
                if not path.is_file() or path.stat().st_size != item["bytes"]:
                    raise ValueError(f"missing/size-mismatched media: {path}")
                actual_hash = sha256_file(path)
                if actual_hash != item["sha256"]:
                    raise ValueError(f"publisher SHA-256 mismatch: {path}")
                local_paths[kind] = path
                checked_files += 1
                if kind == "video":
                    media_hashes[split].add(actual_hash)

            table = pq.read_table(
                local_paths["parquet"], columns=["frame_index", "low_level_task_index"]
            )
            frame_indices = table["frame_index"].to_pylist()
            labels = table["low_level_task_index"].to_pylist()
            if not labels or len(frame_indices) != len(labels):
                raise ValueError(f"empty or unaligned frame/action rows in episode {episode_id}")
            if frame_indices != list(range(len(frame_indices))):
                raise ValueError(f"nonconsecutive frame indices in episode {episode_id}")
            if any(value not in option_ids for value in labels):
                raise ValueError(f"episode label is not one of its supplied options: {episode_id}")

            container = av.open(str(local_paths["video"]))
            try:
                streams = [stream for stream in container.streams if stream.type == "video"]
                if len(streams) != 1:
                    raise ValueError(f"expected one video stream in episode {episode_id}")
                stream = streams[0]
                count = sum(1 for _ in container.decode(stream))
                fps = float(stream.average_rate) if stream.average_rate else None
            finally:
                container.close()
            if count != len(labels) or fps is None or abs(fps - 10.0) > 0.01:
                raise ValueError(
                    f"video/Parquet alignment failure for {episode_id}: frames={count}, "
                    f"rows={len(labels)}, fps={fps}"
                )
            decoded_frames += count

            boundaries = [
                index for index in range(1, len(labels)) if labels[index] != labels[index - 1]
            ]
            eligible = 0
            for boundary in boundaries:
                window = transition_window_for_boundary(
                    labels,
                    boundary=boundary,
                    option_ids=option_ids,
                )
                if window is None:
                    continue
                sample_id = f"{episode_id}:{boundary}"
                if sample_id in sample_ids:
                    raise ValueError(f"duplicate event sample ID: {sample_id}")
                sample_ids.add(sample_id)
                eligible += 1
                target_counts[window.target_option_id] += 1
            event_counts["boundaries"] += len(boundaries)
            event_counts["eligible_windows"] += eligible
            event_counts["edge_skipped"] += len(boundaries) - eligible
            task_group_events[parent_group]["episodes"] += 1
            task_group_events[parent_group]["eligible_windows"] += eligible

        reports[split] = {
            "episodes": len(declared),
            "files_sha256_checked": checked_files,
            "unique_video_assets": len(media_hashes[split]),
            "decoded_video_frames": decoded_frames,
            "event_counts": dict(event_counts),
            "target_counts_by_low_level_id": dict(sorted(target_counts.items())),
            "unique_event_sample_ids": len(sample_ids),
            "task_groups": {
                group: dict(values) for group, values in sorted(task_group_events.items())
            },
        }

    compare_splits = set(reports) == {"train", "validation"}
    overlap = media_hashes["train"] & media_hashes["validation"] if compare_splits else set()
    if compare_splits and overlap:
        raise ValueError(f"train/validation video hash overlap: {len(overlap)}")
    content_overlap = (
        normalized_content["train"] & normalized_content["validation"]
        if compare_splits
        else set()
    )
    # Task option text may legitimately recur across source groups. Record rather
    # than reject it; media and task-group identity are the split boundaries.
    return {
        "format": "habit_transition_dataset_audit_v1",
        "dataset": plan["dataset"],
        "revision": plan["revision"],
        "seed": plan["seed"],
        "metadata_plan_sha256": sha256_file(plan_path),
        "limited_smoke": limit_episodes is not None,
        "limit_episodes_per_split": limit_episodes,
        "reserved_media_loaded": False,
        "split_reports": reports,
        "split_comparison_performed": compare_splits,
        "train_validation_video_hash_overlap": len(overlap) if compare_splits else None,
        "train_validation_option_text_set_overlap": (
            len(content_overlap) if compare_splits else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt-root", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit-episodes", type=int)
    parser.add_argument("--split", choices=("train", "validation", "all"), default="all")
    args = parser.parse_args()
    if args.limit_episodes is not None and args.limit_episodes < 1:
        parser.error("--limit-episodes must be positive")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing audit: {args.output}")
    report = audit(
        args.attempt_root,
        args.metadata_root,
        limit_episodes=args.limit_episodes,
        selected_splits=(
            ("train", "validation") if args.split == "all" else (args.split,)
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, separators=(",", ":")))


if __name__ == "__main__":
    main()
