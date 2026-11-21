"""Leakage-safe task-group splitting for the public HABIT metadata."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROLES = {"collaborator", "coworker", "supervisor"}


def normalize_instruction(value: str) -> str:
    """Normalize instruction text only for exact-duplicate grouping."""
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _detail_ids(path: Path) -> list[str]:
    return [part for part in path.stem.split("_") if part]


def _detail_role(path: Path) -> str | None:
    text = path.read_text(encoding="utf-8")
    match = re.search(
        r"^##\s+(Collaborator|Colaborator|Coworker|Supervisor)\s*$",
        text,
        re.M | re.I,
    )
    if not match:
        return None
    role = match.group(1).casefold()
    return "collaborator" if role == "colaborator" else role


def _sha256_ids(ids: list[int]) -> str:
    payload = "\n".join(str(value) for value in sorted(ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_habit_grouped_split(
    episodes: list[dict[str, Any]],
    task_details_dir: str | Path,
    *,
    seed: int = 17,
) -> dict[str, Any]:
    """Assign complete task groups to train/validation/sealed audit.

    Groups connect task-index aliases, exact normalized high-level instructions,
    and scenario IDs listed together in an official task-details filename.
    Groups with conflicting roles are train-only. Episodes whose scenario has
    no official task-details entry are quarantined. The function returns only
    counts and hashes for the sealed audit IDs, not the IDs themselves.
    """
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if not episodes:
        raise ValueError("episode metadata is empty")

    by_sid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    ids: list[int] = []
    for row in episodes:
        sid = str(row.get("sid", "")).strip()
        instruction = str(row.get("high_level_instruction", "")).strip()
        episode_id = row.get("episode_index")
        if not sid or not instruction or not isinstance(episode_id, int):
            raise ValueError(
                "episode rows require sid, high_level_instruction, and integer episode_index"
            )
        by_sid[sid].append(row)
        ids.append(episode_id)
    if len(ids) != len(set(ids)):
        raise ValueError("episode_index values must be unique")

    parent = {sid: sid for sid in by_sid}

    def find(sid: str) -> str:
        while parent[sid] != sid:
            parent[sid] = parent[parent[sid]]
            sid = parent[sid]
        return sid

    def union(sids: list[str]) -> None:
        present = [sid for sid in sids if sid in parent]
        if present:
            root = find(present[0])
            for sid in present[1:]:
                other = find(sid)
                if root != other:
                    parent[other] = root

    details = sorted(Path(task_details_dir).glob("*.md"))
    if not details:
        raise ValueError("no official task-details markdown files found")
    role_by_sid: dict[str, set[str]] = defaultdict(set)
    detail_ids_seen: set[str] = set()
    for path in details:
        scenario_ids = _detail_ids(path)
        detail_ids_seen.update(scenario_ids)
        union(scenario_ids)
        role = _detail_role(path)
        if role in ROLES:
            for sid in scenario_ids:
                if sid in parent:
                    role_by_sid[sid].add(role)

    by_instruction: dict[str, list[str]] = defaultdict(list)
    by_task_index: dict[str, list[str]] = defaultdict(list)
    for sid, rows in by_sid.items():
        for row in rows:
            by_instruction[normalize_instruction(row["high_level_instruction"])].append(sid)
            by_task_index[str(row.get("task_index", ""))].append(sid)
    for sids in by_instruction.values():
        union(sids)
    for task_index, sids in by_task_index.items():
        if task_index:
            union(sids)

    groups: dict[str, set[str]] = defaultdict(set)
    for sid in by_sid:
        groups[find(sid)].add(sid)

    group_rows: dict[str, list[dict[str, Any]]] = {}
    group_role: dict[str, str] = {}
    group_sids: dict[str, list[str]] = {}
    for sids in groups.values():
        rows = [row for sid in sorted(sids) for row in by_sid[sid]]
        roles = set().union(*(role_by_sid.get(sid, set()) for sid in sids))
        role = next(iter(roles)) if len(roles) == 1 else "mixed" if roles else "unknown"
        group_key = hashlib.sha256("\n".join(sorted(sids)).encode("utf-8")).hexdigest()[:16]
        group_rows[group_key] = rows
        group_role[group_key] = role
        group_sids[group_key] = sorted(sids)

    assignments: dict[str, str] = {}
    pure_by_role: dict[str, list[str]] = defaultdict(list)
    for group_key, role in group_role.items():
        if role in ROLES:
            pure_by_role[role].append(group_key)
        elif role == "mixed":
            assignments[group_key] = "train"
        else:
            assignments[group_key] = "quarantine"

    for role, role_groups in sorted(pure_by_role.items()):
        ordered = sorted(
            role_groups,
            key=lambda key: hashlib.sha256(f"{seed}:{role}:{key}".encode()).hexdigest(),
        )
        if len(ordered) < 3:
            for group_key in ordered:
                assignments[group_key] = "train"
            continue
        validation_n = max(1, round(len(ordered) * 0.15))
        audit_n = max(1, round(len(ordered) * 0.15))
        if validation_n + audit_n >= len(ordered):
            validation_n = audit_n = 1
        for index, group_key in enumerate(ordered):
            if index < len(ordered) - validation_n - audit_n:
                assignments[group_key] = "train"
            elif index < len(ordered) - audit_n:
                assignments[group_key] = "validation"
            else:
                assignments[group_key] = "sealed_audit"

    split_names = ("train", "validation", "sealed_audit", "quarantine")
    split_ids: dict[str, list[int]] = {name: [] for name in split_names}
    for group_key, rows in group_rows.items():
        split_ids[assignments[group_key]].extend(int(row["episode_index"]) for row in rows)
    all_split_ids = [episode_id for values in split_ids.values() for episode_id in values]
    if len(all_split_ids) != len(ids) or len(all_split_ids) != len(set(all_split_ids)):
        raise ValueError("split assignment lost or duplicated episode IDs")

    group_counts = Counter(assignments.values())
    episode_counts = {name: len(values) for name, values in split_ids.items()}
    role_group_counts = {
        split: dict(
            Counter(
                group_role[key]
                for key, value in assignments.items()
                if value == split
            )
        )
        for split in split_names
    }
    return {
        "seed": seed,
        "grouping": [
            "official task-details SID aliases",
            "task_index",
            "exact normalized high-level instruction",
        ],
        "task_detail_file_count": len(details),
        "metadata_sid_count": len(by_sid),
        "task_detail_sid_count": len(detail_ids_seen),
        "sid_without_task_details": sorted(set(by_sid) - detail_ids_seen),
        "group_count": len(groups),
        "group_counts_by_split": dict(group_counts),
        "episode_counts_by_split": episode_counts,
        "group_counts_by_split_and_role": role_group_counts,
        "episode_id_sha256_by_split": {
            name: _sha256_ids(values) for name, values in split_ids.items()
        },
        "train_episode_ids": sorted(split_ids["train"]),
        "validation_episode_ids": sorted(split_ids["validation"]),
        "sealed_audit_episode_ids": None,
        "sealed_audit_episode_count": episode_counts["sealed_audit"],
        "sealed_audit_episode_id_sha256": _sha256_ids(split_ids["sealed_audit"]),
        "quarantined_episode_count": episode_counts["quarantine"],
        "quarantined_episode_id_sha256": _sha256_ids(split_ids["quarantine"]),
    }
