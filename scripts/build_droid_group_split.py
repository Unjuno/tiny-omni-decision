"""Create a deterministic recording- and content-safe development split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from .audit_droid_selected_shards import normalize_instruction
except ImportError:
    from audit_droid_selected_shards import normalize_instruction

SOURCE_FILE_KEY = "episode_metadata/file_path"
RECORDING_GROUP_KEY = "episode_metadata/recording_folderpath"


class DisjointSet:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        if self.rank[root_left] == self.rank[root_right]:
            self.rank[root_left] += 1


def _canonical_path(value: str) -> str:
    return "/".join(part for part in value.replace("\\", "/").casefold().split("/") if part)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_group_split(
    audit: dict[str, Any],
    debug_audit: dict[str, Any],
    *,
    validation_groups: int = 200,
    minimum_train_groups: int = 200,
    seed: int = 17,
) -> dict[str, Any]:
    episodes = audit.get("episodes", [])
    debug_files = {
        _canonical_path(row[SOURCE_FILE_KEY]) for row in debug_audit.get("episodes", [])
    }
    debug_groups = {
        _canonical_path(row[RECORDING_GROUP_KEY]) for row in debug_audit.get("episodes", [])
    }
    retained = [
        row
        for row in episodes
        if _canonical_path(row[SOURCE_FILE_KEY]) not in debug_files
        and _canonical_path(row[RECORDING_GROUP_KEY]) not in debug_groups
    ]
    if not retained:
        raise ValueError("no episodes remain after excluding DROID-100 debug overlap")

    sets = DisjointSet(len(retained))
    first_by_key: dict[tuple[str, str], int] = {}
    for index, row in enumerate(retained):
        keys = [("source", _canonical_path(row[SOURCE_FILE_KEY]))]
        keys.append(("recording", _canonical_path(row[RECORDING_GROUP_KEY])))
        keys.extend(
            ("instruction", normalize_instruction(text))
            for text in row.get("normalized_instructions", [])
            if normalize_instruction(text)
        )
        for key in keys:
            if key in first_by_key:
                sets.union(index, first_by_key[key])
            else:
                first_by_key[key] = index

    components: dict[int, list[int]] = {}
    for index in range(len(retained)):
        components.setdefault(sets.find(index), []).append(index)

    component_rows = []
    for indices in components.values():
        group_ids = sorted(
            {
                _sha(_canonical_path(retained[index][RECORDING_GROUP_KEY]))
                for index in indices
            }
        )
        component_key = _sha(f"droid-r2-seed{seed}:" + "|".join(group_ids))
        component_rows.append((component_key, indices, group_ids))
    component_rows.sort(key=lambda row: row[0])

    total_groups = len(
        {_canonical_path(row[RECORDING_GROUP_KEY]) for row in retained}
    )
    if total_groups < validation_groups + minimum_train_groups:
        raise ValueError(
            f"only {total_groups} independent recording groups remain; "
            f"need at least {validation_groups + minimum_train_groups}"
        )

    validation_indices: set[int] = set()
    validation_group_ids: set[str] = set()
    for _, indices, group_ids in component_rows:
        if len(validation_group_ids) >= validation_groups:
            break
        validation_indices.update(indices)
        validation_group_ids.update(group_ids)

    train_indices = set(range(len(retained))) - validation_indices
    train_group_ids = {
        _sha(_canonical_path(retained[index][RECORDING_GROUP_KEY])) for index in train_indices
    }
    if len(validation_group_ids) < validation_groups or len(train_group_ids) < minimum_train_groups:
        raise ValueError("group/text components cannot satisfy the predeclared split floor")

    train_rows = [retained[index] for index in sorted(train_indices)]
    validation_rows = [retained[index] for index in sorted(validation_indices)]
    shared_instructions = {
        instruction
        for row in train_rows
        for instruction in row.get("normalized_instructions", [])
    } & {
        instruction
        for row in validation_rows
        for instruction in row.get("normalized_instructions", [])
    }
    shared_groups = {
        _canonical_path(row[RECORDING_GROUP_KEY]) for row in train_rows
    } & {_canonical_path(row[RECORDING_GROUP_KEY]) for row in validation_rows}
    shared_sources = {
        _canonical_path(row[SOURCE_FILE_KEY]) for row in train_rows
    } & {_canonical_path(row[SOURCE_FILE_KEY]) for row in validation_rows}
    if shared_instructions or shared_groups or shared_sources:
        raise ValueError("split has source, recording, or normalized instruction overlap")

    return {
        "schema_version": 1,
        "dataset_id": audit["dataset_id"],
        "dataset_version": audit["dataset_version"],
        "selection_seed": seed,
        "selection_manifest_sha256": audit.get("selection_manifest_sha256"),
        "source_partition": "official train split; development-only",
        "split_method": (
            "union source/recording/normalized instruction; "
            "stable SHA-256 component order"
        ),
        "validation_target_groups": validation_groups,
        "minimum_train_groups": minimum_train_groups,
        "excluded_debug_overlap_episodes": len(episodes) - len(retained),
        "train": {
            "episode_count": len(train_rows),
            "recording_groups": len(train_group_ids),
            "episode_ids": sorted(
                _sha(_canonical_path(row[SOURCE_FILE_KEY])) for row in train_rows
            ),
        },
        "validation": {
            "episode_count": len(validation_rows),
            "recording_groups": len(validation_group_ids),
            "episode_ids": sorted(
                _sha(_canonical_path(row[SOURCE_FILE_KEY])) for row in validation_rows
            ),
        },
        "overlap_counts": {
            "source_files": 0,
            "recording_groups": 0,
            "normalized_instruction_content": 0,
        },
        "sealed_audit": None,
        "warning": (
            "This split is development validation, not a sealed audit "
            "or official DROID test."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--debug-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-groups", type=int, default=200)
    parser.add_argument("--minimum-train-groups", type=int, default=200)
    args = parser.parse_args()
    audit = json.loads(args.audit_json.read_text(encoding="utf-8"))
    debug_audit = json.loads(args.debug_audit.read_text(encoding="utf-8"))
    result = build_group_split(
        audit,
        debug_audit,
        validation_groups=args.validation_groups,
        minimum_train_groups=args.minimum_train_groups,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    summary = {
        key: value for key, value in result.items() if key not in {"train", "validation"}
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
