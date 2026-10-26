"""Audit selected full-DROID shards and overlaps with the 100-episode debug set."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from .audit_droid100_groups import (
        episode_example_context,
        episode_example_instructions,
        iter_tfrecord,
    )
    from .download_droid_selected_shards import validate_manifest, verify_metadata
except ImportError:
    from audit_droid100_groups import (
        episode_example_context,
        episode_example_instructions,
        iter_tfrecord,
    )
    from download_droid_selected_shards import validate_manifest, verify_metadata


SOURCE_FILE_KEY = "episode_metadata/file_path"
RECORDING_GROUP_KEY = "episode_metadata/recording_folderpath"


def normalize_instruction(text: str) -> str:
    return re.sub(r"[^\w]+", " ", text.casefold(), flags=re.UNICODE).strip()


def audit_selected_shards(data_dir: Path, manifest: dict, debug_audit: dict) -> dict[str, Any]:
    validate_manifest(manifest)
    verify_metadata(data_dir.parents[2], manifest)

    episodes = []
    instruction_counts: Counter[str] = Counter()
    for shard_row in manifest["selected_shards"]:
        shard_path = data_dir / Path(shard_row["object_name"]).name
        rows = []
        for record in iter_tfrecord(shard_path):
            row = episode_example_context(record)
            instructions = episode_example_instructions(record)
            row["shard_index"] = int(shard_row["index"])
            normalized_instructions = {
                normalize_instruction(value) for value in instructions
            }
            row["normalized_instructions"] = sorted(
                value for value in normalized_instructions if value
            )
            instruction_counts.update(row["normalized_instructions"])
            rows.append(row)
        if len(rows) != int(shard_row["episodes"]):
            raise ValueError(
                f"shard {shard_row['index']} episode count mismatch: "
                f"expected {shard_row['episodes']}, read {len(rows)}"
            )
        episodes.extend(rows)

    if len(episodes) != int(manifest["expected_episode_count"]):
        raise ValueError("selected shard total episode count mismatch")
    files = [row[SOURCE_FILE_KEY] for row in episodes]
    groups = [row[RECORDING_GROUP_KEY] for row in episodes]
    if len(files) != len(set(files)):
        raise ValueError("duplicate source file_path within selected full-DROID subset")

    debug_episodes = debug_audit["episodes"]
    debug_files = {row[SOURCE_FILE_KEY] for row in debug_episodes}
    debug_groups = {row[RECORDING_GROUP_KEY] for row in debug_episodes}
    debug_instructions = {
        normalize_instruction(text)
        for text in debug_audit.get("instruction_episode_frequency", {})
        if normalize_instruction(text)
    }
    group_counts = Counter(groups)
    selected_instructions = set(instruction_counts)
    return {
        "dataset_id": manifest["dataset_id"],
        "dataset_version": manifest["dataset_version"],
        "selection_seed": manifest["selection_seed"],
        "selection_manifest_sha256": manifest.get("manifest_sha256"),
        "episode_count": len(episodes),
        "unique_source_files": len(set(files)),
        "unique_recording_groups": len(group_counts),
        "duplicate_recording_groups": len(episodes) - len(group_counts),
        "unique_normalized_instructions": len(selected_instructions),
        "normalized_instruction_repeats": sum(
            max(0, count - 1) for count in instruction_counts.values()
        ),
        "success_failure_source_folders": {
            "success": sum("/success/" in value for value in files),
            "failure": sum("/failure/" in value for value in files),
        },
        "overlap_with_droid100_debug": {
            "source_files": len(set(files) & debug_files),
            "recording_groups": len(set(groups) & debug_groups),
            "normalized_instructions": len(selected_instructions & debug_instructions),
        },
        "episodes_per_shard": {
            str(row["index"]): sum(item["shard_index"] == int(row["index"]) for item in episodes)
            for row in manifest["selected_shards"]
        },
        "split_warning": (
            "All selected episodes come from the official train split; "
            "this is development data."
        ),
        "episodes": episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--debug-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    manifest["manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    debug_audit = json.loads(args.debug_audit.read_text(encoding="utf-8"))
    result = audit_selected_shards(args.data_dir, manifest, debug_audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "episodes"}, indent=2))


if __name__ == "__main__":
    main()
