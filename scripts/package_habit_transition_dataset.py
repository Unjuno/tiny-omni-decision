"""Attach pinned question and candidate text to a frozen HABIT feature cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tiny_omni_decision.habit_decision import map_habit_choice_target

SOURCE_REVISION = "bae97d251907743f8fb6db7e70c61d361df4dc1e"
AUDIT_SHA256 = "9b7a9d6c0efbc0be055df55277ea107d9be0cd6a45dcbcb5e0571d0257796672"
EPISODES_SHA256 = "0aa53b5ab7c6c22ed8662bc77ce0ba3d5bcfa62bc885f1a928c586f76e95fff5"
SUBTASKS_SHA256 = "dcb0a7fe179c3f015bcef5bf3047de34d92dab2f0b8e2a55521b2fb0a21203a9"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def map_transition_text(
    record: dict[str, Any], episode: dict[str, Any], subtasks: dict[int, str]
) -> dict[str, Any]:
    """Map pinned text to a record, checking the already-frozen numeric target."""
    mapped = map_habit_choice_target(
        episode["low_level_tasks"], record["target_subtask_id"], subtasks
    )
    if mapped.target_index != record["target_option_index"]:
        raise ValueError(f"choice mapping differs from frozen target for {record['id']}")
    return {
        **record,
        "question": f"{episode['high_level_instruction']} What should the robot do next?",
        "options": list(mapped.options),
        "target_index": mapped.target_index,
        "collapsed_duplicate_count": mapped.collapsed_duplicate_count,
    }


def _read_selected_jsonl(path: Path, wanted: set[int], id_field: str) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    pattern = re.compile(r'^\s*\{"' + re.escape(id_field) + r'"\s*:\s*(\d+)\s*,')
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            match = pattern.match(line)
            if not match:
                raise ValueError(f"unexpected pinned JSONL shape in {path.name}")
            identifier = int(match.group(1))
            if identifier in wanted:
                row = json.loads(line)
                if row.get(id_field) != identifier:
                    raise ValueError("selected metadata ID parse mismatch")
                result[identifier] = row
    if set(result) != wanted:
        raise ValueError(f"selected metadata rows missing from {path.name}")
    return result


def package_feature_cache(input_dir: Path, metadata_dir: Path, output_dir: Path) -> dict[str, Any]:
    import numpy as np

    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    input_path = input_dir / "transition-inputs.json"
    feature_path = input_dir / "habit-transition-features.npz"
    episodes_path = metadata_dir / "full_meta_episodes.jsonl"
    subtasks_path = metadata_dir / "full_meta_subtasks.jsonl"
    if sha256_file(episodes_path) != EPISODES_SHA256:
        raise ValueError("pinned episode metadata SHA-256 mismatch")
    if sha256_file(subtasks_path) != SUBTASKS_SHA256:
        raise ValueError("pinned subtask metadata SHA-256 mismatch")
    inputs = json.loads(input_path.read_text(encoding="utf-8"))
    if (inputs.get("source_revision") != SOURCE_REVISION
            or inputs.get("audit_receipt_sha256") != AUDIT_SHA256
            or inputs.get("sealed_audit_accessed") is not False):
        raise ValueError("feature input manifest is not the frozen non-audit pilot")
    records = inputs["records"]
    episode_keys = {(row["split"], row["episode_index"]) for row in records}
    episode_ids = {episode_id for _, episode_id in episode_keys}
    episodes = _read_selected_jsonl(episodes_path, episode_ids, "episode_index")
    target_ids = {row["target_subtask_id"] for row in records}
    subtasks = _read_selected_jsonl(subtasks_path, target_ids, "task_index")
    text_records = []
    for record in records:
        episode = episodes[record["episode_index"]]
        if episode["sid"] != record["sid"] or episode["task_index"] != record["task_index"]:
            raise ValueError(f"episode identity changed for {record['id']}")
        text_records.append(map_transition_text(record, episode, {
            target_id: row["task"] for target_id, row in subtasks.items()
        }))

    with np.load(feature_path, allow_pickle=False) as arrays:
        if arrays["ids"].tolist() != [row["id"] for row in records]:
            raise ValueError("feature array ID order differs from frozen input manifest")
        if arrays["target_index"].tolist() != [row["target_option_index"] for row in records]:
            raise ValueError("feature target array differs from frozen input manifest")
        feature_summary = {
            "video_features_shape": list(arrays["video_features"].shape),
            "sensor_sequence_shape": list(arrays["sensor_sequence"].shape),
        }
    output_dir.mkdir(parents=True)
    shutil.copy2(feature_path, output_dir / feature_path.name)
    (output_dir / "transition-examples.json").write_text(
        json.dumps({
            "source_revision": SOURCE_REVISION,
            "audit_receipt_sha256": AUDIT_SHA256,
            "episode_metadata_sha256": EPISODES_SHA256,
            "subtask_metadata_sha256": SUBTASKS_SHA256,
            "sealed_audit_accessed": False,
            "action_columns_read": False,
            "records": text_records,
        }, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    report = {
        "status": "complete_text_aligned_frozen_feature_package",
        "records": len(text_records),
        "train_records": sum(row["split"] == "train" for row in text_records),
        "validation_records": sum(row["split"] == "validation" for row in text_records),
        "train_task_groups": len({row["sid"] for row in text_records if row["split"] == "train"}),
        "validation_task_groups": len({
            row["sid"] for row in text_records if row["split"] == "validation"
        }),
        **feature_summary,
        "feature_cache_sha256": sha256_file(output_dir / feature_path.name),
        "transition_examples_sha256": sha256_file(output_dir / "transition-examples.json"),
        "feature_input_manifest_sha256": sha256_file(input_path),
        "sealed_audit_accessed": False,
        "action_columns_read": False,
        "training_performed": False,
    }
    (output_dir / "package-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = package_feature_cache(
        args.input_dir.resolve(strict=True),
        args.metadata_dir.resolve(strict=True),
        args.output_dir.resolve(),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
