"""Audit pinned HABIT metadata and create task-group split receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from tiny_omni_decision.habit_preflight import build_habit_grouped_split


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--task-details", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output path must be new; preserve prior receipts")

    episodes = read_jsonl(args.episodes)
    report = build_habit_grouped_split(episodes, args.task_details, seed=args.seed)
    task_details = sorted(args.task_details.glob("*.md"))
    report.update(
        {
            "dataset": "configinc/HABIT",
            "source_revision": args.source_revision,
            "episodes_metadata_sha256": sha256_file(args.episodes),
            "task_details_sha256": hashlib.sha256(
                "\n".join(
                    f"{path.name}:{sha256_file(path)}" for path in task_details
                ).encode("utf-8")
            ).hexdigest(),
            "train_episode_ids_available": True,
            "validation_episode_ids_available": True,
            "sealed_audit_ids_loaded_or_written": False,
        }
    )
    args.output.mkdir(parents=True)
    output_file = args.output / "habit-metadata-preflight.json"
    output_file.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        key: value
        for key, value in report.items()
        if key not in {"train_episode_ids", "validation_episode_ids"}
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {output_file}")


if __name__ == "__main__":
    main()
