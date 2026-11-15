"""Audit pinned HandsOn split/session counts without persisting manifest rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from typing import BinaryIO
from urllib.request import Request, urlopen

DATASET_REVISION = "7e19e2675464ebff9742923b20267e5af5ad645e"
SOURCE_MANIFEST_URL = (
    "https://huggingface.co/datasets/Cyan666/HandsOn-Datasets/resolve/"
    f"{DATASET_REVISION}/source_manifest.jsonl"
)
EXPECTED_BYTES = 42_610_607
EXPECTED_SHA256 = "6cced3356edb916038d160c3262238d0bc606ef678885b510db20b9dea69a942"
USER_AGENT = "tiny-omni-decision-readonly-metadata-audit/1.0"


def audit_stream(
    stream: BinaryIO,
    *,
    expected_bytes: int = EXPECTED_BYTES,
    expected_sha256: str = EXPECTED_SHA256,
) -> dict[str, object]:
    """Return aggregates for split, source task, and collection session only."""
    digest = hashlib.sha256()
    byte_count = 0
    episode_counts: Counter[tuple[str, str]] = Counter()
    sessions_by_split: dict[str, set[str]] = {"train": set(), "validation": set()}
    sessions_by_task: dict[tuple[str, str], set[str]] = defaultdict(set)

    for line_number, line in enumerate(stream, start=1):
        digest.update(line)
        byte_count += len(line)
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid JSONL row at line {line_number}") from error
        split = row.get("split")
        task = row.get("source_task")
        session = row.get("collection_session_id")
        if split not in sessions_by_split:
            raise ValueError(f"unexpected or missing split at line {line_number}")
        if not isinstance(task, str) or not task:
            raise ValueError(f"missing source_task at line {line_number}")
        if not isinstance(session, str) or not session:
            raise ValueError(f"missing collection_session_id at line {line_number}")
        episode_counts[(task, split)] += 1
        sessions_by_split[split].add(session)
        sessions_by_task[(task, split)].add(session)

    actual_sha256 = digest.hexdigest()
    if byte_count != expected_bytes:
        raise ValueError(f"manifest size mismatch: expected {expected_bytes}, got {byte_count}")
    if actual_sha256 != expected_sha256:
        raise ValueError(f"manifest SHA-256 mismatch: {actual_sha256}")

    overlap = sessions_by_split["train"] & sessions_by_split["validation"]
    if overlap:
        raise ValueError(f"train/validation session overlap: {len(overlap)} groups")

    tasks = sorted({task for task, _split in episode_counts})
    validation_group_counts = [
        len(sessions_by_task[(task, "validation")]) for task in tasks
    ]
    per_task = {
        task: {
            split: {
                "episodes": episode_counts[(task, split)],
                "sessions": len(sessions_by_task[(task, split)]),
            }
            for split in ("train", "validation")
            if episode_counts[(task, split)]
        }
        for task in tasks
    }
    return {
        "dataset_revision": DATASET_REVISION,
        "manifest_bytes": byte_count,
        "manifest_sha256": actual_sha256,
        "source_task_count": len(tasks),
        "train_episodes": sum(
            count
            for (_task, split), count in episode_counts.items()
            if split == "train"
        ),
        "validation_episodes": sum(
            count for (_task, split), count in episode_counts.items() if split == "validation"
        ),
        "train_sessions": len(sessions_by_split["train"]),
        "validation_sessions": len(sessions_by_split["validation"]),
        "cross_split_session_overlap": 0,
        "per_task_validation_sessions": {
            "minimum": min(validation_group_counts),
            "median": statistics.median(validation_group_counts),
            "maximum": max(validation_group_counts),
        },
        "per_task": per_task,
    }


def fetch_and_audit() -> dict[str, object]:
    request = Request(SOURCE_MANIFEST_URL, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=60) as response:
        return audit_stream(response)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--confirm-network",
        action="store_true",
        help="fetch only the pinned 42.6 MB source manifest; no media or labels are requested",
    )
    args = parser.parse_args()
    if not args.confirm_network:
        parser.error("pass --confirm-network to fetch the pinned metadata manifest")
    try:
        result = fetch_and_audit()
    except (OSError, ValueError) as error:
        print(f"HandsOn source-manifest audit failed closed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
