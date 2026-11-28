"""Build a fixed object-group split from the metadata-only X-Capture audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

PROTOCOL = "xcapture-cross-sensory-object-retrieval-split-v1"
SOURCE_REPORT = Path("manifests/pretrained_reuse/xcapture-source-screen-20261128.json")
DEFAULT_OUTPUT = Path(
    "manifests/pretrained_reuse/xcapture-object-split-seed17.json"
)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_group_split(
    object_ids: list[str], *, seed: int = 17, sizes: dict[str, int] | None = None
) -> dict[str, Any]:
    """Deterministically assign whole object identities to fixed partitions."""
    counts = sizes or {"train": 400, "validation": 100, "evaluation": 100}
    if not object_ids or any(not isinstance(item, str) or not item for item in object_ids):
        raise ValueError("object IDs must be nonempty strings")
    if len(set(object_ids)) != len(object_ids):
        raise ValueError("object IDs must be unique")
    if any(count < 0 for count in counts.values()) or sum(counts.values()) != len(object_ids):
        raise ValueError("partition sizes must be nonnegative and sum to all object IDs")

    ranked = sorted(
        object_ids,
        key=lambda item: (_sha256(f"{PROTOCOL}|seed={seed}|object_id={item}"), item),
    )
    assignments: dict[str, list[str]] = {}
    cursor = 0
    for name, count in counts.items():
        assignments[name] = sorted(ranked[cursor : cursor + count])
        cursor += count

    seen: set[str] = set()
    hashes: dict[str, str] = {}
    for name, ids in assignments.items():
        if seen.intersection(ids):
            raise ValueError("object identity overlaps partitions")
        seen.update(ids)
        hashes[name] = _sha256("\n".join(ids))
    if seen != set(object_ids):
        raise ValueError("split does not cover every source object")

    return {
        "protocol": PROTOCOL,
        "seed": seed,
        "group_unit": "source object directory ID; all six capture points stay together",
        "assignment_rule": (
            "rank IDs by SHA-256(protocol|seed|object_id), then allocate "
            "in declared partition order"
        ),
        "partition_counts": {name: len(ids) for name, ids in assignments.items()},
        "partition_id_sha256": hashes,
        "partitions": assignments,
    }


def build_source_split(source: dict[str, Any]) -> dict[str, Any]:
    """Bind the split to the audited archive revision and object inventory."""
    if source.get("status") != "ALLOW_TASK_SCOPED_CROSS_SENSORY_RETRIEVAL":
        raise ValueError("source license/task gate is not cleared")
    archive = source.get("archive", {})
    if archive.get("lfs_sha256") != (
        "71855c2d9cdfb6d54ed379ff4059b6c370e6f7ef040830b0ac19063159c477c2"
    ):
        raise ValueError("unexpected X-Capture archive hash")
    task = source.get("task_evidence", {})
    if task.get("object_id_range") != [0, 599] or task.get("objects") != 600:
        raise ValueError("audited object inventory is missing or changed")
    object_ids = [f"{value:05d}" for value in range(600)]
    split = build_group_split(object_ids)
    return {
        "schema_version": 1,
        "dataset_id": source["candidate"],
        "source_revision": source["source"]["revision"],
        "archive_sha256": archive["lfs_sha256"],
        "central_directory_sha256": archive["central_directory_sha256"],
        "source_object_inventory": {
            "observed_count": len(object_ids),
            "first_id": object_ids[0],
            "last_id": object_ids[-1],
            "ids_sha256": _sha256("\n".join(object_ids)),
            "index_verification": (
                "600 contiguous numeric object IDs were parsed from the pinned "
                "ZIP central directory"
            ),
        },
        "task": "cross-sensory object-identity retrieval only; no semantic attribute labels",
        "media_policy": (
            "split metadata only; evaluation media/features remain unopened "
            "until candidate configuration is frozen"
        ),
        **split,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-report", type=Path, default=SOURCE_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    source = json.loads(args.source_report.read_text(encoding="utf-8"))
    payload = json.dumps(build_source_split(source), indent=2) + "\n"
    if args.output.exists():
        if args.output.read_text(encoding="utf-8") != payload:
            raise FileExistsError(f"refusing to overwrite different split: {args.output}")
        print(f"already current: {args.output}")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8", newline="\n")
    print(args.output)


if __name__ == "__main__":
    main()
