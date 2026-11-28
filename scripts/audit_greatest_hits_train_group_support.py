"""Summarize label and source-group support from the pinned TRAIN inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

TASKS = ("material", "action", "reaction")
MISSING = "<MISSING>"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize_train_group_support(
    inventory: dict[str, Any], partition: dict[str, Any]
) -> dict[str, Any]:
    """Compute class counts and distinct TRAIN-parent support; ignore other splits."""
    train_ids = partition.get("train")
    parent_rows = inventory.get("parents")
    train_summary = inventory.get("splits", {}).get("train", {})
    if not isinstance(train_ids, list) or not train_ids:
        raise ValueError("partition must contain a non-empty TRAIN parent list")
    if len(train_ids) != len(set(train_ids)):
        raise ValueError("TRAIN parent IDs must be unique")
    if not isinstance(parent_rows, dict) or not isinstance(train_summary, dict):
        raise ValueError("inventory is missing parent or TRAIN summaries")
    if len(train_ids) != train_summary.get("parents"):
        raise ValueError("TRAIN parent count does not match the inventory summary")
    if len(train_summary.get("fields", [])) != len(TASKS):
        raise ValueError("TRAIN inventory must contain material, action and reaction")
    missing_histogram = train_summary.get("missing_fields", {})
    rows = train_summary.get("rows")
    if not isinstance(rows, int) or rows < 1:
        raise ValueError("TRAIN row count must be a positive integer")
    if sum(missing_histogram.values()) != rows:
        raise ValueError("missing_fields histogram does not sum to TRAIN rows")
    if train_summary.get("complete") != missing_histogram.get("0"):
        raise ValueError("complete count disagrees with zero-missing TRAIN rows")

    task_results: dict[str, Any] = {}
    for index, task in enumerate(TASKS):
        aggregate = train_summary["fields"][index]
        if not isinstance(aggregate, dict) or MISSING not in aggregate:
            raise ValueError(f"{task} aggregate must include an explicit missing count")
        class_counts = {label: count for label, count in aggregate.items() if label != MISSING}
        missing_count = aggregate[MISSING]
        if any(not isinstance(count, int) or count < 0 for count in aggregate.values()):
            raise ValueError(f"{task} counts must be non-negative integers")
        if sum(class_counts.values()) + missing_count != rows:
            raise ValueError(f"{task} labeled and missing counts do not sum to TRAIN rows")

        parent_support = {label: 0 for label in class_counts}
        reconciled_counts = {label: 0 for label in class_counts}
        for parent_id in train_ids:
            parent = parent_rows.get(parent_id)
            if not isinstance(parent, dict) or len(parent.get("fields", [])) != len(TASKS):
                raise ValueError(f"TRAIN parent lacks {task} inventory: {parent_id}")
            counts = parent["fields"][index]
            if not isinstance(counts, dict):
                raise ValueError(f"TRAIN parent has invalid {task} counts: {parent_id}")
            if set(counts) - set(class_counts) - {MISSING}:
                raise ValueError(f"TRAIN parent has an unlisted {task} label: {parent_id}")
            for label, count in counts.items():
                if not isinstance(count, int) or count < 0:
                    raise ValueError(f"TRAIN parent has invalid {task} count: {parent_id}")
                if label != MISSING:
                    reconciled_counts[label] += count
                    if count:
                        parent_support[label] += 1

        if reconciled_counts != class_counts:
            raise ValueError(f"per-parent {task} counts do not match TRAIN totals")
        labeled_count = sum(class_counts.values())
        task_results[task] = {
            "train_rows": rows,
            "labeled_rows": labeled_count,
            "missing_rows": missing_count,
            "missing_fraction": missing_count / rows,
            "class_counts": class_counts,
            "distinct_train_parent_support": parent_support,
            "classes_with_fewer_than_10_train_parents": sorted(
                label for label, count in parent_support.items() if count < 10
            ),
        }

    return {
        "status": "PASS_TRAIN_ONLY_LABEL_GROUP_SUPPORT",
        "source_commit": inventory.get("source_commit"),
        "partition_sha256": inventory.get("partition_sha256"),
        "train_partition_ids_sha256": hashlib.sha256(
            ("\n".join(sorted(train_ids)) + "\n").encode("utf-8")
        ).hexdigest(),
        "train_parent_count": len(train_ids),
        "validation_or_quarantine_labels_loaded": False,
        "sealed_audit_loaded": False,
        "tasks": task_results,
        "interpretation_limit": (
            "Group support is descriptive TRAIN coverage; it does not establish label correctness, "
            "visual observability, or generalization quality."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    receipt = json.loads(args.result.read_text(encoding="utf-8"))
    if receipt.get("status") != "PASS_FULL_OFFICIAL_TRAIN_INVENTORY":
        raise SystemExit("refusing an incomplete/unverified inventory receipt")
    if receipt.get("test_media_or_labels_opened") or receipt.get("sealed_audit_opened"):
        raise SystemExit("refusing an inventory whose test/sealed isolation failed")
    if receipt.get("source_commit") is None:
        raise SystemExit("inventory receipt does not pin its source commit")

    inventory_data = json.loads(args.inventory.read_text(encoding="utf-8"))
    partition_data = json.loads(args.partition.read_text(encoding="utf-8"))
    report = summarize_train_group_support(inventory_data, partition_data)
    report["input_sha256"] = {
        "inventory": sha256_file(args.inventory),
        "partition": sha256_file(args.partition),
        "result_receipt": sha256_file(args.result),
    }
    report["source_commit"] = receipt["source_commit"]
    report["partition_sha256"] = receipt["partition_sha256"]
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
