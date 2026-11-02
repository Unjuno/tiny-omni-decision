"""Summarize prior-unexposed AVATAR TRAIN annotation support by scenario."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from .audit_avatar_multientity_exposure import (
        _sha256_file,
        _source_group_from_annotation,
        collect_prior_group_usage,
    )
except ImportError:  # Direct execution as ``python scripts/audit_avatar_task_support.py``.
    from audit_avatar_multientity_exposure import (
        _sha256_file,
        _source_group_from_annotation,
        collect_prior_group_usage,
    )

TASKS = {"Single-Sound", "Mixed-Sound", "Multi-Entity", "Off-Screen"}
DATASET_REVISION = "a7f0146ff704799d044133e61137836cacc1c233"


def _group_split(groups: set[str], task: str, seed: int, validation_fraction: float):
    ranked = sorted(
        groups,
        key=lambda group: hashlib.sha256(
            f"avatar-task-support-v1:{seed}:{task}:{group}".encode()
        ).hexdigest(),
    )
    validation_count = round(len(ranked) * validation_fraction)
    validation = set(ranked[:validation_count])
    return set(ranked[validation_count:]), validation


def audit_task_support(
    train_annotations: Path,
    *,
    previously_exposed_groups: set[str],
    seed: int = 17,
    validation_fraction: float = 0.2,
) -> dict[str, Any]:
    """Read TRAIN JSON annotations only and return aggregate task support."""
    root = Path(train_annotations).resolve()
    if root.name.lower() != "train" or any(part.lower() == "audit" for part in root.parts):
        raise ValueError("train_annotations must be the isolated TRAIN directory")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")

    state: dict[str, dict[str, Any]] = {
        task: {
            "groups": set(),
            "clips": set(),
            "categories": defaultdict(set),
            "instances": 0,
            "frames": 0,
            "group_instances": Counter(),
            "group_frames": Counter(),
            "group_frame_instance_counts": defaultdict(Counter),
            "group_multi_category_frames": Counter(),
        }
        for task in TASKS
    }
    tree_hash = hashlib.sha256()
    json_count = 0

    for path in sorted(root.rglob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        group = _source_group_from_annotation(path, row)
        clip = row["video_id"]
        annotations = row.get("annotations")
        if not isinstance(annotations, list) or not annotations:
            raise ValueError(f"missing annotations in {path}")
        relative = path.relative_to(root).as_posix()
        tree_hash.update(relative.encode("utf-8") + b"\0")
        tree_hash.update(_sha256_file(path).encode("ascii") + b"\n")
        json_count += 1

        by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for annotation in annotations:
            if not isinstance(annotation, dict):
                raise ValueError(f"invalid annotation in {path}")
            task = annotation.get("task")
            category = annotation.get("audio_visual_category")
            if task not in TASKS or not isinstance(category, str) or not category:
                raise ValueError(f"unknown task or missing category in {path}")
            by_task[task].append(annotation)

        for task, items in by_task.items():
            item = state[task]
            item["groups"].add(group)
            item["clips"].add(clip)
            item["instances"] += len(items)
            item["frames"] += 1
            item["group_instances"][group] += len(items)
            item["group_frames"][group] += 1
            item["group_frame_instance_counts"][group][len(items)] += 1
            frame_categories = {entry["audio_visual_category"] for entry in items}
            item["categories"][group].update(frame_categories)
            item["group_multi_category_frames"][group] += len(frame_categories) > 1

    tasks: dict[str, Any] = {}
    for task, item in sorted(state.items()):
        groups = item["groups"]
        unexposed = groups - previously_exposed_groups
        fit, validation = _group_split(unexposed, task, seed, validation_fraction)
        category_support: dict[str, dict[str, int]] = defaultdict(
            lambda: {"train_groups": 0, "validation_groups": 0}
        )
        unexposed_frame_instance_counts: Counter[int] = Counter()
        for group in unexposed:
            unexposed_frame_instance_counts.update(
                item["group_frame_instance_counts"][group]
            )
        for group in unexposed:
            split_key = "validation_groups" if group in validation else "train_groups"
            for category in item["categories"].get(group, set()):
                category_support[category][split_key] += 1
        shared_categories = [
            support
            for support in category_support.values()
            if support["train_groups"] and support["validation_groups"]
        ]
        tasks[task] = {
            "annotation_instance_count": item["instances"],
            "annotated_frame_count": item["frames"],
            "annotated_clip_count": len(item["clips"]),
            "source_group_count": len(groups),
            "prior_exposed_source_group_count": len(groups & previously_exposed_groups),
            "previously_unexposed_source_group_count": len(unexposed),
            "unexposed_annotation_instance_count": sum(
                item["group_instances"][group] for group in unexposed
            ),
            "unexposed_annotated_frame_count": sum(
                item["group_frames"][group] for group in unexposed
            ),
            "unexposed_group_set_sha256": hashlib.sha256(
                "\n".join(sorted(unexposed)).encode("utf-8")
            ).hexdigest(),
            "illustrative_group_split_not_frozen": {
                "seed": seed,
                "validation_fraction": validation_fraction,
                "train_group_count": len(fit),
                "validation_group_count": len(validation),
                "overlap_count": len(fit & validation),
                "frame_counts_by_instance_count": dict(
                    sorted(unexposed_frame_instance_counts.items())
                ),
                "frames_with_multiple_categories": sum(
                    item["group_multi_category_frames"][group] for group in unexposed
                ),
                "category_count": len(category_support),
                "category_count_present_in_both_splits": len(shared_categories),
                "categories_with_at_least_four_groups_in_each_split": sum(
                    support["train_groups"] >= 4 and support["validation_groups"] >= 4
                    for support in shared_categories
                ),
                "minimum_validation_groups_for_shared_category": min(
                    (support["validation_groups"] for support in shared_categories),
                    default=0,
                ),
            },
        }

    return {
        "format": "avatar_train_task_support_v1",
        "dataset_revision": DATASET_REVISION,
        "train_annotation_json_count": json_count,
        "train_annotations_sha256": tree_hash.hexdigest(),
        "prior_exposed_group_set_sha256": hashlib.sha256(
            "\n".join(sorted(previously_exposed_groups)).encode("utf-8")
        ).hexdigest(),
        "source_group_ids_emitted": False,
        "official_validation_or_sealed_audit_accessed": False,
        "media_bytes_read": False,
        "tasks": tasks,
        "interpretation": (
            "Previously unexposed means absent from inventoried prior fit/evaluation groups. "
            "This is TRAIN annotation support only, not an independent quality result."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-annotations", required=True, type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    fit_groups, evaluation_groups, _, _ = collect_prior_group_usage(args.artifact_root)
    groups = fit_groups | evaluation_groups
    report = audit_task_support(args.train_annotations, previously_exposed_groups=groups)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    summary = {
        task: row["previously_unexposed_source_group_count"]
        for task, row in report["tasks"].items()
    }
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
