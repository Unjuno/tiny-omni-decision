"""Freeze a grouped inner development split from the legacy task's train rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from scripts.prepare_tallyqa_vg_label_independent_task import TASK_ID
from tiny_omni_decision.dataset import sha256_file

PINNED_TASK_MANIFEST_SHA256 = "5428876a3cc92d2a4194d6b3e05e1c964c73c5da49c5b3728577eb961516477a"


class _DisjointSet:
    def __init__(self, values: list[int]) -> None:
        self.parent = {value: value for value in values}
        self.size = {value: 1 for value in values}

    def find(self, value: int) -> int:
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.size[left_root] < self.size[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.size[left_root] += self.size[right_root]


def grouped_split(
    rows: list[dict[str, Any]], *, seed: int = 17, validation_fraction: float = 0.2
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if not rows or not 0 < validation_fraction < 1:
        raise ValueError("non-empty rows and a validation fraction in (0, 1) are required")
    question_ids = [int(row["question_id"]) for row in rows]
    if len(question_ids) != len(set(question_ids)):
        raise ValueError("duplicate question IDs in inner split source")
    image_ids = sorted({int(row["image_id"]) for row in rows})
    dsu = _DisjointSet(image_ids)
    first_image_by_question: dict[str, int] = {}
    for row in rows:
        normalized = str(row["normalized_question"])
        image_id = int(row["image_id"])
        if normalized in first_image_by_question:
            dsu.union(image_id, first_image_by_question[normalized])
        else:
            first_image_by_question[normalized] = image_id

    components: dict[int, list[int]] = defaultdict(list)
    for image_id in image_ids:
        components[dsu.find(image_id)].append(image_id)
    groups = [sorted(group) for group in components.values()]
    groups.sort(
        key=lambda group: hashlib.sha256(
            f"{seed}:".encode("ascii") + ",".join(map(str, group)).encode("ascii")
        ).digest()
    )
    target_validation_images = max(1, round(len(image_ids) * validation_fraction))
    validation_groups: list[list[int]] = []
    validation_count = 0
    remaining: list[list[int]] = []
    for group in groups:
        if validation_count < target_validation_images and (
            validation_count + len(group) <= target_validation_images
        ):
            validation_groups.append(group)
            validation_count += len(group)
        else:
            remaining.append(group)
    for group in remaining[:]:
        if validation_count >= target_validation_images:
            break
        if validation_count + len(group) <= target_validation_images:
            validation_groups.append(group)
            validation_count += len(group)
            remaining.remove(group)
    validation_images = {image_id for group in validation_groups for image_id in group}
    train_rows = [row for row in rows if int(row["image_id"]) not in validation_images]
    validation_rows = [row for row in rows if int(row["image_id"]) in validation_images]
    if not train_rows or not validation_rows:
        raise ValueError("grouped inner split produced an empty partition")
    if {row["image_id"] for row in train_rows} & {row["image_id"] for row in validation_rows}:
        raise AssertionError("image identity crossed the inner split")
    if {row["normalized_question"] for row in train_rows} & {
        row["normalized_question"] for row in validation_rows
    }:
        raise AssertionError("normalized question content crossed the inner split")
    if {row["media_sha256"] for row in train_rows} & {
        row["media_sha256"] for row in validation_rows
    }:
        raise AssertionError("media hash crossed the inner split")
    metadata = {
        "seed": seed,
        "validation_fraction_target": validation_fraction,
        "content_group_components": len(groups),
        "train_questions": len(train_rows),
        "validation_questions": len(validation_rows),
        "train_images": len({int(row["image_id"]) for row in train_rows}),
        "validation_images": len({int(row["image_id"]) for row in validation_rows}),
        "cross_split_image_overlap": 0,
        "cross_split_normalized_question_overlap": 0,
        "cross_split_media_hash_overlap": 0,
        "source_pool_role": "previously_used_legacy_training_rows; development_only",
    }
    return train_rows, validation_rows, metadata


def build_inner_manifest(task: dict[str, Any]) -> dict[str, Any]:
    if task.get("task", {}).get("id") != TASK_ID:
        raise ValueError("source task is not the fixed 16-choice label-independent task")
    train_rows, validation_rows, split = grouped_split(task["train"])
    result = dict(task)
    result["train"] = train_rows
    result["validation"] = validation_rows
    result["task"] = dict(task["task"])
    result["task"]["validation_prior_exposure"] = (
        "This inner holdout is drawn from the legacy four-choice training pool. "
        "It is reserved from this 16-choice fit, but is development-only, not blind."
    )
    result["inner_split"] = split
    result["parent_task_manifest_sha256"] = PINNED_TASK_MANIFEST_SHA256
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_sha = sha256_file(args.task_manifest)
    if source_sha != PINNED_TASK_MANIFEST_SHA256:
        raise ValueError("source task manifest SHA-256 differs from the pinned input")
    source = json.loads(args.task_manifest.read_text(encoding="utf-8"))
    output = build_inner_manifest(source)
    output["source_task_manifest_sha256"] = source_sha
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    args.output.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(output["inner_split"], indent=2))


if __name__ == "__main__":
    main()
