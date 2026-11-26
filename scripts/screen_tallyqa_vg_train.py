"""Screen the pinned TallyQA training member for a VG-only AMT split.

This utility intentionally reads only ``train.json`` from the source archive.
It never opens the official ``test.json`` member.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SOURCE_REPOSITORY = "https://github.com/manoja328/TallyQA_dataset"
SOURCE_COMMIT = "46cdc649ec79c3dcc2720ff227ad07d7ee51da6f"
EXPECTED_ARCHIVE_SHA256 = "e805199fa25c91d2f9df964920c6c36ce8d6cb941c970176f38a92f971bab5ea"
TRAIN_MEMBER = "train.json"
ALLOWED_SOURCE = "amt"
ALLOWED_IMAGE_PREFIXES = frozenset({"VG_100K", "VG_100K_2"})
SEED = 17
VALIDATION_FRACTION = 0.2


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_question(question: str) -> str:
    text = unicodedata.normalize("NFKC", question).casefold()
    return " ".join(re.findall(r"[\w]+", text, flags=re.UNICODE))


def load_training_rows(archive_path: Path) -> list[dict[str, Any]]:
    """Load only the training annotation member; do not inspect archive names."""
    with zipfile.ZipFile(archive_path) as archive:
        with archive.open(TRAIN_MEMBER, "r") as stream:
            rows = json.load(stream)
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("train.json must contain a JSON list of objects")
    return rows


class _DisjointSet:
    def __init__(self, values: list[int]) -> None:
        self.parent = {value: value for value in values}
        self.size = {value: 1 for value in values}

    def find(self, value: int) -> int:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.size[left_root] < self.size[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.size[left_root] += self.size[right_root]


def _ordered_id_hash(values: list[int]) -> str:
    return sha256_bytes("\n".join(str(value) for value in values).encode("ascii"))


def screen_training_rows(
    rows: list[dict[str, Any]], *, seed: int = SEED
) -> dict[str, Any]:
    source_counts = Counter(str(row.get("data_source", "<missing>")) for row in rows)
    prefix_counts = Counter(
        str(row.get("image", "")).replace("\\", "/").split("/")[0]
        for row in rows
    )

    candidates: list[dict[str, Any]] = []
    for row in rows:
        if row.get("data_source") != ALLOWED_SOURCE:
            continue
        image_path = str(row.get("image", "")).replace("\\", "/")
        prefix = image_path.split("/")[0]
        if prefix not in ALLOWED_IMAGE_PREFIXES:
            raise ValueError("AMT training row does not reference an allowed VG path")
        if not image_path or "/" not in image_path:
            raise ValueError("AMT training row has no source-relative image path")
        if not isinstance(row.get("answer"), int) or isinstance(row.get("answer"), bool):
            raise ValueError("AMT training answer must be an integer")
        if not isinstance(row.get("image_id"), int):
            raise ValueError("AMT training image_id must be an integer")
        if not isinstance(row.get("question_id"), int):
            raise ValueError("AMT training question_id must be an integer")
        if not isinstance(row.get("question"), str) or not normalize_question(row["question"]):
            raise ValueError("AMT training question must be non-empty text")
        candidates.append(row)

    if not candidates:
        raise ValueError("no eligible AMT/Visual Genome training rows")
    question_ids = [row["question_id"] for row in candidates]
    if len(set(question_ids)) != len(question_ids):
        raise ValueError("duplicate question_id in eligible training rows")

    image_paths: dict[int, str] = {}
    image_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    question_images: dict[str, list[int]] = defaultdict(list)
    for row in candidates:
        image_id = row["image_id"]
        image_path = str(row["image"]).replace("\\", "/")
        if image_id in image_paths and image_paths[image_id] != image_path:
            raise ValueError("one image_id maps to multiple image paths")
        image_paths[image_id] = image_path
        image_rows[image_id].append(row)
        question_images[normalize_question(row["question"])].append(image_id)

    image_ids = sorted(image_rows)
    if len(image_ids) < 2:
        raise ValueError("at least two eligible image groups are required for a split")
    components = _DisjointSet(image_ids)
    for grouped_images in question_images.values():
        for image_id in grouped_images[1:]:
            components.union(grouped_images[0], image_id)

    grouped_components: dict[int, list[int]] = defaultdict(list)
    for image_id in image_ids:
        grouped_components[components.find(image_id)].append(image_id)
    component_values = [sorted(values) for values in grouped_components.values()]
    component_values.sort(
        key=lambda values: hashlib.sha256(
            (f"{seed}:" + ",".join(str(value) for value in values)).encode("ascii")
        ).digest()
    )

    target_validation_images = max(1, round(len(image_ids) * VALIDATION_FRACTION))
    validation_components: list[list[int]] = []
    validation_count = 0
    remaining_components: list[list[int]] = []
    for component in component_values:
        if validation_count < target_validation_images and (
            validation_count + len(component) <= target_validation_images
        ):
            validation_components.append(component)
            validation_count += len(component)
        else:
            remaining_components.append(component)
    # A final deterministic pass fills any shortfall without splitting a group.
    for component in remaining_components[:]:
        if validation_count >= target_validation_images:
            break
        if validation_count + len(component) <= target_validation_images:
            validation_components.append(component)
            validation_count += len(component)
            remaining_components.remove(component)

    if validation_count == 0 or validation_count >= len(image_ids):
        raise ValueError("deterministic grouped split produced an empty partition")
    validation_image_ids = {
        image_id for component in validation_components for image_id in component
    }
    train_image_ids = set(image_ids) - validation_image_ids
    train_rows = [row for row in candidates if row["image_id"] in train_image_ids]
    validation_rows = [row for row in candidates if row["image_id"] in validation_image_ids]

    train_questions = {normalize_question(row["question"]) for row in train_rows}
    validation_questions = {normalize_question(row["question"]) for row in validation_rows}
    if train_image_ids & validation_image_ids:
        raise AssertionError("image identity crossed the split")
    if train_questions & validation_questions:
        raise AssertionError("normalized question content crossed the split")

    def split_record(split_rows: list[dict[str, Any]], split_images: set[int]) -> dict[str, Any]:
        ordered_questions = [row["question_id"] for row in split_rows]
        ordered_images = sorted(split_images)
        normalized_questions = sorted({normalize_question(row["question"]) for row in split_rows})
        return {
            "question_count": len(split_rows),
            "image_count": len(split_images),
            "question_ids_sha256_ordered_source_order": _ordered_id_hash(ordered_questions),
            "image_ids_sha256_sorted": _ordered_id_hash(ordered_images),
            "normalized_questions_sha256_sorted": sha256_bytes(
                "\n".join(normalized_questions).encode("utf-8")
            ),
            "question_ids": ordered_questions,
            "image_ids": ordered_images,
            "image_paths": {
                str(image_id): image_paths[image_id] for image_id in ordered_images
            },
        }

    component_sizes = sorted((len(values) for values in grouped_components.values()), reverse=True)
    return {
        "source": {
            "repository": SOURCE_REPOSITORY,
            "commit": SOURCE_COMMIT,
            "archive_member": TRAIN_MEMBER,
            "rows_in_training_member": len(rows),
            "data_source_counts": dict(sorted(source_counts.items())),
            "image_directory_counts": dict(sorted(prefix_counts.items())),
        },
        "filter": {
            "data_source": ALLOWED_SOURCE,
            "allowed_image_prefixes": sorted(ALLOWED_IMAGE_PREFIXES),
            "selected_rows": len(candidates),
            "excluded_rows": len(rows) - len(candidates),
            "answer_min": min(row["answer"] for row in candidates),
            "answer_max": max(row["answer"] for row in candidates),
            "image_path_count": len(set(image_paths.values())),
            "unique_questions_normalized": len(
                {normalize_question(row["question"]) for row in candidates}
            ),
        },
        "split": {
            "algorithm": (
                "SHA-256(seed, component image_ids) order; greedy whole-component "
                "assignment to the target validation image count"
            ),
            "seed": seed,
            "validation_fraction_target": VALIDATION_FRACTION,
            "target_validation_images": target_validation_images,
            "content_group_component_count": len(component_sizes),
            "largest_content_group_sizes": component_sizes[:10],
            "image_overlap_count": 0,
            "normalized_question_overlap_count": 0,
            "train": split_record(train_rows, train_image_ids),
            "validation": split_record(validation_rows, validation_image_ids),
        },
        "archive_sha256": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    archive_sha256 = sha256_file(args.archive)
    if archive_sha256 != EXPECTED_ARCHIVE_SHA256:
        raise ValueError("annotation archive SHA-256 does not match the pinned source")
    result = screen_training_rows(load_training_rows(args.archive), seed=args.seed)
    result["archive_sha256"] = archive_sha256
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    compact = {
        key: result["split"][key]
        for key in (
            "seed",
            "content_group_component_count",
            "image_overlap_count",
            "normalized_question_overlap_count",
        )
    }
    compact["train"] = {
        key: value
        for key, value in result["split"]["train"].items()
        if key not in {"question_ids", "image_ids", "image_paths"}
    }
    compact["validation"] = {
        key: value
        for key, value in result["split"]["validation"].items()
        if key not in {"question_ids", "image_ids", "image_paths"}
    }
    compact["filter"] = result["filter"]
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
