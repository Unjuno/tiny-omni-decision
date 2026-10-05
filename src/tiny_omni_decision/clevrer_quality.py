from __future__ import annotations

import hashlib
import random
from collections import Counter
from collections.abc import Iterable
from pathlib import PurePosixPath
from typing import Any

from .dataset import check_train_eval_splits, normalize_jsonl
from .schema import DatasetManifest, DecisionExample, MediaRef


def select_unseen_clevrer_scenes(
    rows: Iterable[dict[str, Any]],
    *,
    previously_observed: set[int],
    train_scene_count: int,
    validation_scene_count: int,
    seed: int,
) -> tuple[set[int], set[int]]:
    """Select reproducible, scene-disjoint development splits from CLEVRER train."""
    if train_scene_count < 1 or validation_scene_count < 1:
        raise ValueError("train and validation scene counts must both be positive")
    available: set[int] = set()
    for row in rows:
        scene = row.get("scene_index")
        if not isinstance(scene, int) or isinstance(scene, bool):
            raise ValueError("CLEVRER rows need integer scene_index values")
        if not 0 <= scene < 10_000:
            raise ValueError("fresh CLEVRER scenes must come from the official train range")
        available.add(scene)
    candidates = sorted(available - previously_observed)
    random.Random(seed).shuffle(candidates)
    required = train_scene_count + validation_scene_count
    if len(candidates) < required:
        raise ValueError(
            f"only {len(candidates)} unseen CLEVRER scenes are available; {required} required"
        )
    validation = set(candidates[:validation_scene_count])
    train = set(candidates[validation_scene_count:required])
    if train & validation or (train | validation) & previously_observed:
        raise AssertionError("CLEVRER selection leaked an observed or shared scene")
    return train, validation


def materialize_clevrer_examples(
    rows: Iterable[dict[str, Any]],
    *,
    train_scenes: set[int],
    validation_scenes: set[int],
    manifest: DatasetManifest,
    media_by_scene: dict[int, MediaRef],
) -> tuple[list[DecisionExample], list[DecisionExample], dict[str, Any]]:
    """Normalize selected train-source scenes and attach verified local media refs."""
    if manifest.dataset_id != "MIT-IBM/CLEVRER" or manifest.split != "train":
        raise ValueError("fresh CLEVRER development data must use the pinned train manifest")
    if not train_scenes or not validation_scenes:
        raise ValueError("fresh CLEVRER train and validation scene sets must be non-empty")
    if train_scenes & validation_scenes:
        raise ValueError("CLEVRER scene identities overlap between train and validation")
    requested = train_scenes | validation_scenes
    rows_by_scene: dict[int, dict[str, Any]] = {}
    for row in rows:
        scene = row.get("scene_index")
        if scene in requested:
            if scene in rows_by_scene:
                raise ValueError(f"CLEVRER source repeats scene {scene}")
            rows_by_scene[scene] = row
    missing_scenes = requested - rows_by_scene.keys()
    if missing_scenes:
        raise ValueError(f"CLEVRER source is missing selected scenes: {sorted(missing_scenes)}")
    if requested - media_by_scene.keys():
        raise ValueError("a selected CLEVRER scene is missing its verified local video")

    split_examples: dict[str, list[DecisionExample]] = {"train": [], "validation": []}
    for scene, row in rows_by_scene.items():
        split = "train" if scene in train_scenes else "validation"
        for example in normalize_jsonl([row], manifest, "clevrer-video-native"):
            if example.task_type == "static_descriptive":
                continue
            media = media_by_scene[scene]
            if media.kind != "video" or not media.path or not media.sha256:
                raise ValueError(f"scene {scene} needs a local video path and verified SHA-256")
            split_examples[split].append(
                example.model_copy(update={"split": split, "media": [media]})
            )
    if not split_examples["train"] or not split_examples["validation"]:
        raise ValueError("selected scenes produced an empty CLEVRER split")
    integrity = check_train_eval_splits(
        split_examples["train"], split_examples["validation"]
    )
    if integrity.get("status") != "disjoint":
        raise ValueError("CLEVRER fresh scene split failed the existing overlap gate")

    def stats(examples: list[DecisionExample]) -> dict[str, Any]:
        task_counts = Counter(item.task_type for item in examples)
        scenes = {
            int(item.source_record_id.split(":", 1)[0]) for item in examples
        }
        total = len(examples)
        return {
            "examples": total,
            "unique_scenes": len(scenes),
            "scene_ids_sha256": hashlib.sha256(
                "\n".join(map(str, sorted(scenes))).encode()
            ).hexdigest(),
            "task_type_counts": dict(sorted(task_counts.items())),
            "task_type_fractions": {
                str(key): value / total for key, value in sorted(task_counts.items())
            },
            "unique_parent_questions": len({item.task_group_id for item in examples}),
            "unique_media_sha256": len({item.media[0].sha256 for item in examples}),
        }

    return (
        split_examples["train"],
        split_examples["validation"],
        {
            "train": stats(split_examples["train"]),
            "validation": stats(split_examples["validation"]),
            "split_integrity": integrity,
        },
    )


def media_path_for_scene(
    scene: int, *, media_root: str = "raw/teacher-quality-next/clevrer"
) -> str:
    """Return a data-root-relative path for one official CLEVRER train video."""
    if not 0 <= scene < 10_000:
        raise ValueError("CLEVRER scene id is outside the official train range")
    return PurePosixPath(media_root, f"video_{scene:05}.mp4").as_posix()
