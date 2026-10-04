from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Any

from .corpus import source_asset_identity
from .dataset import check_train_eval_splits, normalize_jsonl
from .schema import DatasetManifest, DecisionExample, MediaRef

VIDEO_TASK_TYPES = (
    "temporal_descriptive",
    "explanatory",
    "predictive",
    "counterfactual",
)


def _scene_id(example: DecisionExample, source: str) -> int:
    if example.source != source or example.modality != "video":
        raise ValueError(f"expected a {source} video example, got {example.id}")
    scene_text = example.source_record_id.split(":", 1)[0]
    try:
        return int(scene_text)
    except ValueError as exc:
        raise ValueError(f"CLEVRER source record has an invalid scene id: {example.id}") from exc


def _selected_native_examples(
    rows: Iterable[dict[str, Any]],
    *,
    scene_ids: set[int],
    manifest: DatasetManifest,
) -> list[DecisionExample]:
    seen_scenes: set[int] = set()
    result: list[DecisionExample] = []
    for row in rows:
        scene_id = row.get("scene_index")
        if scene_id not in scene_ids:
            continue
        if not isinstance(scene_id, int) or isinstance(scene_id, bool):
            raise ValueError("CLEVRER scene_index must be an integer")
        if scene_id in seen_scenes:
            raise ValueError(f"CLEVRER source repeats video scene {scene_id}")
        seen_scenes.add(scene_id)
        result.extend(
            example
            for example in normalize_jsonl([row], manifest, "clevrer-video-native")
            if example.task_type != "static_descriptive"
        )

    missing = scene_ids - seen_scenes
    if missing:
        raise ValueError(f"CLEVRER source is missing frozen v1 scenes: {sorted(missing)}")
    if not result:
        raise ValueError("no temporal descriptive or native reasoning questions were available")
    return result


def _materialized_video_media_by_scene(
    examples: Iterable[DecisionExample], source: str
) -> dict[int, MediaRef]:
    media_by_scene: dict[int, MediaRef] = {}
    for example in examples:
        scene_id = _scene_id(example, source)
        video_media = [item for item in example.media if item.kind == "video"]
        if len(video_media) != 1 or not video_media[0].path:
            raise ValueError(
                f"frozen v1 scene {scene_id} must have one locally materialized video"
            )
        media = video_media[0]
        previous = media_by_scene.setdefault(scene_id, media)
        if previous != media:
            raise ValueError(f"frozen v1 scene {scene_id} has inconsistent video references")
    return media_by_scene


def _split_statistics(examples: list[DecisionExample]) -> dict[str, Any]:
    video_examples = [item for item in examples if item.modality == "video"]
    task_counts: Counter[str] = Counter()
    task_questions: dict[str, set[str]] = defaultdict(set)
    task_assets: dict[str, set[str]] = defaultdict(set)
    for example in video_examples:
        if example.task_type not in VIDEO_TASK_TYPES:
            raise ValueError(f"video example has an unsupported v2 task type: {example.id}")
        task_type = str(example.task_type)
        task_counts[task_type] += 1
        if not example.task_group_id:
            raise ValueError(f"CLEVRER task has no parent question id: {example.id}")
        task_questions[task_type].add(example.task_group_id)
        task_assets[task_type].add(source_asset_identity(example))

    missing_tasks = set(VIDEO_TASK_TYPES) - task_counts.keys()
    if missing_tasks:
        raise ValueError(f"video split is missing reasoning types: {sorted(missing_tasks)}")
    task_total = sum(task_counts.values())
    scene_ids = sorted({_scene_id(item, "MIT-IBM/CLEVRER") for item in video_examples})
    non_video_counts = Counter(item.modality for item in examples if item.modality != "video")
    return {
        "examples": len(examples),
        "video_examples": len(video_examples),
        "non_video_examples_by_modality": dict(sorted(non_video_counts.items())),
        "unique_video_scenes": scene_ids,
        "unique_video_assets": len({source_asset_identity(item) for item in video_examples}),
        "video_task_examples": dict(sorted(task_counts.items())),
        "unique_video_questions_by_type": {
            task: len(items) for task, items in sorted(task_questions.items())
        },
        "unique_video_assets_by_type": {
            task: len(items) for task, items in sorted(task_assets.items())
        },
        "realized_video_task_example_mix": {
            task: task_counts[task] / task_total for task in VIDEO_TASK_TYPES
        },
    }


def build_video_native_corpora(
    base_train: list[DecisionExample],
    base_validation: list[DecisionExample],
    raw_train_rows: Iterable[dict[str, Any]],
    raw_validation_rows: Iterable[dict[str, Any]],
    train_manifest: DatasetManifest,
    validation_manifest: DatasetManifest,
) -> tuple[list[DecisionExample], list[DecisionExample], dict[str, Any]]:
    """Replace only frozen v1 CLEVRER rows with video-native, scene-grouped questions."""
    if train_manifest.dataset_id != validation_manifest.dataset_id:
        raise ValueError("CLEVRER train and validation manifests must name the same source")
    if train_manifest.split != "train" or validation_manifest.split != "validation":
        raise ValueError("video-native corpus needs pinned CLEVRER train and validation manifests")

    base_train_video = [item for item in base_train if item.modality == "video"]
    base_validation_video = [item for item in base_validation if item.modality == "video"]
    if not base_train_video or not base_validation_video:
        raise ValueError("frozen v1 train and validation must both contain CLEVRER video examples")
    train_scenes = {
        _scene_id(item, train_manifest.dataset_id) for item in base_train_video
    }
    validation_scenes = {
        _scene_id(item, validation_manifest.dataset_id) for item in base_validation_video
    }
    if train_scenes & validation_scenes:
        raise ValueError("frozen v1 train and validation share CLEVRER video scenes")
    train_media_by_scene = _materialized_video_media_by_scene(
        base_train_video, train_manifest.dataset_id
    )
    validation_media_by_scene = _materialized_video_media_by_scene(
        base_validation_video, validation_manifest.dataset_id
    )

    native_train_video = _selected_native_examples(
        raw_train_rows, scene_ids=train_scenes, manifest=train_manifest
    )
    native_validation_video = _selected_native_examples(
        raw_validation_rows, scene_ids=validation_scenes, manifest=validation_manifest
    )
    if {
        _scene_id(item, train_manifest.dataset_id) for item in native_train_video
    } != train_scenes:
        raise ValueError("normalized CLEVRER train videos differ from frozen v1 scene identities")
    if {
        _scene_id(item, validation_manifest.dataset_id) for item in native_validation_video
    } != validation_scenes:
        raise ValueError(
            "normalized CLEVRER validation videos differ from frozen v1 scene identities"
        )
    native_train_video = [
        item.model_copy(
            update={
                "media": [
                    train_media_by_scene[_scene_id(item, train_manifest.dataset_id)]
                ]
            }
        )
        for item in native_train_video
    ]
    native_validation_video = [
        item.model_copy(
            update={
                "media": [
                    validation_media_by_scene[
                        _scene_id(item, validation_manifest.dataset_id)
                    ]
                ]
            }
        )
        for item in native_validation_video
    ]

    train = [item for item in base_train if item.modality != "video"] + native_train_video
    validation = [
        item for item in base_validation if item.modality != "video"
    ] + native_validation_video
    integrity = check_train_eval_splits(train, validation)
    return train, validation, {
        "train": _split_statistics(train),
        "validation": _split_statistics(validation),
        "split_integrity": integrity,
    }
