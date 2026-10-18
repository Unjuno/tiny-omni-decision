"""Select a deterministic scene-disjoint CLEVRER temporal split."""

from __future__ import annotations

import hashlib
from typing import Any

from tiny_omni_decision.clevrer import (
    CHOICE_TRUTH_OPTIONS,
    REASONING_TYPES,
    TAXONOMIES,
    classify,
)

TEMPORAL_TASK_TYPES = (
    "temporal_descriptive",
    "explanatory",
    "predictive",
    "counterfactual",
)


def _content_key(row: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    question = " ".join(str(row["question"]).casefold().split())
    options = tuple(sorted(" ".join(str(option).casefold().split()) for option in row["options"]))
    return question, options


def _stable_key(seed: int, split: str, identity: str) -> str:
    return hashlib.sha256(f"{seed}|{split}|{identity}".encode()).hexdigest()


def _assign_split(row: dict[str, Any], split: str) -> dict[str, Any]:
    assigned = dict(row)
    assigned["split"] = split
    assigned["id"] = (
        f"clevrer-{split}-{int(row['scene_index']):05d}-q{int(row['question_id']):03d}"
    )
    return assigned


def _temporal_candidates(raw_rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    by_scene: dict[int, list[dict[str, Any]]] = {}
    seen_scenes: set[int] = set()
    for raw_scene in raw_rows:
        scene = int(raw_scene["scene_index"])
        if scene in seen_scenes:
            raise ValueError(f"duplicate CLEVRER scene record: {scene}")
        seen_scenes.add(scene)
        video_filename = str(raw_scene["video_filename"])
        for question in raw_scene.get("questions", []):
            taxonomy = question.get("question_subtype")
            target = question.get("answer")
            if (
                question.get("question_type") != "descriptive"
                or taxonomy not in TAXONOMIES
                or target not in TAXONOMIES[taxonomy]
                or classify(question) != "temporal_descriptive"
            ):
                continue
            question_id = int(question["question_id"])
            by_scene.setdefault(scene, []).append(
                {
                    "id": f"clevrer-train-{scene:05d}-q{question_id:03d}",
                    "split": "train",
                    "scene_index": scene,
                    "scene_group_id": f"clevrer-video-{scene:05d}",
                    "video_filename": video_filename,
                    "question_id": question_id,
                    "question_type": "temporal_descriptive",
                    "taxonomy": str(taxonomy),
                    "question": str(question["question"]),
                    "options": list(TAXONOMIES[taxonomy]),
                    "target": str(target),
                    "program": question.get("program", []),
                }
            )
    return by_scene


def select_temporal_split(
    raw_rows: list[dict[str, Any]],
    *,
    validation_scenes: int,
    train_examples: int,
    seed: int,
    excluded_scenes: set[int] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return unique-content temporal rows, with validation allocated by scene first."""
    if validation_scenes < 0 or train_examples < 1:
        raise ValueError("validation_scenes must be non-negative and train_examples positive")
    excluded = excluded_scenes or set()
    candidates = _temporal_candidates(raw_rows)
    candidates = {scene: rows for scene, rows in candidates.items() if scene not in excluded}

    def ordered_rows(scene: int, split: str) -> list[dict[str, Any]]:
        return sorted(
            candidates[scene],
            key=lambda row: _stable_key(seed, split, f"{scene}|{row['question_id']}"),
        )

    validation: list[dict[str, Any]] = []
    validation_content: set[tuple[str, tuple[str, ...]]] = set()
    validation_scene_ids: set[int] = set()
    scenes_by_hash = sorted(
        candidates,
        key=lambda scene: _stable_key(seed, "validation-scene", str(scene)),
    )
    for scene in scenes_by_hash:
        for row in ordered_rows(scene, "validation-question"):
            key = _content_key(row)
            if key in validation_content:
                continue
            validation.append(_assign_split(row, "validation"))
            validation_scene_ids.add(scene)
            validation_content.add(key)
            break
        if len(validation_scene_ids) == validation_scenes:
            break
    if len(validation_scene_ids) != validation_scenes:
        raise ValueError(
            "cannot satisfy temporal split quotas: "
            f"only {len(validation_scene_ids)} validation scenes available, "
            f"need {validation_scenes}"
        )

    train: list[dict[str, Any]] = []
    train_content: set[tuple[str, tuple[str, ...]]] = set()
    train_scenes = sorted(
        (scene for scene in candidates if scene not in validation_scene_ids),
        key=lambda scene: _stable_key(seed, "train-scene", str(scene)),
    )
    per_scene = {scene: ordered_rows(scene, "train-question") for scene in train_scenes}
    max_questions_per_scene = max((len(rows) for rows in per_scene.values()), default=0)
    for question_rank in range(max_questions_per_scene):
        for scene in train_scenes:
            rows = per_scene[scene]
            if question_rank >= len(rows):
                continue
            row = rows[question_rank]
            key = _content_key(row)
            if key in validation_content or key in train_content:
                continue
            train.append(_assign_split(row, "train"))
            train_content.add(key)
            if len(train) == train_examples:
                break
        if len(train) == train_examples:
            break

    if len(train) != train_examples:
        raise ValueError(
            "cannot satisfy temporal split quotas: "
            f"only {len(train)} unique train examples available, need {train_examples}"
        )
    train_scene_ids = {row["scene_index"] for row in train}
    if train_scene_ids & validation_scene_ids:
        raise ValueError("CLEVRER video scene overlap across train and validation")
    if train_content & validation_content:
        raise ValueError("normalized CLEVRER question/options overlap across splits")
    return train, validation


def _task_groups(
    raw_rows: list[dict[str, Any]], excluded_scenes: set[int]
) -> dict[str, dict[int, list[list[dict[str, Any]]]]]:
    groups: dict[str, dict[int, list[list[dict[str, Any]]]]] = {
        task: {} for task in TEMPORAL_TASK_TYPES
    }
    seen_scenes: set[int] = set()
    for raw_scene in raw_rows:
        scene = int(raw_scene["scene_index"])
        if scene in seen_scenes:
            raise ValueError(f"duplicate CLEVRER scene record: {scene}")
        seen_scenes.add(scene)
        if scene in excluded_scenes:
            continue
        filename = str(raw_scene["video_filename"])
        for question in raw_scene.get("questions", []):
            kind = question.get("question_type")
            question_id = int(question["question_id"])
            if kind == "descriptive":
                taxonomy = question.get("question_subtype")
                target = question.get("answer")
                if (
                    taxonomy not in TAXONOMIES
                    or target not in TAXONOMIES.get(str(taxonomy), [])
                    or classify(question) != "temporal_descriptive"
                ):
                    continue
                group = [
                    {
                        "scene_index": scene,
                        "video_filename": filename,
                        "question_id": question_id,
                        "parent_question_id": f"clevrer:{scene}:{question_id}",
                        "question_type": "temporal_descriptive",
                        "taxonomy": str(taxonomy),
                        "question": str(question["question"]),
                        "options": list(TAXONOMIES[str(taxonomy)]),
                        "target": str(target),
                        "program": question.get("program", []),
                    }
                ]
                groups["temporal_descriptive"].setdefault(scene, []).append(group)
                continue
            if kind not in REASONING_TYPES:
                continue
            question_text = str(question.get("question", "")).strip()
            choices = question.get("choices")
            if not question_text or not isinstance(choices, list):
                continue
            candidates: list[dict[str, Any]] = []
            choice_ids: set[int] = set()
            for choice in choices:
                choice_id = int(choice["choice_id"])
                statement = str(choice.get("choice", "")).strip()
                answer = str(choice.get("answer", "")).strip().casefold()
                if choice_id in choice_ids:
                    raise ValueError(
                        f"duplicate CLEVRER choice ID {scene}/{question_id}/{choice_id}"
                    )
                choice_ids.add(choice_id)
                if not statement or answer not in CHOICE_TRUTH_OPTIONS:
                    candidates = []
                    break
                candidates.append(
                    {
                        "scene_index": scene,
                        "video_filename": filename,
                        "question_id": question_id,
                        "choice_id": choice_id,
                        "parent_question_id": f"clevrer:{scene}:{question_id}",
                        "question_type": str(kind),
                        "taxonomy": "binary_choice_truth",
                        "question": f"{question_text}\nCandidate statement: {statement}",
                        "source_question": question_text,
                        "candidate_statement": statement,
                        "options": list(CHOICE_TRUTH_OPTIONS),
                        "target": answer,
                        "program": question.get("program", []),
                    }
                )
            if {row["target"] for row in candidates} != set(CHOICE_TRUTH_OPTIONS):
                continue
            groups[str(kind)].setdefault(scene, []).append(candidates)
    return groups


def _pick_group_rows(
    candidates: list[dict[str, Any]], task: str, split: str, seed: int,
    excluded_content: set[tuple[str, tuple[str, ...]]],
) -> list[dict[str, Any]] | None:
    ordered = sorted(
        candidates,
        key=lambda row: _stable_key(
            seed, f"{split}-{task}-question", f"{row['question_id']}|{row.get('choice_id', -1)}"
        ),
    )
    if task == "temporal_descriptive":
        for row in ordered:
            if _content_key(row) not in excluded_content:
                return [_assign_task_split(row, split)]
        return None
    correct = [row for row in ordered if row["target"] == "correct"]
    wrong = [row for row in ordered if row["target"] == "wrong"]
    for positive in correct:
        positive_key = _content_key(positive)
        if positive_key in excluded_content:
            continue
        for negative in wrong:
            negative_key = _content_key(negative)
            if negative_key not in excluded_content and negative_key != positive_key:
                return [
                    _assign_task_split(positive, split),
                    _assign_task_split(negative, split),
                ]
    return None


def _assign_task_split(row: dict[str, Any], split: str) -> dict[str, Any]:
    assigned = dict(row)
    assigned["split"] = split
    if "choice_id" in row:
        assigned["id"] = (
            f"clevrer-{split}-{int(row['scene_index']):05d}-q{int(row['question_id']):03d}"
            f"-c{int(row['choice_id']):03d}"
        )
    else:
        assigned["id"] = (
            f"clevrer-{split}-{int(row['scene_index']):05d}-q{int(row['question_id']):03d}"
        )
    return assigned


def select_temporal_task_mix(
    raw_rows: list[dict[str, Any]],
    *,
    validation_scenes_per_type: int,
    train_example_counts: dict[str, int],
    seed: int,
    excluded_scenes: set[int] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select a frozen temporal task mix with scene, content, and parent grouping gates."""
    if validation_scenes_per_type < 1:
        raise ValueError("validation_scenes_per_type must be positive")
    if set(train_example_counts) != set(TEMPORAL_TASK_TYPES):
        raise ValueError("train_example_counts must specify all four temporal task types")
    for task, count in train_example_counts.items():
        if count < 1 or (task != "temporal_descriptive" and count % 2):
            raise ValueError("reasoning train row quotas must be positive even counts")

    groups = _task_groups(raw_rows, excluded_scenes or set())
    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    train_scenes: set[int] = set()
    validation_scenes: set[int] = set()
    train_content: set[tuple[str, tuple[str, ...]]] = set()
    validation_content: set[tuple[str, tuple[str, ...]]] = set()

    for task in TEMPORAL_TASK_TYPES:
        selected_for_task = 0
        scene_order = sorted(
            groups[task], key=lambda scene: _stable_key(seed, "validation-scene", f"{task}|{scene}")
        )
        for scene in scene_order:
            if scene in validation_scenes:
                continue
            parent_groups = sorted(
                groups[task][scene],
                key=lambda candidate: _stable_key(
                    seed, "validation-parent", f"{task}|{scene}|{candidate[0]['question_id']}"
                ),
            )
            selected = next(
                (
                    rows
                    for candidate in parent_groups
                    if (
                        rows := _pick_group_rows(
                            candidate, task, "validation", seed, validation_content
                        )
                    )
                    is not None
                ),
                None,
            )
            if selected is None:
                continue
            validation.extend(selected)
            validation_scenes.add(scene)
            validation_content.update(_content_key(row) for row in selected)
            selected_for_task += 1
            if selected_for_task == validation_scenes_per_type:
                break
        if selected_for_task != validation_scenes_per_type:
            raise ValueError(
                f"cannot satisfy validation quota for {task}: "
                f"selected {selected_for_task}, need {validation_scenes_per_type}"
            )

    if train_scenes & validation_scenes:
        raise ValueError("CLEVRER scene overlap across train and validation")

    for task in TEMPORAL_TASK_TYPES:
        quota = train_example_counts[task]
        selected_for_task = 0
        scene_order = sorted(
            groups[task], key=lambda scene: _stable_key(seed, "train-scene", f"{task}|{scene}")
        )
        for scene in scene_order:
            if scene in validation_scenes or scene in train_scenes:
                continue
            parent_groups = sorted(
                groups[task][scene],
                key=lambda candidate: _stable_key(
                    seed, "train-parent", f"{task}|{scene}|{candidate[0]['question_id']}"
                ),
            )
            selected = next(
                (
                    rows
                    for candidate in parent_groups
                    if (
                    rows := _pick_group_rows(
                        candidate,
                        task,
                        "train",
                        seed,
                        train_content | validation_content,
                    )
                    )
                    is not None
                ),
                None,
            )
            if selected is None or selected_for_task + len(selected) > quota:
                continue
            train.extend(selected)
            train_scenes.add(scene)
            train_content.update(_content_key(row) for row in selected)
            selected_for_task += len(selected)
            if selected_for_task == quota:
                break
        if selected_for_task != quota:
            raise ValueError(
                f"cannot satisfy train quota for {task}: selected {selected_for_task}, need {quota}"
            )

    if train_scenes & validation_scenes:
        raise ValueError("CLEVRER scene overlap across train and validation")
    if train_content & validation_content:
        raise ValueError("normalized question/options overlap across splits")
    return (
        sorted(
            train,
            key=lambda row: (
                row["scene_index"], row["question_id"], row.get("choice_id", -1)
            ),
        ),
        sorted(
            validation,
            key=lambda row: (row["scene_index"], row["question_id"], row.get("choice_id", -1)),
        ),
    )
