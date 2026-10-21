from __future__ import annotations

import re
from typing import Any

TASK_TOKENS = {
    "after",
    "before",
    "end",
    "filter_collision",
    "filter_in",
    "filter_moving",
    "filter_out",
    "get_frame",
    "start",
}
TASK_RE = re.compile(
    r"\b(first|last|before|after|enter(?:s|ed|ing)?|exit(?:s|ed|ing)?|begin(?:s|ning)?|ends?|collision|collide|moving|moves?|move|frame)\b",
    re.I,
)
TAXONOMIES = {
    "exist": ["no", "yes"],
    "query_color": ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"],
    "query_material": ["rubber", "metal"],
    "query_shape": ["cube", "sphere", "cylinder"],
    "count": [str(index) for index in range(6)],
}
REASONING_TYPES = {"explanatory", "predictive", "counterfactual"}
CHOICE_TRUTH_OPTIONS = ["wrong", "correct"]


def classify(question: dict[str, Any]) -> str:
    program = question.get("program")
    temporal = (
        bool(TASK_TOKENS.intersection(map(str, program)))
        if isinstance(program, list)
        else bool(TASK_RE.search(str(question.get("question") or "")))
    )
    return "temporal_descriptive" if temporal else "static_descriptive"


def expand_descriptive_questions(
    raw_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    *,
    split: str = "validation",
) -> list[dict[str, Any]]:
    """Expand sampled scenes while retaining their pinned media and split identity."""
    if split not in {"train", "validation"}:
        raise ValueError("unsupported descriptive split")
    media_by_scene: dict[int, dict[str, Any]] = {}
    for row in validation_rows:
        if row.get("split", split) != split:
            raise ValueError("sampled row split mismatch")
        scene = int(row["scene_index"])
        media = {
            key: row[key]
            for key in (
                "media_path",
                "media_sha256",
                "media_bytes",
                "video_filename",
                "scene_group_id",
                "source",
                "split",
            )
            if key in row
        }
        if scene in media_by_scene and media_by_scene[scene] != media:
            raise ValueError(f"inconsistent sampled media identity for scene {scene}")
        media_by_scene[scene] = media

    raw_by_scene = {int(row["scene_index"]): row for row in raw_rows}
    if len(raw_by_scene) != len(raw_rows):
        raise ValueError("raw CLEVRER validation data has duplicate scene rows")
    missing = set(media_by_scene) - set(raw_by_scene)
    if missing:
        raise ValueError(f"raw validation questions missing sampled scenes: {sorted(missing)}")

    expanded: list[dict[str, Any]] = []
    for scene in sorted(media_by_scene):
        raw_scene = raw_by_scene[scene]
        media = media_by_scene[scene]
        if str(raw_scene["video_filename"]) != str(media["video_filename"]):
            raise ValueError(f"raw video filename does not match sampled scene {scene}")
        for question in raw_scene.get("questions", []):
            if question.get("question_type") != "descriptive":
                continue
            taxonomy = question.get("question_subtype")
            target = question.get("answer")
            if taxonomy not in TAXONOMIES or target not in TAXONOMIES[taxonomy]:
                continue
            question_id = int(question["question_id"])
            expanded.append(
                {
                    **media,
                    "id": f"clevrer-{split}-{scene:05d}-q{question_id:03d}",
                    "scene_index": scene,
                    "question_id": question_id,
                    "question_type": classify(question),
                    "taxonomy": taxonomy,
                    "question": str(question["question"]),
                    "options": list(TAXONOMIES[taxonomy]),
                    "target": str(target),
                    "program": question.get("program", []),
                }
            )
    expanded.sort(key=lambda row: (row["scene_index"], row["question_id"]))
    ids = [row["id"] for row in expanded]
    if len(ids) != len(set(ids)):
        raise ValueError("expanded CLEVRER question IDs are not unique")
    return expanded


def clevrer_binary_choice_content_key(row: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    """Return the normalized question/options identity used by split leakage gates."""
    question = re.sub(r"\s+", " ", str(row["question"]).casefold()).strip()
    options = tuple(
        sorted(re.sub(r"\s+", " ", str(option).casefold()).strip() for option in row["options"])
    )
    return question, options


def expand_binary_reasoning_choices(
    raw_rows: list[dict[str, Any]],
    sampled_rows: list[dict[str, Any]],
    *,
    split: str,
    excluded_content: set[tuple[str, tuple[str, ...]]] | None = None,
) -> list[dict[str, Any]]:
    """Map each non-descriptive CLEVRER choice to a binary truth Decision.

    All choices retain their parent scene and question identity. Callers pass
    one already scene-grouped split at a time and may provide content keys from
    the training split to exclude exact normalized validation duplicates.
    """
    if split not in {"train", "validation"}:
        raise ValueError("split must be 'train' or 'validation'")

    media_by_scene: dict[int, dict[str, Any]] = {}
    for row in sampled_rows:
        if row.get("split") != split:
            raise ValueError(f"sampled media row is not marked as {split}")
        scene = int(row["scene_index"])
        media = {
            key: row[key]
            for key in (
                "media_path",
                "media_sha256",
                "media_bytes",
                "video_filename",
                "scene_group_id",
                "source",
                "split",
            )
            if key in row
        }
        if scene in media_by_scene and media_by_scene[scene] != media:
            raise ValueError(f"inconsistent sampled media identity for scene {scene}")
        media_by_scene[scene] = media

    raw_by_scene = {int(row["scene_index"]): row for row in raw_rows}
    if len(raw_by_scene) != len(raw_rows):
        raise ValueError("raw CLEVRER reasoning data has duplicate scene rows")
    missing = set(media_by_scene) - set(raw_by_scene)
    if missing:
        raise ValueError(f"raw CLEVRER reasoning data missing scenes: {sorted(missing)}")

    expanded: list[dict[str, Any]] = []
    excluded_content = excluded_content or set()
    for scene in sorted(media_by_scene):
        raw_scene = raw_by_scene[scene]
        media = media_by_scene[scene]
        if str(raw_scene["video_filename"]) != str(media["video_filename"]):
            raise ValueError(f"raw video filename does not match sampled scene {scene}")
        for question in raw_scene.get("questions", []):
            question_type = question.get("question_type")
            if question_type not in REASONING_TYPES:
                continue
            question_id = int(question["question_id"])
            question_text = str(question.get("question", "")).strip()
            if not question_text:
                raise ValueError(f"CLEVRER reasoning question {scene}/{question_id} is empty")
            choices = question.get("choices")
            if not isinstance(choices, list) or not choices:
                raise ValueError(f"CLEVRER reasoning question {scene}/{question_id} has no choices")
            choice_ids: set[int] = set()
            for choice in choices:
                choice_id = int(choice["choice_id"])
                if choice_id in choice_ids:
                    raise ValueError(
                        f"duplicate CLEVRER choice ID {scene}/{question_id}/{choice_id}"
                    )
                choice_ids.add(choice_id)
                statement = str(choice.get("choice", "")).strip()
                answer = str(choice.get("answer", "")).strip().casefold()
                if not statement:
                    raise ValueError(
                        f"CLEVRER choice statement is empty: {scene}/{question_id}/{choice_id}"
                    )
                if answer not in CHOICE_TRUTH_OPTIONS:
                    raise ValueError(
                        "choice answer must be 'correct' or 'wrong' at "
                        f"{scene}/{question_id}/{choice_id}"
                    )
                candidate_question = f"{question_text}\nCandidate statement: {statement}"
                row = {
                    **media,
                    "id": (f"clevrer-{split}-{scene:05d}-q{question_id:03d}-c{choice_id:03d}"),
                    "scene_index": scene,
                    "question_id": question_id,
                    "parent_question_id": f"clevrer:{split}:{scene}:{question_id}",
                    "choice_id": choice_id,
                    "question_type": str(question_type),
                    "taxonomy": "binary_choice_truth",
                    "question": candidate_question,
                    "source_question": question_text,
                    "candidate_statement": statement,
                    "options": list(CHOICE_TRUTH_OPTIONS),
                    "target": answer,
                    "program": question.get("program", []),
                }
                if clevrer_binary_choice_content_key(row) not in excluded_content:
                    expanded.append(row)

    expanded.sort(key=lambda row: (row["scene_index"], row["question_id"], row["choice_id"]))
    ids = [row["id"] for row in expanded]
    if len(ids) != len(set(ids)):
        raise ValueError("expanded CLEVRER reasoning IDs are not unique")
    return expanded
