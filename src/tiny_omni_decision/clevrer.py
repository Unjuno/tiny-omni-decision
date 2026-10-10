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


def classify(question: dict[str, Any]) -> str:
    program = question.get("program")
    temporal = (
        bool(TASK_TOKENS.intersection(map(str, program)))
        if isinstance(program, list)
        else bool(TASK_RE.search(str(question.get("question") or "")))
    )
    return "temporal_descriptive" if temporal else "static_descriptive"


def expand_descriptive_questions(
    raw_rows: list[dict[str, Any]], validation_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Expand sampled validation scenes while retaining their pinned media identity."""
    media_by_scene: dict[int, dict[str, Any]] = {}
    for row in validation_rows:
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
                    "id": f"clevrer-validation-{scene:05d}-q{question_id:03d}",
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
