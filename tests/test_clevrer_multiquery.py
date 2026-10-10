from __future__ import annotations

import pytest

from tiny_omni_decision.clevrer import expand_descriptive_questions


def _question(
    question_id: int, text: str, program: list[str], answer: str = "yes"
) -> dict[str, object]:
    return {
        "question_id": question_id,
        "question": text,
        "question_type": "descriptive",
        "question_subtype": "exist",
        "program": program,
        "answer": answer,
    }


def test_multiquery_expansion_uses_all_supported_descriptive_questions_per_video():
    validation = [
        {
            "scene_index": 21,
            "video_filename": "video_00021.mp4",
            "media_path": "media/validation/video_00021.mp4",
            "media_sha256": "sha-scene-21",
            "media_bytes": 1024,
            "scene_group_id": "clevrer:validation:21",
            "source": "MIT-IBM/CLEVRER@test",
            "split": "validation",
        },
        {
            "scene_index": 21,
            "video_filename": "video_00021.mp4",
            "media_path": "media/validation/video_00021.mp4",
            "media_sha256": "sha-scene-21",
            "media_bytes": 1024,
            "scene_group_id": "clevrer:validation:21",
            "source": "MIT-IBM/CLEVRER@test",
            "split": "validation",
        },
        {
            "scene_index": 22,
            "video_filename": "video_00022.mp4",
            "media_path": "media/validation/video_00022.mp4",
            "media_sha256": "sha-scene-22",
            "media_bytes": 2048,
            "scene_group_id": "clevrer:validation:22",
            "source": "MIT-IBM/CLEVRER@test",
            "split": "validation",
        },
    ]
    raw = [
        {
            "scene_index": 21,
            "video_filename": "video_00021.mp4",
            "questions": [
                _question(2, "Is a green object present?", ["objects", "exist"]),
                {
                    **_question(3, "How many objects move?", ["objects", "count"], "2"),
                    "question_subtype": "count",
                },
                {
                    **_question(4, "What happens next?", ["events"]),
                    "question_type": "predictive",
                    "choices": [{"choice": "event", "answer": "correct"}],
                },
            ],
        },
        {
            "scene_index": 22,
            "video_filename": "video_00022.mp4",
            "questions": [
                _question(0, "Does the blue sphere exist at the end?", ["end", "exist"], "no"),
            ],
        },
    ]

    expanded = expand_descriptive_questions(raw, validation)

    assert [row["id"] for row in expanded] == [
        "clevrer-validation-00021-q002",
        "clevrer-validation-00021-q003",
        "clevrer-validation-00022-q000",
    ]
    assert [row["question_type"] for row in expanded] == [
        "static_descriptive",
        "static_descriptive",
        "temporal_descriptive",
    ]
    assert expanded[0]["media_sha256"] == expanded[1]["media_sha256"]
    assert expanded[0]["scene_group_id"] == expanded[1]["scene_group_id"]
    assert expanded[2]["media_sha256"] == "sha-scene-22"
    assert len({row["id"] for row in expanded}) == len(expanded)


def test_multiquery_expansion_rejects_scene_media_mismatch():
    with pytest.raises(ValueError, match="filename does not match"):
        expand_descriptive_questions(
            [
                {
                    "scene_index": 21,
                    "video_filename": "wrong-video.mp4",
                    "questions": [_question(0, "Is it visible?", ["objects", "exist"])],
                }
            ],
            [
                {
                    "scene_index": 21,
                    "video_filename": "video_00021.mp4",
                    "media_path": "media/validation/video_00021.mp4",
                    "media_sha256": "sha-scene-21",
                    "scene_group_id": "clevrer:validation:21",
                    "split": "validation",
                }
            ],
        )


def test_multiquery_expansion_rejects_duplicate_scene_rows():
    repeated = {"scene_index": 21, "video_filename": "video_00021.mp4", "questions": []}
    with pytest.raises(ValueError, match="duplicate scene rows"):
        expand_descriptive_questions([repeated, repeated], [])
