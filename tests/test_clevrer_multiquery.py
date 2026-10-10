from __future__ import annotations

import pytest

from tiny_omni_decision.clevrer import (
    clevrer_binary_choice_content_key,
    expand_binary_reasoning_choices,
    expand_descriptive_questions,
)


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


def test_binary_reasoning_expansion_maps_each_choice_to_one_decision():
    raw = [
        {
            "scene_index": 21,
            "video_filename": "video_00021.mp4",
            "questions": [
                {
                    "question_id": 7,
                    "question": "What will happen next?",
                    "question_type": "predictive",
                    "choices": [
                        {"choice_id": 0, "choice": "The red ball moves.", "answer": "correct"},
                        {"choice_id": 1, "choice": "The blue cube moves.", "answer": "wrong"},
                    ],
                },
                {
                    "question_id": 8,
                    "question": "Why did it happen?",
                    "question_type": "explanatory",
                    "choices": [
                        {"choice_id": 0, "choice": "Because of a collision.", "answer": "wrong"},
                    ],
                },
                {
                    "question_id": 9,
                    "question": "Without the cube, what would occur?",
                    "question_type": "counterfactual",
                    "choices": [
                        {"choice_id": 1, "choice": "The sphere exits.", "answer": "correct"},
                    ],
                },
                _question(10, "Describe the scene.", []),
            ],
        }
    ]
    sampled = [
        {
            "scene_index": 21,
            "video_filename": "video_00021.mp4",
            "media_path": "media/train/video_00021.mp4",
            "media_sha256": "a" * 64,
            "scene_group_id": "clevrer:train:21",
            "split": "train",
        }
    ]

    expanded = expand_binary_reasoning_choices(raw, sampled, split="train")

    assert [row["id"] for row in expanded] == [
        "clevrer-train-00021-q007-c000",
        "clevrer-train-00021-q007-c001",
        "clevrer-train-00021-q008-c000",
        "clevrer-train-00021-q009-c001",
    ]
    assert [row["question_type"] for row in expanded] == [
        "predictive",
        "predictive",
        "explanatory",
        "counterfactual",
    ]
    assert [row["target"] for row in expanded] == ["correct", "wrong", "wrong", "correct"]
    assert all(row["options"] == ["wrong", "correct"] for row in expanded)
    assert all("Candidate statement:" in row["question"] for row in expanded)
    assert all(row["scene_group_id"] == "clevrer:train:21" for row in expanded)


def test_binary_reasoning_expansion_filters_exact_cross_split_content():
    question = {
        "question_id": 2,
        "question": "Which event occurs?",
        "question_type": "explanatory",
        "choices": [
            {"choice_id": 0, "choice": "The red sphere moves.", "answer": "correct"},
            {"choice_id": 1, "choice": "The blue cube moves.", "answer": "wrong"},
        ],
    }
    raw = [{"scene_index": 31, "video_filename": "video_00031.mp4", "questions": [question]}]
    sampled = [
        {
            "scene_index": 31,
            "video_filename": "video_00031.mp4",
            "media_path": "media/validation/video_00031.mp4",
            "media_sha256": "b" * 64,
            "scene_group_id": "clevrer:validation:31",
            "split": "validation",
        }
    ]
    previous = {
        "question": "Which event occurs?\nCandidate statement: The red sphere moves.",
        "options": ["wrong", "correct"],
    }
    excluded = {clevrer_binary_choice_content_key(previous)}

    expanded = expand_binary_reasoning_choices(
        raw, sampled, split="validation", excluded_content=excluded
    )

    assert len(expanded) == 1
    assert expanded[0]["candidate_statement"] == "The blue cube moves."


@pytest.mark.parametrize("answer", ["yes", "", None])
def test_binary_reasoning_expansion_rejects_unknown_choice_labels(answer):
    raw = [
        {
            "scene_index": 21,
            "video_filename": "video_00021.mp4",
            "questions": [
                {
                    "question_id": 7,
                    "question": "What next?",
                    "question_type": "predictive",
                    "choices": [
                        {"choice_id": 0, "choice": "The red ball moves.", "answer": answer}
                    ],
                }
            ],
        }
    ]
    sampled = [
        {
            "scene_index": 21,
            "video_filename": "video_00021.mp4",
            "media_path": "media/train/video_00021.mp4",
            "media_sha256": "c" * 64,
            "scene_group_id": "clevrer:train:21",
            "split": "train",
        }
    ]

    with pytest.raises(ValueError, match="choice answer must be 'correct' or 'wrong'"):
        expand_binary_reasoning_choices(raw, sampled, split="train")
