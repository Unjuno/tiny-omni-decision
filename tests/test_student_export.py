from __future__ import annotations

import pytest

from tiny_omni_decision.schema import DecisionExample
from tiny_omni_decision.student_export import align_teacher_predictions


def make_example(sample_id: str, options: list[str], target: str) -> DecisionExample:
    return DecisionExample.model_validate(
        {
            "id": sample_id,
            "modality": "text",
            "state": "",
            "question": "Choose one",
            "options": options,
            "target": target,
            "media": [],
            "source": "fixture",
            "source_revision": "a" * 40,
            "source_record_id": sample_id,
            "split": "evaluation",
            "source_target": target,
            "provenance": {},
        }
    )


def test_align_teacher_predictions_joins_by_id_and_preserves_choice_order() -> None:
    examples = [
        make_example("one", ["red", "blue", "green"], "blue"),
        make_example("two", ["left", "right"], "left"),
    ]
    teacher = [
        {
            "sample_id": "two",
            "source": "fixture",
            "modality": "text",
            "target": 0,
            "option_labels": ["A", "B"],
            "option_probabilities": [0.8, 0.2],
        },
        {
            "sample_id": "one",
            "source": "fixture",
            "modality": "text",
            "target": 1,
            "option_labels": ["A", "B", "C"],
            "option_probabilities": [0.1, 0.7, 0.2],
        },
    ]

    paired = align_teacher_predictions(examples, teacher)

    assert [row["sample_id"] for row in paired] == ["one", "two"]
    assert paired[0]["options"] == ["red", "blue", "green"]
    assert paired[0]["target"] == 1
    assert paired[0]["option_probabilities"] == [0.1, 0.7, 0.2]


@pytest.mark.parametrize(
    ("bad_teacher", "match"),
    [
        ([{"sample_id": "wrong"}], "ID"),
        (
            [
                {
                    "sample_id": "one",
                    "source": "fixture",
                    "modality": "text",
                    "target": 0,
                    "option_labels": ["A", "B"],
                    "option_probabilities": [0.5, 0.5],
                }
            ],
            "metadata",
        ),
    ],
)
def test_align_teacher_predictions_fails_closed(bad_teacher, match) -> None:
    example = make_example("one", ["red", "blue", "green"], "blue")
    with pytest.raises(ValueError, match=match):
        align_teacher_predictions([example], bad_teacher)


def test_align_teacher_predictions_rejects_duplicate_teacher_ids() -> None:
    example = make_example("one", ["yes", "no"], "yes")
    row = {
        "sample_id": "one",
        "source": "fixture",
        "modality": "text",
        "target": 0,
        "option_labels": ["A", "B"],
        "option_probabilities": [0.6, 0.4],
    }
    with pytest.raises(ValueError, match="unique"):
        align_teacher_predictions([example], [row, row])
