from __future__ import annotations

import pytest

from tiny_omni_decision.perception_test_dataset import (
    assert_video_id_disjoint,
    build_perception_test_video_qa,
)


def _video(split: str, video_id: str, questions: list[dict[str, object]]) -> dict[str, object]:
    return {
        "metadata": {"split": split, "video_id": video_id},
        "mc_question": questions,
    }


def _question(question_id: int, answer_id: int = 1) -> dict[str, object]:
    return {
        "id": question_id,
        "question": "Which option matches the observed event?",
        "options": ["first", "second", "third"],
        "answer_id": answer_id,
        "area": "Memory",
        "reasoning": "descriptive",
        "tag": ["event recall"],
    }


def test_video_qa_preserves_three_options_and_video_group_identity() -> None:
    rows = {
        "video_4": _video("train", "video_4", [_question(7), _question(9, 2)]),
        "video_8": _video("train", "video_8", [_question(2, 0)]),
    }
    result = build_perception_test_video_qa(
        rows,
        split="train",
        video_paths={"video_4": "train/video_4.mp4", "video_8": "train/video_8.mp4"},
    )

    assert [row.source_record_id for row in result] == [
        "video_4/7",
        "video_4/9",
        "video_8/2",
    ]
    assert [row.target for row in result] == ["B: second", "C: third", "A: first"]
    assert all(row.modality == "video" for row in result)
    assert all(len(row.options) == 3 for row in result)
    assert all(row.provenance.license == "CC-BY-4.0" for row in result)
    assert result[0].media[0].path == "train/video_4.mp4"


def test_rejects_video_id_split_overlap() -> None:
    with pytest.raises(ValueError, match="video ID overlap"):
        assert_video_id_disjoint(["video_1", "video_2"], ["video_2", "video_3"])


def test_rejects_wrong_split_or_metadata_video_id() -> None:
    with pytest.raises(ValueError, match="split metadata"):
        build_perception_test_video_qa(
            {"video_1": _video("valid", "video_1", [_question(1)])},
            split="train",
            video_paths={"video_1": "train/video_1.mp4"},
        )

    with pytest.raises(ValueError, match="video ID metadata"):
        build_perception_test_video_qa(
            {"video_1": _video("train", "video_2", [_question(1)])},
            split="train",
            video_paths={"video_1": "train/video_1.mp4"},
        )


def test_rejects_heldout_test_and_invalid_answer_index() -> None:
    with pytest.raises(ValueError, match="only train and valid"):
        build_perception_test_video_qa(
            {}, split="test", video_paths={}
        )

    with pytest.raises(ValueError, match="answer_id"):
        build_perception_test_video_qa(
            {"video_1": _video("train", "video_1", [_question(1, 3)])},
            split="train",
            video_paths={"video_1": "train/video_1.mp4"},
        )


def test_rejects_missing_path_and_unsafe_path() -> None:
    row = {"video_1": _video("train", "video_1", [_question(1)])}
    with pytest.raises(ValueError, match="video path is missing"):
        build_perception_test_video_qa(row, split="train", video_paths={})

    with pytest.raises(ValueError, match="relative"):
        build_perception_test_video_qa(
            row, split="train", video_paths={"video_1": "../video_1.mp4"}
        )


def test_preserves_answer_index_for_duplicate_source_option_text() -> None:
    question = _question(3, answer_id=2)
    question["options"] = ["same distractor", "same distractor", "correct"]
    result = build_perception_test_video_qa(
        {"video_1": _video("valid", "video_1", [question])},
        split="valid",
        video_paths={"video_1": "valid/video_1.mp4"},
    )

    assert result[0].options == [
        "A: same distractor",
        "B: same distractor",
        "C: correct",
    ]
    assert result[0].target == "C: correct"
    assert result[0].source_target == 2


def test_rejects_duplicate_question_ids_and_non_mapping_question() -> None:
    duplicate_id = [_question(1), _question(1, 2)]
    with pytest.raises(ValueError, match="duplicate question id"):
        build_perception_test_video_qa(
            {"video_1": _video("train", "video_1", duplicate_id)},
            split="train",
            video_paths={"video_1": "train/video_1.mp4"},
        )

    with pytest.raises(ValueError, match="question annotation must be an object"):
        build_perception_test_video_qa(
            {"video_1": _video("train", "video_1", [None])},  # type: ignore[list-item]
            split="train",
            video_paths={"video_1": "train/video_1.mp4"},
        )
