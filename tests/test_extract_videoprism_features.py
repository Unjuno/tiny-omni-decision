from __future__ import annotations

import pytest

from scripts.extract_frozen_videoprism_features import uniform_indices, validate_split_rows


def _row(split: str, scene: int, media_hash: str, question: str) -> dict:
    return {
        "id": f"{split}-{scene}-{question}",
        "split": split,
        "scene_index": scene,
        "media_sha256": media_hash,
        "question": question,
        "options": ["yes", "no"],
    }


def test_uniform_indices_are_deterministic_and_cover_endpoints():
    assert uniform_indices(12, 8) == [0, 2, 3, 5, 6, 8, 9, 11]
    assert uniform_indices(12, 8) == uniform_indices(12, 8)


def test_validate_split_rows_accepts_scene_grouped_independent_splits():
    validate_split_rows(
        [_row("train", 1, "a", "train question")],
        [_row("validation", 2, "b", "validation question")],
    )


@pytest.mark.parametrize("overlap", ["scene", "media", "content"])
def test_validate_split_rows_rejects_cross_split_overlap(overlap):
    train = [_row("train", 1, "a", "same content")]
    validation = [_row("validation", 2, "b", "other content")]
    if overlap == "scene":
        validation[0]["scene_index"] = 1
    elif overlap == "media":
        validation[0]["media_sha256"] = "a"
    else:
        validation[0]["question"] = "SAME CONTENT"

    with pytest.raises(ValueError, match="overlap"):
        validate_split_rows(train, validation)
