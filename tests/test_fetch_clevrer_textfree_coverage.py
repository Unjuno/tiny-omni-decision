from __future__ import annotations

import pytest

from scripts.fetch_clevrer_textfree_coverage import validate_manifests


def row(sample_id: str, split: str, scene: int, question: str) -> dict[str, object]:
    return {
        "id": sample_id,
        "split": split,
        "scene_index": scene,
        "video_filename": f"video_{scene:05d}.mp4",
        "question": question,
        "options": ["no", "yes"],
    }


def test_manifest_gate_accepts_group_disjoint_content_disjoint_splits() -> None:
    validate_manifests(
        [row("tr-1", "train", 1, "Was there a collision?")],
        [row("va-1", "validation", 2, "Did the object enter?")],
    )


@pytest.mark.parametrize(
    ("train", "validation", "message"),
    [
        (
            [row("tr-1", "train", 1, "Question A")],
            [row("va-1", "validation", 1, "Question B")],
            "scene identity",
        ),
        (
            [row("tr-1", "train", 1, "Question A")],
            [row("va-1", "validation", 2, "Question A")],
            "normalized question/options",
        ),
        (
            [row("tr-1", "train", 1, "Question A")],
            [row("va-1", "train", 2, "Question B")],
            "marked validation",
        ),
    ],
)
def test_manifest_gate_rejects_split_integrity_failures(
    train: list[dict[str, object]],
    validation: list[dict[str, object]],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_manifests(train, validation)
