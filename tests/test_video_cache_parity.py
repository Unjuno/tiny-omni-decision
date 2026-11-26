from __future__ import annotations

import pytest

from scripts.verify_video_cache_parity import _select_validation_scenes


def _row(scene: int, question: str, media_hash: str | None = None) -> dict[str, object]:
    return {
        "scene_index": scene,
        "id": question,
        "media_sha256": media_hash or f"hash-{scene}",
    }


def test_select_validation_scenes_groups_and_orders_questions():
    rows = [
        _row(8, "q8-a"),
        _row(3, "q3-a"),
        _row(8, "q8-b"),
        _row(3, "q3-b"),
        _row(5, "singleton"),
    ]

    groups = _select_validation_scenes(rows, scene_limit=2)

    assert [(scene, [row["id"] for row in group]) for scene, group in groups] == [
        (3, ["q3-a", "q3-b"]),
        (8, ["q8-a", "q8-b"]),
    ]


def test_select_validation_scenes_preserves_single_scene_default():
    rows = [_row(8, "q8-a"), _row(8, "q8-b"), _row(3, "q3-a"), _row(3, "q3-b")]

    groups = _select_validation_scenes(rows, scene_limit=1)

    assert [scene for scene, _ in groups] == [3]


def test_select_validation_scenes_rejects_inconsistent_scene_media():
    rows = [_row(3, "q3-a", "hash-a"), _row(3, "q3-b", "hash-b")]

    with pytest.raises(ValueError, match="inconsistent video hashes"):
        _select_validation_scenes(rows, scene_limit=1)


def test_select_validation_scenes_requires_a_positive_limit():
    with pytest.raises(ValueError, match="at least 1"):
        _select_validation_scenes([_row(3, "q3-a"), _row(3, "q3-b")], scene_limit=0)
