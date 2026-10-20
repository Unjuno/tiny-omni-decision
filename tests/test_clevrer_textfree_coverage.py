from __future__ import annotations

import pytest

from scripts.prepare_clevrer_textfree_coverage import build_temporal_coverage_rows


def _question(scene: int, question_id: int, question: str, answer: str) -> dict[str, object]:
    return {
        "question_id": question_id,
        "question_type": "descriptive",
        "question_subtype": "query_color",
        "question": question,
        "answer": answer,
        "program": ["events", "filter_collision", "query_color"],
    }


def _scene(scene_id: int) -> dict[str, object]:
    return {
        "scene_index": scene_id,
        "video_filename": f"video_{scene_id:05d}.mp4",
        "questions": [
            _question(scene_id, 0, "What color was it before collision?", "blue"),
            _question(
                scene_id,
                1,
                f"Which color appears after collision in scene {scene_id}?",
                "red",
            ),
            {
                "question_id": 2,
                "question_type": "descriptive",
                "question_subtype": "query_color",
                "question": f"What color is object {scene_id}?",
                "answer": "green",
                "program": ["objects", "query_color"],
            },
        ],
    }


def test_temporal_coverage_split_is_deterministic_grouped_and_content_disjoint():
    raw = [_scene(scene_id) for scene_id in range(10)]
    candidates = set(range(10))

    first = build_temporal_coverage_rows(raw, candidates, seed=17, train_fraction=0.8)
    replay = build_temporal_coverage_rows(raw, candidates, seed=17, train_fraction=0.8)

    assert [row["id"] for row in first["train_rows"]] == [row["id"] for row in replay["train_rows"]]
    assert [row["id"] for row in first["validation_rows"]] == [
        row["id"] for row in replay["validation_rows"]
    ]
    train_scenes = {row["scene_group_id"] for row in first["train_rows"]}
    validation_scenes = {row["scene_group_id"] for row in first["validation_rows"]}
    assert len(train_scenes) == 8
    assert len(validation_scenes) == 2
    assert not train_scenes & validation_scenes
    assert all(row["question_type"] == "temporal_descriptive" for row in first["train_rows"])
    assert all(row["question_type"] == "temporal_descriptive" for row in first["validation_rows"])
    assert all("natural_language_question" not in row for row in first["train_rows"])
    assert first["report"]["normalized_question_option_overlap"] == 0
    assert first["report"]["validation_rows_dropped_for_content_overlap"] == 2


def test_every_candidate_scene_must_have_temporal_rows():
    rows = [_scene(1), _scene(2)]
    rows[1]["questions"] = [rows[1]["questions"][2]]
    with pytest.raises(ValueError, match="candidate scene has no temporal descriptive rows"):
        build_temporal_coverage_rows(rows, {1, 2}, seed=17, train_fraction=0.5)


def test_all_rows_from_a_scene_stay_in_one_split_and_ids_are_stable():
    raw = [_scene(scene_id) for scene_id in range(6)]
    result = build_temporal_coverage_rows(raw, set(range(6)), seed=17, train_fraction=0.5)
    all_rows = result["train_rows"] + result["validation_rows"]
    by_scene: dict[str, set[str]] = {}
    for row in all_rows:
        by_scene.setdefault(str(row["scene_group_id"]), set()).add(str(row["split"]))
        assert row["id"] == (
            f"clevrer-temporal-{int(row['scene_index']):05d}-q{int(row['question_id']):03d}"
        )
    assert all(len(splits) == 1 for splits in by_scene.values())


def test_candidate_scene_set_must_match_source_rows():
    with pytest.raises(ValueError, match="candidate scene IDs are missing"):
        build_temporal_coverage_rows([_scene(1)], {1, 2}, seed=17, train_fraction=0.5)
