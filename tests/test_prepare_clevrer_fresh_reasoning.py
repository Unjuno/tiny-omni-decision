from __future__ import annotations

import pytest

from scripts.fetch_clevrer_fresh_reasoning import _selected_partition_names
from scripts.prepare_clevrer_fresh_reasoning import select_fresh_splits, source_scene_ids
from tiny_omni_decision.clevrer import clevrer_binary_choice_content_key


def _scene(scene_id: int, *, split: str = "validation") -> dict[str, object]:
    questions = []
    for offset, kind in enumerate(("explanatory", "predictive", "counterfactual")):
        questions.append(
            {
                "question_id": scene_id * 10 + offset,
                "question": f"What happens in source scene {scene_id} for {kind}?",
                "question_type": kind,
                "choices": [
                    {"choice_id": 0, "choice": f"wrong statement {scene_id}", "answer": "wrong"},
                    {"choice_id": 1, "choice": f"true statement {scene_id}", "answer": "correct"},
                ],
            }
        )
    return {
        "scene_index": scene_id,
        "video_filename": f"video_{scene_id:05d}.mp4",
        "questions": questions,
        "split": split,
    }


def _scene_with_types(scene_id: int, kinds: tuple[str, ...], *, split: str) -> dict[str, object]:
    scene = _scene(scene_id, split=split)
    scene["questions"] = [
        question for question in scene["questions"] if question["question_type"] in kinds
    ]
    return scene


def test_source_scene_identity_uses_media_and_decision_ids_not_unrelated_numbers():
    assert source_scene_ids(
        {
            "scene_index": 7,
            "id": "clevrer-validation-10007-q011-c003",
            "media_path": "media/validation/video_10007.mp4",
            "optimizer_updates": 128,
        }
    ) == {7, 10_007}


def test_training_media_preparation_does_not_load_the_holdout_manifest():
    assert _selected_partition_names(False) == ("train", "development")
    assert _selected_partition_names(True) == ("holdout",)


def test_fresh_split_selection_is_deterministic_and_scene_content_disjoint():
    train = [_scene(i, split="train") for i in range(10)]
    validation = [_scene(i + 10_000) for i in range(10)]
    result = select_fresh_splits(
        train,
        validation,
        excluded_train_scenes={0},
        excluded_validation_scenes={10_000},
        previously_used_content=set(),
        train_scene_count=4,
        development_scene_count=2,
        holdout_scene_count=2,
        seed=17,
    )
    replay = select_fresh_splits(
        train,
        validation,
        excluded_train_scenes={0},
        excluded_validation_scenes={10_000},
        previously_used_content=set(),
        train_scene_count=4,
        development_scene_count=2,
        holdout_scene_count=2,
        seed=17,
    )

    assert result == replay
    train_scenes = {row["scene_index"] for row in result["train"]}
    dev_scenes = {row["scene_index"] for row in result["development"]}
    holdout_scenes = {row["scene_index"] for row in result["holdout"]}
    assert len(train_scenes) == 4
    assert len(dev_scenes) == len(holdout_scenes) == 2
    assert train_scenes.isdisjoint(dev_scenes | holdout_scenes)
    assert dev_scenes.isdisjoint(holdout_scenes)
    assert {row["question_type"] for row in result["train"]} == {
        "explanatory",
        "predictive",
        "counterfactual",
    }
    all_content = [
        clevrer_binary_choice_content_key(row)
        for split in ("train", "development", "holdout")
        for row in result[split]
    ]
    assert len(all_content) == len(set(all_content))


def test_fresh_split_selection_rejects_insufficient_scene_supply():
    with pytest.raises(ValueError, match="not enough eligible scenes"):
        select_fresh_splits(
            [_scene(1, split="train")],
            [_scene(10_001)],
            excluded_train_scenes=set(),
            excluded_validation_scenes=set(),
            previously_used_content=set(),
            train_scene_count=2,
            development_scene_count=1,
            holdout_scene_count=1,
            seed=17,
        )


def test_candidate_content_used_before_selection_is_removed_as_a_whole_question():
    train = [_scene(i, split="train") for i in range(4)]
    validation = [_scene(i + 10_000) for i in range(4)]
    prior = {
        ("what happens in source scene 10000 for predictive?", ("wrong", "correct"))
    }
    selected = select_fresh_splits(
        train,
        validation,
        excluded_train_scenes=set(),
        excluded_validation_scenes=set(),
        previously_used_content=prior,
        train_scene_count=2,
        development_scene_count=1,
        holdout_scene_count=1,
        seed=17,
    )
    assert 10_000 not in {row["scene_index"] for row in selected["development"]}
    assert 10_000 not in {row["scene_index"] for row in selected["holdout"]}


def test_task_types_are_required_per_partition_not_per_scene():
    train = [
        _scene_with_types(i, (kind,), split="train")
        for i, kind in enumerate(("explanatory", "predictive", "counterfactual", "explanatory"))
    ]
    validation = [
        _scene_with_types(10_000 + i, (kind,), split="validation")
        for i, kind in enumerate(
            (
                "explanatory",
                "predictive",
                "counterfactual",
                "explanatory",
                "predictive",
                "counterfactual",
            )
        )
    ]
    selected = select_fresh_splits(
        train,
        validation,
        excluded_train_scenes=set(),
        excluded_validation_scenes=set(),
        previously_used_content=set(),
        train_scene_count=3,
        development_scene_count=3,
        holdout_scene_count=3,
        seed=17,
    )
    for rows in selected.values():
        assert {row["question_type"] for row in rows} == {
            "explanatory",
            "predictive",
            "counterfactual",
        }
