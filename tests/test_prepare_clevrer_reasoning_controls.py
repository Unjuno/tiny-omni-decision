from __future__ import annotations

import pytest

from scripts.prepare_clevrer_reasoning_controls import select_reasoning_rows
from tiny_omni_decision.clevrer import (
    clevrer_binary_choice_content_key,
    expand_binary_reasoning_choices,
)


def _scene(scene_id: int) -> dict[str, object]:
    questions = []
    for offset, kind in enumerate(("explanatory", "predictive", "counterfactual")):
        question_id = scene_id * 10 + offset
        questions.append(
            {
                "question_id": question_id,
                "question": f"What happens in scene {scene_id} for {kind}?",
                "question_type": kind,
                "choices": [
                    {"choice_id": 0, "choice": f"{kind} candidate A {scene_id}", "answer": "wrong"},
                    {
                        "choice_id": 1,
                        "choice": f"{kind} candidate B {scene_id}",
                        "answer": "correct",
                    },
                ],
            }
        )
    return {
        "scene_index": scene_id,
        "video_filename": f"video_{scene_id:05d}.mp4",
        "questions": questions,
    }


def test_content_disjoint_scene_selection_is_deterministic_and_keeps_task_groups():
    raw = [_scene(scene_id) for scene_id in range(5)]
    prior = expand_binary_reasoning_choices(
        [raw[0]],
        [{"scene_index": 0, "video_filename": raw[0]["video_filename"], "split": "validation"}],
        split="validation",
    )
    excluded = {clevrer_binary_choice_content_key(row) for row in prior}

    selected = select_reasoning_rows(
        raw,
        excluded_scenes={0},
        excluded_content=excluded,
        scene_count=2,
        seed=17,
    )
    replay = select_reasoning_rows(
        raw,
        excluded_scenes={0},
        excluded_content=excluded,
        scene_count=2,
        seed=17,
    )

    assert [row["id"] for row in selected] == [row["id"] for row in replay]
    selected_scenes = {row["scene_index"] for row in selected}
    assert len(selected_scenes) == 2
    assert 0 not in selected_scenes
    assert {row["question_type"] for row in selected} == {
        "explanatory",
        "predictive",
        "counterfactual",
    }
    keys = [clevrer_binary_choice_content_key(row) for row in selected]
    assert not (set(keys) & excluded)
    assert len(keys) == len(set(keys))


def test_selection_fails_if_not_enough_fully_content_disjoint_scenes():
    raw = [_scene(1)]
    rows = expand_binary_reasoning_choices(
        raw,
        [{"scene_index": 1, "video_filename": raw[0]["video_filename"], "split": "validation"}],
        split="validation",
    )
    excluded = {clevrer_binary_choice_content_key(row) for row in rows}

    with pytest.raises(ValueError, match="only 0 content-disjoint scenes"):
        select_reasoning_rows(
            raw,
            excluded_scenes=set(),
            excluded_content=excluded,
            scene_count=1,
        )
