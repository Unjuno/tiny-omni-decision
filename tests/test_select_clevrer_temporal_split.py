import pytest

from scripts.select_clevrer_temporal_split import (
    select_temporal_split,
    select_temporal_task_mix,
)


def _scene(scene_id: int, n_questions: int = 4) -> dict:
    questions = []
    for index in range(n_questions):
        questions.append(
            {
                "question_id": index,
                "question": (
                    f"Which event occurs before collision in scene {scene_id}, variant {index}?"
                ),
                "question_type": "descriptive",
                "question_subtype": "exist",
                "program": ["filter_collision", "before", f"variant_{index}"],
                "answer": "yes" if index % 2 else "no",
            }
        )
    return {
        "scene_index": scene_id,
        "video_filename": f"video_{scene_id:05d}.mp4",
        "questions": questions,
    }


def test_temporal_split_is_deterministic_scene_disjoint_and_content_disjoint():
    rows = [_scene(i) for i in range(30)]
    first = select_temporal_split(rows, validation_scenes=5, train_examples=20, seed=17)
    second = select_temporal_split(rows, validation_scenes=5, train_examples=20, seed=17)

    assert first == second
    train, validation = first
    train_scenes = {row["scene_index"] for row in train}
    validation_scenes = {row["scene_index"] for row in validation}
    assert train_scenes.isdisjoint(validation_scenes)
    assert len(validation_scenes) == 5
    assert len(train) == 20
    assert len(train_scenes) == 20
    assert all(row["split"] == "train" and row["id"].startswith("clevrer-train-") for row in train)
    assert all(
        row["split"] == "validation" and row["id"].startswith("clevrer-validation-")
        for row in validation
    )
    assert all(row["question_type"] == "temporal_descriptive" for row in train + validation)
    assert {
        (row["question"].casefold(), tuple(sorted(row["options"]))) for row in train
    }.isdisjoint(
        {(row["question"].casefold(), tuple(sorted(row["options"]))) for row in validation}
    )


def test_temporal_split_excludes_previously_observed_scenes():
    train, validation = select_temporal_split(
        [_scene(i) for i in range(20)],
        validation_scenes=4,
        train_examples=8,
        seed=17,
        excluded_scenes={0, 1, 2, 3},
    )

    assert {row["scene_index"] for row in train + validation}.isdisjoint({0, 1, 2, 3})


def test_temporal_split_fails_closed_when_quotas_cannot_be_met():
    with pytest.raises(ValueError, match="cannot satisfy"):
        select_temporal_split(
            [_scene(0), _scene(1)],
            validation_scenes=2,
            train_examples=1,
            seed=17,
        )


def test_temporal_split_rejects_duplicate_scene_records():
    with pytest.raises(ValueError, match="duplicate CLEVRER scene"):
        select_temporal_split(
            [_scene(1), _scene(1)],
            validation_scenes=0,
            train_examples=1,
            seed=17,
        )


def _mixed_scene(scene_id: int) -> dict:
    return {
        "scene_index": scene_id,
        "video_filename": f"video_{scene_id:05d}.mp4",
        "questions": [
            {
                "question_id": 0,
                "question": f"What happens before impact in scene {scene_id}?",
                "question_type": "descriptive",
                "question_subtype": "exist",
                "program": ["filter_collision", "before"],
                "answer": "yes",
            },
            *[
                {
                    "question_id": index,
                    "question": f"{kind} question for scene {scene_id}?",
                    "question_type": kind,
                    "program": ["events", "before", kind],
                    "choices": [
                        {
                            "choice_id": 0,
                            "choice": f"correct event {kind} in {scene_id}",
                            "answer": "correct",
                        },
                        {
                            "choice_id": 1,
                            "choice": f"wrong event {kind} in {scene_id}",
                            "answer": "wrong",
                        },
                    ],
                }
                for index, kind in enumerate(
                    ("explanatory", "predictive", "counterfactual"), start=1
                )
            ],
        ],
    }


def test_mixed_temporal_selector_balances_tasks_and_binary_labels_without_leakage():
    train, validation = select_temporal_task_mix(
        [_mixed_scene(i) for i in range(40)],
        validation_scenes_per_type=2,
        train_example_counts={
            "temporal_descriptive": 4,
            "explanatory": 2,
            "predictive": 2,
            "counterfactual": 2,
        },
        seed=17,
        excluded_scenes={0, 1},
    )

    train_scenes = {row["scene_index"] for row in train}
    validation_scenes = {row["scene_index"] for row in validation}
    assert train_scenes.isdisjoint(validation_scenes | {0, 1})
    assert validation_scenes.isdisjoint({0, 1})
    assert len(validation_scenes) == 8
    assert {row["question_type"] for row in train} == {
        "temporal_descriptive",
        "explanatory",
        "predictive",
        "counterfactual",
    }
    assert {row["question_type"] for row in validation} == {
        "temporal_descriptive",
        "explanatory",
        "predictive",
        "counterfactual",
    }
    assert (
        sum(
            row["target"] == "correct"
            for row in train
            if row["taxonomy"] == "binary_choice_truth"
        )
        == 3
    )
    assert (
        sum(
            row["target"] == "wrong"
            for row in train
            if row["taxonomy"] == "binary_choice_truth"
        )
        == 3
    )

    def key(row):
        return (
            " ".join(row["question"].casefold().split()),
            tuple(sorted(" ".join(option.casefold().split()) for option in row["options"])),
        )
    assert {key(row) for row in train}.isdisjoint({key(row) for row in validation})


def test_mixed_temporal_selector_fails_closed_when_a_task_quota_is_unavailable():
    with pytest.raises(ValueError, match="cannot satisfy"):
        select_temporal_task_mix(
            [_mixed_scene(0)],
            validation_scenes_per_type=1,
            train_example_counts={
                "temporal_descriptive": 2,
                "explanatory": 2,
                "predictive": 2,
                "counterfactual": 2,
            },
            seed=17,
        )
