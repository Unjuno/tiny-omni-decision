from scripts.fetch_clevrer_probe import choose
from scripts.train_frozen_video_probe import _uniform_indices


def _question(question_id: int, text: str, program: list[str]) -> dict[str, object]:
    return {
        "question_id": question_id,
        "question": text,
        "question_type": "descriptive",
        "question_subtype": "exist",
        "program": program,
        "answer": "yes",
    }


def _rows() -> list[dict[str, object]]:
    return [
        {
            "scene_index": scene,
            "video_filename": f"video_{scene:05d}.mp4",
            "questions": [
                _question(0, f"Are objects visible in scene {scene}?", ["objects", "exist"]),
                _question(
                    1,
                    f"What happens before collision in scene {scene}?",
                    ["filter_collision", "before"],
                ),
            ],
        }
        for scene in range(8)
    ]


def test_clevrer_probe_is_deterministic_scene_grouped_and_excludes_old_scenes():
    selected = choose(_rows(), "train", per_type=2, seed=17)
    repeated = choose(_rows(), "train", per_type=2, seed=17)

    assert [(row["scene_index"], row["question_id"]) for row in selected] == [
        (row["scene_index"], row["question_id"]) for row in repeated
    ]
    scenes = {row["scene_index"] for row in selected}
    assert len(scenes) == 2
    assert not ({0, 1, 2} & {row["scene_index"] for row in selected})
    assert {row["question_type"] for row in selected} == {
        "static_descriptive",
        "temporal_descriptive",
    }
    for scene in scenes:
        assert {row["question_type"] for row in selected if row["scene_index"] == scene} == {
            "temporal_descriptive",
            "static_descriptive",
        }


def test_validation_scene_selector_excludes_train_question_content():
    train = choose(_rows(), "train", per_type=2, seed=17)
    train_content = {
        (row["question"].casefold().strip(), tuple(sorted(row["options"]))) for row in train
    }
    validation = choose(_rows(), "validation", per_type=2, seed=17, excluded_content=train_content)
    validation_content = {
        (row["question"].casefold().strip(), tuple(sorted(row["options"]))) for row in validation
    }

    assert not train_content & validation_content


def test_video_probe_selects_eight_deterministic_temporal_positions():
    indices = _uniform_indices(64, requested=8)

    assert indices == [0, 9, 18, 27, 36, 45, 54, 63]
