import pytest

torch = pytest.importorskip("torch")  # noqa: F401
np = pytest.importorskip("numpy")
pytest.importorskip("transformers")
pytest.importorskip("av")

from scripts.diagnose_clevrer_video_multiquery import (  # noqa: E402
    expand_descriptive_questions,
)
from scripts.fetch_clevrer_probe import choose  # noqa: E402
from scripts.train_frozen_video_probe import _uniform_indices, _video_features  # noqa: E402


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


def test_multiquery_expansion_uses_all_supported_descriptive_questions_per_video():
    validation = [
        {
            "scene_index": 21,
            "video_filename": "video_00021.mp4",
            "media_path": "media/validation/video_00021.mp4",
            "media_sha256": "sha-scene-21",
            "media_bytes": 1024,
            "scene_group_id": "clevrer:validation:21",
            "source": "MIT-IBM/CLEVRER@test",
            "split": "validation",
        },
        {
            "scene_index": 21,
            "video_filename": "video_00021.mp4",
            "media_path": "media/validation/video_00021.mp4",
            "media_sha256": "sha-scene-21",
            "media_bytes": 1024,
            "scene_group_id": "clevrer:validation:21",
            "source": "MIT-IBM/CLEVRER@test",
            "split": "validation",
        },
        {
            "scene_index": 22,
            "video_filename": "video_00022.mp4",
            "media_path": "media/validation/video_00022.mp4",
            "media_sha256": "sha-scene-22",
            "media_bytes": 2048,
            "scene_group_id": "clevrer:validation:22",
            "source": "MIT-IBM/CLEVRER@test",
            "split": "validation",
        },
    ]
    raw = [
        {
            "scene_index": 21,
            "video_filename": "video_00021.mp4",
            "questions": [
                _question(2, "Is a green object present?", ["objects", "exist"]),
                {
                    **_question(3, "How many objects move?", ["objects", "count"]),
                    "question_subtype": "count",
                    "answer": "2",
                },
                {
                    **_question(4, "What happens next?", ["events"]),
                    "question_type": "predictive",
                    "choices": [{"choice": "event", "answer": "correct"}],
                },
            ],
        },
        {
            "scene_index": 22,
            "video_filename": "video_00022.mp4",
            "questions": [
                {
                    **_question(0, "Does the blue sphere exist at the end?", ["end", "exist"]),
                    "answer": "no",
                },
            ],
        },
    ]

    expanded = expand_descriptive_questions(raw, validation)

    assert [row["id"] for row in expanded] == [
        "clevrer-validation-00021-q002",
        "clevrer-validation-00021-q003",
        "clevrer-validation-00022-q000",
    ]
    assert [row["question_type"] for row in expanded] == [
        "static_descriptive",
        "static_descriptive",
        "temporal_descriptive",
    ]
    assert expanded[0]["media_sha256"] == expanded[1]["media_sha256"]
    assert expanded[0]["scene_group_id"] == expanded[1]["scene_group_id"]
    assert expanded[2]["media_sha256"] == "sha-scene-22"
    assert len({row["id"] for row in expanded}) == len(expanded)


def test_multiquery_expansion_rejects_scene_media_mismatch():
    with pytest.raises(ValueError, match="filename does not match"):
        expand_descriptive_questions(
            [
                {
                    "scene_index": 21,
                    "video_filename": "wrong-video.mp4",
                    "questions": [_question(0, "Is it visible?", ["objects", "exist"])],
                }
            ],
            [
                {
                    "scene_index": 21,
                    "video_filename": "video_00021.mp4",
                    "media_path": "media/validation/video_00021.mp4",
                    "media_sha256": "sha-scene-21",
                    "scene_group_id": "clevrer:validation:21",
                    "split": "validation",
                }
            ],
        )


def test_video_probe_selects_eight_deterministic_temporal_positions():
    indices = _uniform_indices(64, requested=8)

    assert indices == [0, 9, 18, 27, 36, 45, 54, 63]


def test_video_features_share_one_forward_per_underlying_video(tmp_path, monkeypatch):
    import scripts.train_frozen_video_probe as video_probe

    class Encoder:
        def __init__(self):
            self.calls = 0

        def __call__(self, clip):
            self.calls += 1
            return torch.arange(768, dtype=torch.float32).reshape(1, 1, 768).expand(
                clip.shape[0], 2, 768
            )

    rows = [
        {"id": "question-a", "media_sha256": "same-video", "media_path": "scene.mp4"},
        {"id": "question-b", "media_sha256": "same-video", "media_path": "scene.mp4"},
    ]
    monkeypatch.setattr(
        video_probe, "_decode_video", lambda _path: np.zeros((8, 2, 2, 3), dtype=np.uint8)
    )
    def transform(_frames):
        return (torch.zeros((3, 8, 2, 2)),)

    encoder = Encoder()

    features, _, unique_videos, cache = _video_features(
        rows, tmp_path, encoder, transform, torch.device("cpu")
    )

    assert encoder.calls == 1
    assert unique_videos == 1
    assert list(cache) == ["same-video"]
    assert features.shape == (2, 768)
    assert torch.equal(features[0], features[1])
