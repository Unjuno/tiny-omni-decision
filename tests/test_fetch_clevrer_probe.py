import pytest

torch = pytest.importorskip("torch")  # noqa: F401
np = pytest.importorskip("numpy")
pytest.importorskip("transformers")
pytest.importorskip("av")

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


def test_scene_selector_excludes_previously_consumed_scenes_deterministically():
    excluded = {3, 4, 5}
    selected = choose(
        _rows(), "train", per_type=2, seed=17, excluded_scenes=excluded
    )
    repeated = choose(
        _rows(), "train", per_type=2, seed=17, excluded_scenes=excluded
    )

    scenes = {row["scene_index"] for row in selected}
    assert scenes == {row["scene_index"] for row in repeated}
    assert scenes.isdisjoint(excluded)
    assert scenes.isdisjoint({0, 1, 2})


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
