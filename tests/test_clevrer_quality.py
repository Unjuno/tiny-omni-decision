from __future__ import annotations

from pathlib import Path

import pytest

from tiny_omni_decision.clevrer_quality import (
    materialize_clevrer_examples,
    media_path_for_scene,
    select_unseen_clevrer_scenes,
)
from tiny_omni_decision.dataset import check_train_eval_splits
from tiny_omni_decision.io import load_structured_file
from tiny_omni_decision.schema import DatasetManifest, MediaRef

ROOT = Path(__file__).resolve().parents[1]


def _manifest() -> DatasetManifest:
    raw = load_structured_file(ROOT / "manifests/candidates/clevrer-video-native.yaml")
    return DatasetManifest.model_validate(raw)


def _row(scene: int) -> dict[str, object]:
    return {
        "scene_index": scene,
        "video_filename": f"video_{scene:05}.mp4",
        "questions": [
            {
                "question_id": 4,
                "question": f"What happens in scene {scene}?",
                "question_type": "explanatory",
                "choices": [
                    {
                        "choice_id": 0,
                        "choice": f"the objects collide in scene {scene}",
                        "answer": "correct",
                    },
                    {
                        "choice_id": 1,
                        "choice": f"the objects never move in scene {scene}",
                        "answer": "wrong",
                    },
                ],
            }
        ],
    }


def test_fresh_scene_selection_is_deterministic_and_excludes_seen_scenes() -> None:
    rows = [_row(scene) for scene in range(12)]
    kwargs = {
        "previously_observed": {0, 1, 2, 3},
        "train_scene_count": 4,
        "validation_scene_count": 3,
        "seed": 17,
    }
    first = select_unseen_clevrer_scenes(rows, **kwargs)
    second = select_unseen_clevrer_scenes(rows, **kwargs)
    assert first == second
    train, validation = first
    assert len(train) == 4
    assert len(validation) == 3
    assert train.isdisjoint(validation)
    assert not (train | validation) & kwargs["previously_observed"]


def test_fresh_clevrer_examples_keep_every_scene_in_one_split() -> None:
    rows = [_row(20), _row(21)]
    media = {
        scene: MediaRef(
            kind="video",
            path=media_path_for_scene(scene),
            sha256=f"{scene:064x}",
            license="CC0-1.0",
        )
        for scene in (20, 21)
    }
    train, validation, report = materialize_clevrer_examples(
        rows,
        train_scenes={20},
        validation_scenes={21},
        manifest=_manifest(),
        media_by_scene=media,
    )
    assert {item.split for item in train} == {"train"}
    assert {item.split for item in validation} == {"validation"}
    assert {item.source_record_id.split(":", 1)[0] for item in train} == {"20"}
    assert {item.source_record_id.split(":", 1)[0] for item in validation} == {"21"}
    assert {item.task_group_id for item in train} == {"20:4"}
    assert {item.task_group_id for item in validation} == {"21:4"}
    assert all(item.media[0].sha256 for item in train + validation)
    assert report["train"]["unique_scenes"] == 1
    assert report["validation"]["unique_scenes"] == 1
    assert check_train_eval_splits(train, validation)["status"] == "disjoint"


def test_fresh_scene_selection_rejects_insufficient_unseen_data() -> None:
    with pytest.raises(ValueError, match="unseen CLEVRER scenes"):
        select_unseen_clevrer_scenes(
            [_row(0), _row(1)],
            previously_observed={0},
            train_scene_count=1,
            validation_scene_count=1,
            seed=17,
        )


def test_fresh_clevrer_builder_rejects_duplicate_scene_rows() -> None:
    row = _row(30)
    with pytest.raises(ValueError, match="repeats scene"):
        materialize_clevrer_examples(
            [row, row],
            train_scenes={30},
            validation_scenes={31},
            manifest=_manifest(),
            media_by_scene={
                30: MediaRef(
                    kind="video",
                    path=media_path_for_scene(30),
                    sha256="a" * 64,
                    license="CC0-1.0",
                ),
                31: MediaRef(
                    kind="video",
                    path=media_path_for_scene(31),
                    sha256="b" * 64,
                    license="CC0-1.0",
                ),
            },
        )
