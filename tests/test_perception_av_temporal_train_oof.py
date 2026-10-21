import hashlib
import json

import numpy as np
import pytest

pytest.importorskip("torch")

from scripts.audit_perception_av_temporal_train_oof import (  # noqa: E402
    _group_bootstrap,
    assign_group_folds,
    load_train_only,
)


def _save_feature(directory, stem, record, values):
    path = directory / f"{stem}.npy"
    values = np.asarray(values, dtype=np.float32)
    np.save(path, values)
    record["feature_sha256"] = hashlib.sha256(values.tobytes()).hexdigest()
    (directory / f"{stem}.json").write_text(json.dumps(record), encoding="utf-8")


def test_group_folds_are_deterministic_and_keep_each_video_whole():
    rows = [
        {
            "video_id": f"video-{index}",
            "target_index": index % 2,
            "label_id": index % 3,
        }
        for index in range(25)
    ]
    first = assign_group_folds(rows, folds=5, seed=17)
    second = assign_group_folds(rows, folds=5, seed=17)
    assert first == second
    assert set(first) == {row["video_id"] for row in rows}
    assert set(first.values()) == set(range(5))


def test_loader_uses_only_train_manifest_rows_and_train_features(tmp_path):
    video_dir = tmp_path / "video" / "train"
    audio_dir = tmp_path / "audio" / "train"
    video_dir.mkdir(parents=True)
    audio_dir.mkdir(parents=True)
    row = {
        "sample_id": "train:v1:2:3",
        "video_id": "v1",
        "label_id": 4,
        "label_name": "label",
        "target_index": 0,
        "options": [{"event_id": 2}, {"event_id": 3}],
    }
    _save_feature(
        video_dir,
        "v1",
        {"sample_id": row["sample_id"], "video_id": "v1", "split": "train"},
        np.ones((8, 768), dtype=np.float32),
    )
    for event_id in (2, 3):
        _save_feature(
            audio_dir,
            f"v1__{event_id}",
            {
                "sample_id": row["sample_id"],
                "event_id": event_id,
                "split": "train",
            },
            np.full((384,), event_id, dtype=np.float32),
        )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "train": [row],
                "validation": [{"sample_id": "valid:v2", "video_id": "v2", "target_index": 1}],
            }
        ),
        encoding="utf-8",
    )
    loaded = load_train_only(manifest_path, video_dir, audio_dir)
    assert [item["sample_id"] for item in loaded] == [row["sample_id"]]
    assert loaded[0]["video"].shape == (8, 768)
    assert loaded[0]["audio"].shape == (2, 384)


def test_group_bootstrap_is_fixed_seed_and_uses_video_rows():
    scores = np.asarray([[2.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    targets = np.asarray([0, 1, 1])
    first = _group_bootstrap(scores, targets)
    second = _group_bootstrap(scores, targets)
    assert first == second
    assert first["unit"] == "video group; one row per group"
    assert first["replicates"] == 5000
