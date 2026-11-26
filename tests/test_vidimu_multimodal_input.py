"""Input-only checks; no fit, model download or real dataset reads."""

import json

import numpy as np
import pytest

pytest.importorskip("torch")
from scripts import train_vidimu_multimodal_arm as runner


@pytest.fixture
def fixture_cache(tmp_path, monkeypatch):
    folder = tmp_path / "vidimu-video-features-01"
    folder.mkdir()
    rows = [{"subject": "S01", "activity": "A01", "trial": "T01"},
            {"subject": "S02", "activity": "A02", "trial": "T01"}]
    np.savez_compressed(folder / "train.npz", values=np.ones((2, 8, 768), np.float32))
    (folder / "train-records.json").write_text(json.dumps(rows))
    (folder / "result.json").write_text(json.dumps({
        "feature_hashes": {"train": runner.sha(folder / "train.npz")}}))
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "load_sensor", lambda _: (
        np.zeros((2, 360), np.float32), np.array([0, 1]), np.array([0, 1]), rows))
    return folder, rows


@pytest.mark.parametrize("arm,dimension", [("video", 6144), ("joint", 6504)])
def test_aligned_shapes(fixture_cache, arm, dimension):
    values, _, _, rows = runner.load("training", arm)
    assert values.shape == (2, dimension)
    assert len(rows) == 2


def test_reordered_trials_rejected(fixture_cache):
    folder, rows = fixture_cache
    (folder / "train-records.json").write_text(json.dumps(rows[::-1]))
    with pytest.raises(ValueError, match="identities or order"):
        runner.load("training", "joint")


def test_changed_cache_rejected(fixture_cache):
    folder, _ = fixture_cache
    with (folder / "train.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        runner.load("training", "video")
