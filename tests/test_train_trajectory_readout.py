import json

import pytest

pytest.importorskip("torch")
from scripts.train_trajectory_readout import load_inputs  # noqa: E402


def test_incomplete_extraction_cannot_start_training(tmp_path, monkeypatch):
    import scripts.train_trajectory_readout as module

    root = tmp_path / "root"
    pairs = root / "perception-occlusion-pairs-01"
    pairs.mkdir(parents=True)
    for split in ("train", "valid"):
        (pairs / f"{split}.json").write_text(json.dumps({"pairs": []}))
    monkeypatch.setattr(module, "ROOT", root)
    monkeypatch.setattr(
        module, "PINS", {split: module.sha(pairs / f"{split}.json") for split in ("train", "valid")}
    )
    video, text = tmp_path / "video", tmp_path / "text"
    video.mkdir()
    text.mkdir()
    (video / "result.json").write_text(json.dumps({"status": "RUNNING"}))
    (text / "result.json").write_text(json.dumps({"status": "PASS_FROZEN_TEXT_FEATURES"}))
    with pytest.raises(ValueError, match="complete video feature extraction required"):
        load_inputs(video, text)
