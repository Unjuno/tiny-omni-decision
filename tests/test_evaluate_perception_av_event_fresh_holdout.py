import hashlib
import json

import numpy as np
import pytest

from scripts.evaluate_perception_av_event_fresh_holdout import _load_features


def _feature_row(tmp_path, split="fresh_holdout", video_id="video_1"):
    audio_path = tmp_path / f"{video_id}-audio.npy"
    visual_path = tmp_path / f"{video_id}-visual.npy"
    np.save(audio_path, np.ones(512, dtype=np.float32), allow_pickle=False)
    np.save(visual_path, np.ones((2, 257, 384), dtype=np.float32), allow_pickle=False)

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    return {
        "id": f"fresh_holdout:{video_id}:2",
        "video_id": video_id,
        "sound_id": 2,
        "split": split,
        "target_index": 1,
        "candidate_count": 2,
        "candidate_order_sha256": "a" * 64,
        "audio_feature_path": str(audio_path),
        "audio_feature_sha256": digest(audio_path),
        "visual_feature_path": str(visual_path),
        "visual_feature_sha256": digest(visual_path),
    }


def test_fresh_holdout_feature_reader_checks_hashes_shapes_and_target(tmp_path):
    manifest = tmp_path / "features.jsonl"
    rows = [_feature_row(tmp_path, video_id=f"video_{index}") for index in (1, 2)]
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    rows = _load_features(manifest)
    assert len(rows) == 2
    assert rows[0]["audio"].shape == (512,)
    assert rows[0]["visual"].shape == (2, 384)
    assert rows[0]["target_index"] == 1


def test_fresh_holdout_feature_reader_rejects_other_splits(tmp_path):
    manifest = tmp_path / "features.jsonl"
    manifest.write_text(json.dumps(_feature_row(tmp_path, split="fit")) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="only fresh_holdout"):
        _load_features(manifest)
