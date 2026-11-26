from collections import Counter

import pytest

from tiny_omni_decision.av_correspondence import build_quartets


def test_balanced_correspondence_and_input_order_independence():
    ids = [f"v{i}" for i in range(12)]
    rows = build_quartets(ids, split="train", limit=8)
    assert rows == build_quartets(ids[::-1], split="train", limit=8)
    for start in (0, 4):
        group = rows[start : start + 4]
        assert Counter(r["target_index"] for r in group) == Counter(range(4))
        assert len({r["audio_video_id"] for r in group}) == 4
        assert all(
            set(r["candidate_video_ids"]) == set(group[0]["candidate_video_ids"]) for r in group
        )
        for r in group:
            assert r["candidate_video_ids"][r["target_index"]] == r["audio_video_id"]
        favorite = group[0]["candidate_video_ids"][0]
        assert sum(r["audio_video_id"] == favorite for r in group) == 1


def test_invalid_or_test_split_rejected():
    for ids, split, limit in [
        (["a"] * 4, "train", 4),
        (["a", "b", "c", "d"], "test", 4),
        (["a", "b", "c", "d"], "train", 3),
    ]:
        with pytest.raises(ValueError):
            build_quartets(ids, split=split, limit=limit)
