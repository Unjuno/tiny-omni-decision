import pytest

from scripts.plan_perception_av_event_media import select_fresh_ids, stable_order


def test_video_selection_is_hash_deterministic_and_excludes_seen_assets():
    candidates = [f"video_{index}" for index in range(20)]
    existing = {"video_1", "video_4", "video_7"}
    selected = select_fresh_ids(candidates, existing, "train", 8)
    assert selected == select_fresh_ids(list(reversed(candidates)), existing, "train", 8)
    assert len(selected) == len(set(selected)) == 8
    assert not (set(selected) & existing)


def test_selection_is_split_and_purpose_specific():
    ids = [f"video_{index}" for index in range(20)]
    assert stable_order(ids, "train", "acquisition") != stable_order(ids, "valid", "acquisition")
    assert stable_order(ids, "train", "acquisition") != stable_order(ids, "train", "model-split")


def test_selection_fails_when_not_enough_unseen_eligible_videos_exist():
    with pytest.raises(ValueError, match="need 4 fresh valid"):
        select_fresh_ids(["video_1", "video_2"], {"video_1"}, "valid", 4)
