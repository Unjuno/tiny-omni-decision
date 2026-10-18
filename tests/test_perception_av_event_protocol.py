import pytest

from scripts.perception_av_event_protocol import (
    deterministic_candidate_order,
    model_selection_video_ids,
)


def test_video_group_selection_is_deterministic_and_exactly_60_of_300():
    video_ids = [f"video_{index}" for index in range(300)]
    selected = model_selection_video_ids(video_ids)
    assert selected == model_selection_video_ids(reversed(video_ids))
    assert len(selected) == 60
    assert len(set(video_ids) - selected) == 240
    assert not selected.intersection(set(video_ids) - selected)


def test_video_group_selection_rejects_duplicate_ids():
    with pytest.raises(ValueError, match="unique"):
        model_selection_video_ids(["video_a", "video_a"], selection_count=1)


def test_candidate_option_order_is_deterministic_per_event():
    order = deterministic_candidate_order("video_1", 8, [2, 5, 9, 11])
    assert order == deterministic_candidate_order("video_1", 8, [11, 2, 9, 5])
    assert sorted(order) == [2, 5, 9, 11]
    assert order != deterministic_candidate_order("video_1", 9, [2, 5, 9, 11])


def test_candidate_order_rejects_invalid_candidates():
    with pytest.raises(ValueError, match="nonempty and unique"):
        deterministic_candidate_order("video_1", 0, [])
    with pytest.raises(ValueError, match="nonnegative integers"):
        deterministic_candidate_order("video_1", 0, [1, -2])
