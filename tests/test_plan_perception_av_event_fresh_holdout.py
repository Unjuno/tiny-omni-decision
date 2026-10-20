import pytest

from scripts.plan_perception_av_event_fresh_holdout import select_holdout_ids


def test_holdout_selection_is_deterministic_and_excludes_previous_ids():
    candidates = {f"video_{index}" for index in range(100)}
    used = {f"video_{index}" for index in range(40)}
    first = select_holdout_ids(candidates, used, count=25)
    second = select_holdout_ids(candidates, used, count=25)
    assert first == second
    assert len(first) == len(set(first)) == 25
    assert set(first).isdisjoint(used)


def test_holdout_selection_fails_closed_when_not_enough_fresh_ids():
    with pytest.raises(ValueError, match="need 3 fresh TRAIN videos; found 2"):
        select_holdout_ids({"video_1", "video_2"}, set(), count=3)


@pytest.mark.parametrize("count", [0, -1])
def test_holdout_selection_rejects_nonpositive_count(count):
    with pytest.raises(ValueError, match="positive"):
        select_holdout_ids({"video_1"}, set(), count=count)
