import pytest

from scripts.plan_perception_av_event_scale import (
    partition_videos,
    select_unused_video_ids,
)


def test_scale_plan_excludes_prior_and_fixed_evaluation_video_ids():
    prior = {f"video_{index}" for index in range(100)}
    evaluated = {f"video_{index}" for index in range(100, 300)}
    candidates = prior | evaluated | {f"video_{index}" for index in range(300, 688)}
    selected = select_unused_video_ids(candidates, prior, evaluated)
    assert len(selected) == 388
    assert selected.isdisjoint(prior)
    assert selected.isdisjoint(evaluated)


def test_scale_partition_is_deterministic_and_video_disjoint():
    ids = {f"video_{index}" for index in range(388)}
    first = partition_videos(ids)
    second = partition_videos(ids)
    assert first == second
    assert {name: len(values) for name, values in first.items()} == {
        "fit": 233,
        "selection": 58,
        "calibration": 39,
        "development_validation": 58,
    }
    for index, left in enumerate(first.values()):
        for right in list(first.values())[index + 1 :]:
            assert set(left).isdisjoint(right)
    assert set().union(*(set(values) for values in first.values())) == ids


def test_scale_plan_fails_closed_on_changed_inventory_or_holdout_overlap():
    with pytest.raises(ValueError, match="missing from the eligible inventory"):
        select_unused_video_ids({f"video_{i}" for i in range(888)}, set(), {"video_999"})
    with pytest.raises(ValueError, match="pinned unused/evaluated video inventory"):
        select_unused_video_ids({"video_1"}, set(), set())
    with pytest.raises(ValueError, match="expected 388"):
        partition_videos({"video_1"})
