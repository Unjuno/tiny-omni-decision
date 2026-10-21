from __future__ import annotations

import pytest

from scripts.prepare_perception_av_temporal_matching_scale import select_grouped_cohorts


def _candidate(video_id: str, label_id: int) -> dict[str, object]:
    return {"video_id": video_id, "label_id": label_id, "payload": video_id}


def test_select_grouped_cohorts_is_deterministic_disjoint_and_nested():
    candidates = {
        0: [_candidate(f"v{i}", 0) for i in range(24)],
        1: [_candidate(f"v{i}", 1) for i in range(24, 48)],
        2: [_candidate(f"v{i}", 2) for i in range(48, 96)],
    }

    first = select_grouped_cohorts(
        candidates, train_count=48, small_train_count=16, validation_count=16, seed=17
    )
    second = select_grouped_cohorts(
        candidates, train_count=48, small_train_count=16, validation_count=16, seed=17
    )

    assert first == second
    assert len(first["train"]) == 48
    assert len(first["train_small"]) == 16
    assert len(first["validation"]) == 16
    train_ids = {row["video_id"] for row in first["train"]}
    small_ids = {row["video_id"] for row in first["train_small"]}
    validation_ids = {row["video_id"] for row in first["validation"]}
    assert small_ids <= train_ids
    assert train_ids.isdisjoint(validation_ids)
    assert len(train_ids | validation_ids) == 64


def test_select_grouped_cohorts_rejects_video_reuse_across_label_buckets():
    candidates = {
        0: [_candidate("shared", 0), _candidate("a", 0)],
        1: [_candidate("shared", 1), _candidate("b", 1)],
    }

    cohorts = select_grouped_cohorts(
        candidates, train_count=1, small_train_count=1, validation_count=2, seed=17
    )

    identifiers = [row["video_id"] for row in cohorts["train"] + cohorts["validation"]]
    assert len(identifiers) == len(set(identifiers))


def test_select_grouped_cohorts_fails_when_unique_video_groups_are_insufficient():
    candidates = {0: [_candidate("only", 0)]}

    with pytest.raises(ValueError, match="unique video groups"):
        select_grouped_cohorts(
            candidates, train_count=1, small_train_count=1, validation_count=1, seed=17
        )
