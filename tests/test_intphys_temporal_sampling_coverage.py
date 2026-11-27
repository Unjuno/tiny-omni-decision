from __future__ import annotations

import pytest

from scripts.audit_intphys_temporal_sampling_coverage import (
    summarize_group_collisions,
    uniform_frame_numbers,
)


def test_uniform_frame_numbers_preserves_endpoints_and_expected_eight_frame_policy() -> None:
    assert uniform_frame_numbers(100, 8) == (1, 15, 29, 43, 57, 71, 85, 100)
    assert uniform_frame_numbers(5, 5) == (1, 2, 3, 4, 5)
    assert uniform_frame_numbers(5, 1) == (1,)


@pytest.mark.parametrize("count", [8, 16, 25, 32, 50, 100])
def test_predeclared_frame_counts_are_unique_and_endpoint_inclusive(count: int) -> None:
    positions = uniform_frame_numbers(100, count)
    assert len(positions) == count
    assert len(set(positions)) == count
    assert positions[0] == 1
    assert positions[-1] == 100


@pytest.mark.parametrize(
    ("total", "requested"),
    [(0, 1), (5, 0), (5, 6)],
)
def test_uniform_frame_numbers_rejects_invalid_counts(total: int, requested: int) -> None:
    with pytest.raises(ValueError):
        uniform_frame_numbers(total, requested)


def test_collision_summary_counts_only_exact_opposite_label_identity_within_group() -> None:
    rows = {
        "O1/01": [
            (True, "same"),
            (True, "possible-2"),
            (False, "same"),
            (False, "impossible-2"),
        ],
        "O1/02": [
            (True, "shared-across-scenes"),
            (True, "possible-4"),
            (False, "other-3"),
            (False, "other-4"),
        ],
    }
    result = summarize_group_collisions(rows)
    assert result["scene_groups"] == 2
    assert result["cross_label_pairs"] == 8
    assert result["groups_with_exact_cross_label_sequence_collision"] == 1
    assert result["exact_cross_label_sequence_collision_pairs"] == 1
    assert result["colliding_group_ids"] == ["O1/01"]


def test_collision_summary_rejects_unbalanced_or_incomplete_scene_groups() -> None:
    with pytest.raises(ValueError, match="four rows"):
        summarize_group_collisions({"O1/01": [(True, "a"), (False, "b")]})
    with pytest.raises(ValueError, match="2/2 label balance"):
        summarize_group_collisions(
            {"O1/01": [(True, "a"), (True, "b"), (True, "c"), (False, "d")]}
        )
