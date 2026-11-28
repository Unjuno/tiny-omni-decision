import random

import pytest

from scripts.build_physionpp_blinded_review_pack import (
    ANNOTATION_COLUMNS,
    _randomized_target_order,
    _selected_frame_indices,
)


def test_selected_frame_indices_are_evenly_spaced_and_include_endpoints():
    assert _selected_frame_indices(724, 8) == {
        0,
        103,
        207,
        310,
        413,
        516,
        620,
        723,
    }


@pytest.mark.parametrize(
    ("frame_count", "count"),
    [(1, 8), (10, 1), (4, 5)],
)
def test_selected_frame_indices_reject_incompatible_counts(frame_count, count):
    with pytest.raises(ValueError, match="incompatible"):
        _selected_frame_indices(frame_count, count)


def test_randomized_pair_sides_are_deterministic_and_balanced_per_pair():
    first = _randomized_target_order(random.Random(19))
    repeated = _randomized_target_order(random.Random(19))

    assert first == repeated
    assert set(first) == {"No", "Yes"}


def test_annotation_template_has_blinded_observation_fields():
    assert "left_contact_observed" in ANNOTATION_COLUMNS
    assert "right_contact_observed" in ANNOTATION_COLUMNS
    assert "uncertain_or_occluded" in ANNOTATION_COLUMNS
    assert "target" not in ANNOTATION_COLUMNS
