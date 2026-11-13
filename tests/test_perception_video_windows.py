import pytest

from tiny_omni_decision.perception_video_windows import plan_frame_windows


def test_overlapping_eight_frame_windows_cover_every_annotated_frame_once():
    plan = plan_frame_windows(list(range(19)))
    assert plan["windows"] == [
        list(range(0, 8)),
        list(range(4, 12)),
        list(range(8, 16)),
        list(range(11, 19)),
    ]
    assert set(plan["frame_to_window_slot"]) == set(range(19))
    for frame_id, (window_index, slot_index) in plan["frame_to_window_slot"].items():
        assert plan["windows"][window_index][slot_index] == frame_id
    assert plan["frame_to_window_slot"][0] == (0, 0)
    assert plan["frame_to_window_slot"][18] == (3, 7)


def test_short_frame_list_repeats_last_frame_only_for_padding():
    plan = plan_frame_windows([2, 5, 9])
    assert plan["windows"] == [[2, 5, 9, 9, 9, 9, 9, 9]]
    assert plan["frame_to_window_slot"] == {2: (0, 0), 5: (0, 1), 9: (0, 2)}


def test_frame_window_planning_rejects_empty_duplicate_or_unsorted_ids():
    for frame_ids in ([], [3, 3], [4, 2]):
        with pytest.raises(ValueError, match="unique increasing"):
            plan_frame_windows(frame_ids)
