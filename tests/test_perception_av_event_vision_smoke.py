import pytest

from scripts.audit_perception_av_event_vision_smoke import (
    normalized_box_to_pixels,
    track_box_at_frame,
    validate_selected_frame,
)


def test_normalized_xyxy_box_maps_to_containing_pixel_crop():
    assert normalized_box_to_pixels([0.1, 0.2, 0.25, 0.4], 100, 200) == (
        20,
        20,
        50,
        40,
    )


@pytest.mark.parametrize(
    "box",
    [
        [0.2, 0.2, 0.2, 0.4],
        [0.2, 0.4, 0.3, 0.3],
        [-0.1, 0.0, 0.5, 0.5],
        [0.1, 0.1, 1.1, 0.9],
        [0.0, 0.0, float("nan"), 1.0],
    ],
)
def test_invalid_normalized_crop_fails_closed(box):
    with pytest.raises(ValueError):
        normalized_box_to_pixels(box, 100, 200)


def test_track_crop_requires_one_unmasked_aligned_box():
    track = {
        "frame_ids": [20, 40],
        "bounding_boxes": [[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5]],
        "is_masked": [0, 1],
    }
    assert track_box_at_frame(track, 20) == [0.1, 0.2, 0.3, 0.4]
    with pytest.raises(ValueError, match="unmasked"):
        track_box_at_frame(track, 40)
    with pytest.raises(ValueError, match="exactly one"):
        track_box_at_frame(track, 30)


def test_selected_frame_uses_decoded_bounds_even_when_metadata_count_may_differ():
    assert validate_selected_frame(510, [491, 545], 824) == 510
    with pytest.raises(ValueError, match="decoded event interval"):
        validate_selected_frame(824, [491, 825], 824)
    with pytest.raises(ValueError, match="decoded event interval"):
        validate_selected_frame(490, [491, 545], 824)
