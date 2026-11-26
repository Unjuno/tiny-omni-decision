import numpy as np
import pytest

from tiny_omni_decision.proposal_opportunity import bbox_opportunity, source_box_at_time


def test_opportunity_reports_all_raw_and_nms_without_selecting_a_prediction():
    boxes = np.array([[0.0, 0.0, 1.0, 1.0], [0.5, 0.5, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]])
    values = bbox_opportunity(boxes, np.array([0, 2]), [0.5, 0.5, 1.0, 1.0])
    assert values["raw_max_iou"] == 1.0 and values["raw_best_index"] == 1
    assert values["nms_max_iou"] == 0.25 and values["nms_best_index"] == 0
    unknown = bbox_opportunity(boxes, np.array([0, 2]), None)
    assert unknown["raw_max_iou"] is None and unknown["nms_max_iou"] is None
    assert unknown["source_box_available"] is False


def test_source_boxes_are_range_limited_with_interpolation_and_masked_brackets():
    track = dict(
        frame_ids=[10, 20],
        bounding_boxes=[[0.0, 0.0, 0.2, 0.2], [0.4, 0.4, 0.6, 0.6]],
        is_masked=[0, 1],
    )
    missing = source_box_at_time(track, 9)
    assert missing["box"] is None and missing["bracket_frames"] is None
    value = source_box_at_time(track, 15)
    np.testing.assert_allclose(value["box"], [0.2, 0.2, 0.4, 0.4], atol=1e-15, rtol=0)
    assert value["bracket_frames"] == [10, 20] and value["source_masked_flags"] == [0, 1]
    assert value["nearest_annotation_distance_frames"] == 5
    exact = source_box_at_time(track, 10)
    assert exact["bracket_frames"] == [10, 10] and exact["source_frame_exact"]
    assert source_box_at_time(track, 21)["box"] is None


def test_invalid_box_geometry_and_nms_indices_fail_closed():
    with pytest.raises(ValueError):
        bbox_opportunity(np.array([[0.5, 0.0, 0.1, 1.0]]), np.array([0]), [0, 0, 1, 1])
    with pytest.raises(ValueError):
        bbox_opportunity(np.array([[0.0, 0.0, 1.0, 1.0]]), np.array([1]), [0, 0, 1, 1])
    with pytest.raises(ValueError):
        source_box_at_time(dict(frame_ids=[0, 0], bounding_boxes=[[0, 0, 1, 1]] * 2), 0)
