import numpy as np
import pytest

from tiny_omni_decision.perception_video_roi import (
    map_bbox_to_token_grid,
    roi_average,
)


def test_wide_frame_letterbox_keeps_edge_box_inside_token_grid():
    # 1920x1080 content is scaled to 288x162 and vertically padded by 63px.
    assert map_bbox_to_token_grid([0.0, 0.0, 0.1, 1.0], 1920, 1080) == pytest.approx(
        (0.0, 3.5, 1.6, 12.5)
    )


def test_tall_frame_letterbox_maps_top_edge_without_center_crop_loss():
    # 1080x1920 content is scaled to 162x288 and horizontally padded by 63px.
    assert map_bbox_to_token_grid([0.0, 0.0, 1.0, 0.1], 1080, 1920) == pytest.approx(
        (3.5, 0.0, 12.5, 1.6)
    )


def test_roi_average_weights_all_intersecting_tokens_by_overlap_area():
    tokens = np.array([[[1.0], [3.0]], [[5.0], [7.0]]], dtype=np.float32)
    assert roi_average(tokens, (0.5, 0.5, 1.5, 1.5)) == pytest.approx([4.0])


def test_roi_average_rejects_empty_or_nonfinite_boxes():
    tokens = np.zeros((2, 2, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="positive finite"):
        roi_average(tokens, (1.0, 1.0, 1.0, 2.0))
    with pytest.raises(ValueError, match="positive finite"):
        roi_average(tokens, (0.0, 0.0, float("nan"), 1.0))


def test_roi_average_reads_only_tokens_under_the_requested_region():
    tokens = np.array([[[2.0], [float("nan")]], [[float("nan")], [float("nan")]]])
    assert roi_average(tokens, (0.0, 0.0, 1.0, 1.0)) == pytest.approx([2.0])
