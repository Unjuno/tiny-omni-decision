import pytest

from scripts.compare_slot_mask_diagnostics import compare


def result(value, frame=30):
    return {
        "rows": [
            {
                "video_id": "v",
                "points": [
                    {"frame": 0, "fixed_iou": 0.5, "oracle_iou": 0.5, "static_iou": 1.0},
                    {"frame": frame, "fixed_iou": value, "oracle_iou": 0.5, "static_iou": 0.3},
                ],
            }
        ]
    }


def test_compare_excludes_initial_frame():
    report = compare(result(0.1), result(0.4))
    assert report["later_points"] == 1
    assert report["macro_video_delta"]["fixed_iou"] == pytest.approx(0.3)


def test_compare_rejects_different_frames():
    with pytest.raises(ValueError, match="frame"):
        compare(result(0.1), result(0.4, 31))


def test_compare_rejects_changed_static_geometry():
    candidate = result(0.4)
    candidate["rows"][0]["points"][1]["static_iou"] = 0.8
    with pytest.raises(ValueError, match="geometry"):
        compare(result(0.1), candidate)
