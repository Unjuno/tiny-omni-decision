import numpy as np
import pytest

from scripts.diagnose_slot_masks import box_mask, compare_masks


def test_box_uses_pixel_centers():
    expected = np.zeros((4, 4), bool)
    expected[1:3, 1:3] = True
    assert np.array_equal(box_mask([0.25, 0.25, 0.75, 0.75], 4, 4), expected)


def test_fixed_identity_differs_from_rematching():
    a = np.zeros((4, 4), int)
    a[:, 2:] = 1
    masks = np.eye(2)[np.stack([a, 1 - a])].transpose(0, 3, 1, 2)
    rows = compare_masks(masks, [0, 30], {0: [0, 0, 0.5, 1], 30: [0, 0, 0.5, 1]})
    assert rows["selected_slot"] == 0
    assert rows["points"][1]["fixed_iou"] == 0
    assert rows["points"][1]["oracle_iou"] == 1
    assert rows["points"][1]["static_iou"] == 1


def test_missing_initial_box_is_explicit():
    assert compare_masks(np.ones((1, 1, 4, 4)), [0], {})["status"] == "NO_FRAME0_BOX"


def test_nonfinite_rejected():
    with pytest.raises(ValueError):
        compare_masks(np.full((1, 1, 4, 4), np.nan), [0], {0: [0, 0, 1, 1]})
