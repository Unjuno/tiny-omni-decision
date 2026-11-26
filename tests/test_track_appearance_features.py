import numpy as np
import pytest

from tiny_omni_decision.track_appearance_features import track_appearance_features


def grids():
    t, y, x = np.meshgrid(np.arange(2), np.arange(2), np.arange(2), indexing="ij")
    low = (10 * t + 2 * y + x)[..., None].astype(np.float32)
    return low, low * 2


def test_affine_fields_route_each_point_and_time_at_cell_centers():
    low, high = grids()
    xy = np.array([[[0.25, 0.25], [0.75, 0.25]], [[0.75, 0.75], [0.25, 0.75]]], np.float32)
    result, supported = track_appearance_features(low, high, xy)
    np.testing.assert_array_equal(
        result, np.array([[[0, 0], [11, 22]], [[3, 6], [12, 24]]], np.float32)
    )
    assert supported.all() and result.dtype == np.float32
    midpoint = np.full((1, 2, 2), 0.5, np.float32)
    r, _ = track_appearance_features(low, high, midpoint)
    np.testing.assert_array_equal(r, np.array([[[1.5, 3], [11.5, 23]]], np.float32))


def test_border_padding_outside_flags_all_points_retained_and_no_mutation():
    low, high = grids()
    xy = np.array([[[-0.2, 0.25], [1.2, 0.75]], [[0, 0], [1, 1]]], np.float32)
    before = [v.copy() for v in (low, high, xy)]
    actual, supported = track_appearance_features(low, high, xy)
    np.testing.assert_array_equal(
        actual, np.array([[[0, 0], [13, 26]], [[0, 0], [13, 26]]], np.float32)
    )
    np.testing.assert_array_equal(supported, [[False, False], [True, True]])
    for original, value in zip(before, (low, high, xy), strict=True):
        np.testing.assert_array_equal(original, value)


def test_point_and_joint_time_permutations_and_repeat_exact():
    low, high = grids()
    xy = np.array([[[0.3, 0.4], [0.6, 0.7]], [[0.4, 0.5], [0.7, 0.6]]], np.float32)
    a, mask = track_appearance_features(low, high, xy)
    b, m = track_appearance_features(low, high, xy[::-1])
    assert a[::-1].tobytes() == b.tobytes() and mask[::-1].tobytes() == m.tobytes()
    c, _ = track_appearance_features(low[::-1], high[::-1], xy[:, ::-1])
    assert c.tobytes() == a[:, ::-1].tobytes()
    assert track_appearance_features(low, high, xy)[0].tobytes() == a.tobytes()


def test_huge_finite_outside_coordinates_are_retained_with_border_features():
    low, high = grids()
    xy = np.full((1, 2, 2), np.finfo(np.float64).max)
    value, supported = track_appearance_features(low, high, xy)
    np.testing.assert_array_equal(value, np.array([[[3, 6], [13, 26]]], np.float32))
    assert not supported.any()


def test_fp32_overflow_is_rejected_after_safe_accumulation():
    low, high = grids()
    low = np.full(low.shape, np.finfo(np.float64).max)
    with pytest.raises(ValueError, match="FP32"):
        track_appearance_features(low, high, np.full((1, 2, 2), 0.5, np.float32))


@pytest.mark.parametrize("kind", ["time", "shape", "empty", "nan", "integer", "complex"])
def test_invalid_inputs_fail_closed(kind):
    low, high = grids()
    xy = np.ones((1, 2, 2), np.float32)
    if kind == "time":
        xy = xy[:, :1]
    elif kind == "shape":
        low = low[..., 0]
    elif kind == "empty":
        xy = xy[:0]
    elif kind == "nan":
        xy[0, 0, 0] = np.nan
    elif kind == "integer":
        low = low.astype(np.int32)
    else:
        xy = xy.astype(np.complex64)
    with pytest.raises(ValueError):
        track_appearance_features(low, high, xy)
