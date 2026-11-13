import pytest

np = pytest.importorskip('numpy')

from tiny_omni_decision.mask_trajectory_features import (  # noqa: E402
    mask_query_points,
    mask_trajectory_features,
    proposal_point_membership,
)


def test_mask_query_pixel_centers_and_empty_mask_flags():
    masks = np.zeros((3, 4, 4), dtype=bool)
    masks[0, 1, 3] = True
    masks[1] = True
    points, valid = mask_query_points(masks, width=8, height=4)
    assert points.dtype == np.float64 and points.shape == (3, 2)
    np.testing.assert_array_equal(points, [[7, 1.5], [3, 1.5], [0, 0]])
    assert valid.tolist() == [True, True, False]


def test_concave_mask_query_is_inside_not_centroid_in_hole():
    mask = np.ones((1, 5, 5), dtype=bool)
    mask[0, 1:4, 1:4] = False
    points, valid = mask_query_points(mask, width=5, height=5)
    # COM lies in the hole; equal-distance ties pick the first foreground pixel.
    np.testing.assert_array_equal(points, [[2.5, .5]])
    assert valid.all()
    assert proposal_point_membership(points, mask, width=5, height=5)[0, 0]


def test_mask_query_permutation_and_no_input_mutation():
    _, _, masks = fixture()
    original = masks.copy()
    p, valid = mask_query_points(masks, width=4, height=4)
    order = [3, 1, 0, 2]
    q, flags = mask_query_points(masks[order], width=4, height=4)
    np.testing.assert_array_equal(q, p[order])
    np.testing.assert_array_equal(flags, valid[order])
    np.testing.assert_array_equal(masks, original)


def test_mask_query_invalid_masks_and_raster_rejected():
    _, _, masks = fixture()
    for bad in (masks.astype(np.int8), masks[0], masks[:0]):
        with pytest.raises(ValueError):
            mask_query_points(bad)
    with pytest.raises(ValueError):
        mask_query_points(masks, width=0)
    with pytest.raises(ValueError):
        mask_query_points(masks, height=np.inf)


def test_exact_centroid_ties_choose_first_foreground_pixel():
    masks = np.array([[[1, 1, 1], [0, 0, 1], [1, 0, 1]]], dtype=bool)
    points, flags = mask_query_points(masks, width=3, height=3)
    np.testing.assert_array_equal(points, [[1.5, .5]])
    assert flags.all()


def test_exact_centroid_integer_distances_do_not_overflow():
    masks = np.ones((1, 1, 65536), dtype=bool)
    points, flags = mask_query_points(masks, width=65536, height=1)
    np.testing.assert_array_equal(points, [[32767.5, .5]])
    assert flags.all()


@pytest.mark.parametrize('width,x', [(3, np.nextafter(1., -np.inf)),
                                    (6, np.nextafter(2., -np.inf))])
def test_membership_does_not_round_across_internal_pixel_boundary(width, x):
    masks = np.array([[[True, False, False]]])
    members = proposal_point_membership([[x, .5]], masks, width=width, height=1)
    assert members[0, 0]


@pytest.mark.parametrize('bad', [complex(1, np.nan), complex(1, 2)])
def test_nonreal_queries_and_features_are_rejected_before_cast(bad):
    masks = np.ones((1, 1, 1), dtype=bool)
    with pytest.raises(ValueError):
        mask_trajectory_features(np.array([[bad]]), [[.5, .5]], masks, width=1, height=1)
    with pytest.raises(ValueError):
        proposal_point_membership(np.array([[bad, .5]]), masks, width=1, height=1)


def fixture():
    features = np.arange(4 * 12, dtype=np.float32).reshape(4, 12)
    xy = np.array([[1, 1], [3, 1], [1, 3], [3, 3]], dtype=np.float64)
    masks = np.zeros((4, 4, 4), dtype=bool)
    masks[0, :2] = True
    masks[1, 2:] = True
    masks[2] = True
    return features, xy, masks


def test_mean_all_candidates_and_explicit_unsupported():
    features, xy, masks = fixture()
    pooled, valid, counts = mask_trajectory_features(features, xy, masks, width=4, height=4)
    assert pooled.shape == (4, 12) and pooled.dtype == np.float32
    assert valid.tolist() == [True, True, True, False]
    assert counts.tolist() == [2, 2, 4, 0] and counts.dtype == np.int64
    np.testing.assert_array_equal(pooled[0], features[:2].mean(0))
    np.testing.assert_array_equal(pooled[1], features[2:].mean(0))
    np.testing.assert_array_equal(pooled[2], features.mean(0))
    np.testing.assert_array_equal(pooled[3], np.zeros(12, dtype=np.float32))


def test_overlap_is_preserved_without_exclusive_assignment():
    _, xy, masks = fixture()
    members = proposal_point_membership(xy, masks, width=4, height=4)
    assert members.dtype == np.bool_ and members.shape == (4, 4)
    assert members.tolist() == [[True, True, False, False], [False, False, True, True],
                              [True, True, True, True], [False, False, False, False]]


def test_all_unsupported_retained_as_invalid_not_fabricated_observations():
    features, xy, masks = fixture()
    masks[:] = False
    pooled, valid, counts = mask_trajectory_features(features, xy, masks, width=4, height=4)
    assert pooled.shape == (4, 12) and not valid.any() and not counts.any()
    assert not pooled.any()


def test_point_permutation_and_mask_permutation():
    features, xy, masks = fixture()
    expected = mask_trajectory_features(features, xy, masks, width=4, height=4)
    order = [3, 1, 0, 2]
    permuted_points = mask_trajectory_features(features[order], xy[order], masks, width=4, height=4)
    permuted_masks = mask_trajectory_features(features, xy, masks[order], width=4, height=4)
    for a, b, c in zip(expected, permuted_points, permuted_masks, strict=True):
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(a[order], c)


def test_temporal_columns_preserved_not_averaged_over_time():
    features, xy, masks = fixture()
    normal, _, _ = mask_trajectory_features(features, xy, masks, width=4, height=4)
    reversed_time = features.reshape(4, 3, 4)[:, ::-1].reshape(4, 12)
    reverse, _, _ = mask_trajectory_features(reversed_time, xy, masks, width=4, height=4)
    np.testing.assert_array_equal(normal.reshape(4, 3, 4)[:, ::-1].reshape(4, 12), reverse)
    assert not np.array_equal(normal, reverse)


def test_scaled_xy_order_and_outside_points_are_not_clipped():
    masks = np.ones((1, 2, 4), dtype=bool)
    masks[0, 1, 3] = False
    xy = [[0, 0], [7.9, 0], [0, 3.9], [7.9, 3.9], [-.01, 0], [8, 0], [0, 4], [1e30, 0]]
    members = proposal_point_membership(xy, masks, width=8, height=4)
    assert members.tolist() == [[True, True, True, False, False, False, False, False]]


@pytest.mark.parametrize('bad', [np.nan, np.inf])
def test_nonfinite_query_or_feature_fails_closed(bad):
    features, xy, masks = fixture()
    xy[0, 0] = bad
    with pytest.raises(ValueError):
        mask_trajectory_features(features, xy, masks, width=4, height=4)
    _, xy, _ = fixture()
    features[0, 0] = bad
    with pytest.raises(ValueError):
        mask_trajectory_features(features, xy, masks, width=4, height=4)


@pytest.mark.parametrize('width,height', [(0, 4), (4, -1), (np.nan, 4), (4, np.inf)])
def test_invalid_raster_rejected(width, height):
    features, xy, masks = fixture()
    with pytest.raises(ValueError):
        mask_trajectory_features(features, xy, masks, width=width, height=height)


def test_nonboolean_and_shape_mismatches_rejected():
    features, xy, masks = fixture()
    for invalid in (masks.astype(np.float32), masks[0], np.zeros((0, 4, 4), dtype=bool)):
        with pytest.raises(ValueError):
            mask_trajectory_features(features, xy, invalid, width=4, height=4)
    for invalid in (features[:2], features[:, :0], features[0]):
        with pytest.raises(ValueError):
            mask_trajectory_features(invalid, xy, masks, width=4, height=4)
    for invalid in (xy[:, :1], xy[:0], xy[:, :, None]):
        with pytest.raises(ValueError):
            mask_trajectory_features(features, invalid, masks, width=4, height=4)


def test_pooling_does_not_mutate_inputs_and_avoids_float32_sum_overflow():
    features, xy, masks = fixture()
    features[:] = np.finfo(np.float32).max
    originals = [a.copy() for a in (features, xy, masks)]
    tokens, valid, _ = mask_trajectory_features(features, xy, masks, width=4, height=4)
    assert np.isfinite(tokens).all()
    np.testing.assert_array_equal(tokens[valid], features[:3])
    for original, a in zip(originals, (features, xy, masks), strict=True):
        np.testing.assert_array_equal(original, a)


def test_existing_candidate_readout_respects_support_mask():
    torch = pytest.importorskip('torch')
    from tiny_omni_decision.trajectory_readout import TrajectoryReadout

    features, xy, masks = fixture()
    tokens, valid, _ = mask_trajectory_features(features, xy, masks, width=4, height=4)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        model = TrajectoryReadout(input_dim=12).eval()
        candidates = torch.randn(1, 3, 1536)
    supported = torch.from_numpy(valid)[None]
    original = torch.from_numpy(tokens)[None]
    poisoned_padding = original.clone()
    poisoned_padding[:, ~supported[0]] = 10000
    with torch.inference_mode():
        expected = model(original, candidates, supported)
        actual = model(poisoned_padding, candidates, supported)
    assert expected.shape == (1, 3) and torch.isfinite(expected).all()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    with pytest.raises(ValueError):
        model(original, candidates, torch.zeros_like(supported))
