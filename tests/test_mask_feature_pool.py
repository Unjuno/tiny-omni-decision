import numpy as np
import pytest

from tiny_omni_decision import mask_feature_pool as module


def fixture():
    masks = np.zeros((3, 4, 4), np.uint8)
    masks[0, :2, :2] = 1
    masks[1, :2] = 1
    low = np.array([[1, 3], [5, 7]], np.float32)[..., None]
    high = np.broadcast_to(np.array([10, 20], np.float32), (4, 4, 2)).copy()
    return masks, [low, high]


def test_hand_checked_area_mean_population_std_and_empty_retention():
    masks, grids = fixture()
    mean, std, counts = module.mask_feature_statistics(masks, grids)
    assert mean.dtype == std.dtype == np.float32 and counts.dtype == np.int64
    assert np.array_equal(mean, [[1, 10, 20], [2, 10, 20], [0, 0, 0]])
    assert np.array_equal(std, [[0, 0, 0], [1, 0, 0], [0, 0, 0]])
    assert np.array_equal(counts, [4, 8, 0])


def test_partial_cells_weight_foreground_pixels_not_bounding_box():
    masks = np.zeros((1, 4, 4), np.uint8)
    masks[0, 0, 0] = masks[0, 3, 3] = 1
    low = np.array([[1, 3], [5, 7]], np.float32)[..., None]
    high = np.arange(16, dtype=np.float32).reshape(4, 4, 1)
    mean, std, counts = module.mask_feature_statistics(masks, [low, high])
    assert np.array_equal(mean, [[4, 7.5]]) and np.array_equal(std, [[3, 7.5]])
    assert np.array_equal(counts, [2])


def test_large_constant_field_has_exact_zero_variance_without_cancellation():
    masks, _ = fixture()
    grid = np.full((2, 2, 3), 1e12, np.float32)
    mean, std, counts = module.mask_feature_statistics(masks, [grid])
    assert np.array_equal(mean[:2], np.full((2, 3), 1e12, np.float32))
    assert np.array_equal(std, np.zeros((3, 3), np.float32))
    assert counts[2] == 0 and not mean[2].any()


def test_mask_order_overlap_and_inputs_are_preserved_without_rng_consumption():
    masks, grids = fixture()
    before = [a.tobytes() for a in [masks, *grids]]
    rng = np.random.get_state()
    normal = module.mask_feature_statistics(masks, grids)
    order = [2, 0, 1, 0]
    reordered = module.mask_feature_statistics(masks[order], grids)
    for a, b in zip(normal, reordered, strict=True):
        assert np.array_equal(a[order], b)
    assert [a.tobytes() for a in [masks, *grids]] == before
    after = np.random.get_state()
    assert rng[0] == after[0] and np.array_equal(rng[1], after[1]) and rng[2:] == after[2:]


@pytest.mark.parametrize('kind', ['no_masks', 'no_pixels', 'float_masks', 'nonbinary',
                                 'no_grids', 'half_grid', 'nan_grid', 'misaligned_grid',
                                 'too_large_grid', 'empty_channels'])
def test_unaligned_nonfinite_or_nonbinary_inputs_fail_closed(kind):
    masks, grids = fixture()
    if kind == 'no_masks':
        masks = masks[:0]
    elif kind == 'no_pixels':
        masks = masks[:, :0]
    elif kind == 'float_masks':
        masks = masks.astype(np.float32)
    elif kind == 'nonbinary':
        masks[0, 0, 0] = 2
    elif kind == 'no_grids':
        grids = []
    elif kind == 'half_grid':
        grids[0] = grids[0].astype(np.float16)
    elif kind == 'nan_grid':
        grids[0][0, 0] = np.nan
    elif kind == 'misaligned_grid':
        grids[0] = np.zeros((3, 2, 1), np.float32)
    elif kind == 'too_large_grid':
        grids[0] = np.zeros((8, 8, 1), np.float32)
    else:
        grids[0] = np.zeros((2, 2, 0), np.float32)
    with pytest.raises(ValueError):
        module.mask_feature_statistics(masks, grids)
