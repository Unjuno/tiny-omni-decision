import pytest

np = pytest.importorskip('numpy')

from tiny_omni_decision import mask_trajectory_features as module  # noqa: E402


def bind(masks, ids, frame=3, frames=6):
    return module.timed_mask_queries(masks, ids, frame_index=frame, frames=frames,
                                     width=8, height=4)


def test_time_xy_conversion_keeps_all_ids_duplicates_and_empty_flags():
    masks = np.zeros((3, 4, 4), dtype=bool)
    masks[0, 1, 3] = True
    masks[1, 1, 3] = True
    original = masks.copy()
    q, ids, valid = bind(masks, [91, 8, 23])
    assert q.dtype == np.float32 and ids.dtype == np.int64
    np.testing.assert_array_equal(q, [[3, 1.5, 7], [3, 1.5, 7], [3, 0, 0]])
    np.testing.assert_array_equal(ids, [91, 8, 23])
    assert valid.tolist() == [True, True, False]
    np.testing.assert_array_equal(masks, original)


def test_last_original_frame_not_retained_time_ordinal():
    masks = np.ones((1, 4, 4), dtype=bool)
    q, _, _ = bind(masks, [0], frame=557, frames=558)
    np.testing.assert_array_equal(q, [[557, 1.5, 3]])


@pytest.mark.parametrize('frame,frames', [(-1, 6), (6, 6), (1.5, 6), (True, 6),
                                         (0, 0), (0, 6.5), (0, True)])
def test_invalid_original_time_fails_closed(frame, frames):
    with pytest.raises(ValueError):
        bind(np.ones((1, 4, 4), bool), [0], frame, frames)


@pytest.mark.parametrize('ids', [[1, 1], [1], [-1, 2], [1., 2.], [True, False]])
def test_proposal_identity_must_be_unique_aligned_nonnegative_integers(ids):
    with pytest.raises(ValueError):
        bind(np.ones((2, 4, 4), bool), ids)


def test_timed_queries_follow_mask_permutation_without_aliasing():
    masks = np.zeros((2, 4, 4), bool)
    masks[0, 0, 0] = True
    masks[1, 3, 3] = True
    ids = np.array([17, 2], np.int64)
    a = bind(masks, ids)
    b = bind(masks[::-1], ids[::-1])
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x[::-1], y)
    a[1][0] = 999
    assert ids.tolist() == [17, 2]


def test_pre_observation_support_uses_original_times_keeps_backwards_predictions():
    queries = np.array([[0, 1, 2], [18, 1, 2], [557, 1, 2]], np.float32)
    flags = module.query_observation_support(queries, np.array([0, 18, 234, 557]))
    assert flags.dtype == np.bool_
    np.testing.assert_array_equal(flags, [[1, 1, 1, 1], [0, 1, 1, 1], [0, 0, 0, 1]])


@pytest.mark.parametrize('times', [[0, 0], [18, 0], [-1, 2], [0., 2.]])
def test_support_rejects_ambiguous_or_noninteger_sample_times(times):
    with pytest.raises(ValueError):
        module.query_observation_support(np.array([[0, 1, 2]], np.float32), times)
