import pytest

np = pytest.importorskip('numpy')

from tiny_omni_decision import spatial_relations as relations  # noqa: E402


def test_nearest_anchor_uses_same_time_and_retains_every_query():
    q = np.array([[[0, 0], [9, 0]], [[3, 0], [1, 0]]], np.float32)
    anchors = np.array([[[1, 0], [8, 0]], [[4, 0], [0, 0]]], np.float32)
    result = relations.nearest_anchors(q, anchors)
    np.testing.assert_array_equal(result['indices'], [[0, 0], [1, 1]])
    np.testing.assert_array_equal(result['distances'], [[1, 1], [1, 1]])
    assert result['indices'].dtype == np.int64
    assert result['distances'].dtype == np.float64


def test_duplicates_exact_ties_first_input_index_and_margins():
    q = np.array([[[0., 0]], [[0., 0]]])
    anchors = np.array([[[-1., 0]], [[1., 0]], [[1., 0]]])
    result = relations.nearest_anchors(q, anchors)
    assert result['indices'].tolist() == [[0], [0]]
    assert result['ties'].tolist() == [[3], [3]]
    assert result['margins'].tolist() == [[0], [0]]


def test_outside_points_not_clipped_or_discarded_and_one_anchor_margin():
    q = np.array([[[-3., 0]], [[5., 0]]])
    anchors = np.array([[[0., 0]]])
    result = relations.nearest_anchors(q, anchors)
    np.testing.assert_array_equal(result['distances'], [[3], [5]])
    np.testing.assert_array_equal(result['margins'], [[0], [0]])


def test_query_and_time_permutation_no_input_mutation():
    q = np.arange(12, dtype=np.float32).reshape(2, 3, 2)
    anchors = np.array([[[0., 0], [4, 4], [9, 9]], [[10., 10], [5, 5], [0, 0]]])
    before = q.copy()
    a = relations.nearest_anchors(q, anchors)
    b = relations.nearest_anchors(q[::-1, ::-1], anchors[:, ::-1])
    for key in a:
        np.testing.assert_array_equal(a[key][::-1, ::-1], b[key])
    np.testing.assert_array_equal(q, before)


def test_large_finite_distances_use_hypot_not_overflowing_square():
    q = np.array([[[1e200, 1e200]]])
    a = np.array([[[0., 0]], [[1e200, 0]]])
    r = relations.nearest_anchors(q, a)
    assert r['indices'].tolist() == [[1]] and r['distances'][0, 0] == 1e200


@pytest.mark.parametrize('bad', [np.nan, np.inf, complex(1, 2)])
def test_nonfinite_or_nonreal_points_fail_closed(bad):
    with pytest.raises(ValueError):
        relations.nearest_anchors(np.array([[[bad, 0]]]), np.zeros((1, 1, 2), np.float32))


@pytest.mark.parametrize('shape', [(0, 2, 2), (1, 2, 3), (1, 1, 2), (1, 2)])
def test_empty_misaligned_time_or_xy_shape_rejected(shape):
    with pytest.raises(ValueError):
        relations.nearest_anchors(np.ones(shape), np.zeros((2, 2, 2), np.float32))


def test_future_relation_summary_excludes_birth_and_keeps_unknown_ambiguous():
    labels = np.array([[3, 3, -1, -2], [1, 1, 1, 2]], np.int64)
    result = relations.future_relation_summary(np.array([3, 1]), labels,
                                               np.array([18, 0]), np.array([0, 18, 36, 54]))
    assert result == dict(queries=2, future_denominator=5, hits=2,
                         geometric_relation_rate=.4, unknown=1, ambiguous=1,
                         resolved=3, known_wrong=1)


def test_no_future_observations_not_converted_to_perfect_rate():
    r = relations.future_relation_summary(np.array([3]), np.array([[3, 3]]),
                                           np.array([18]), np.array([0, 18]))
    assert r['future_denominator'] == 0 and r['geometric_relation_rate'] is None


@pytest.mark.parametrize('labels', [np.array([[1., 1]]), np.array([[1, -3]]),
                                   np.array([[1]]), np.array([1, 1])])
def test_invalid_future_label_contract_fails_closed(labels):
    with pytest.raises(ValueError):
        relations.future_relation_summary(np.array([1]), labels,
                                           np.array([0]), np.array([0, 18]))


def test_seed_role_requires_whole_box_overlap_and_own_point():
    boxes = np.array([[0, 0, 1, 1], [0, 0, .25, .25], [0, 0, 1, 1]])
    xy = np.array([[.5, .5], [.1, .1], [-.1, .5]])
    labels = relations.seed_box_roles(boxes, xy, np.array([3]), np.array([[0, 0, 1, 1]]))
    np.testing.assert_array_equal(labels, [3, -1, -1])


def test_seed_role_exact_overlap_threshold_and_ambiguous_boundary_retained():
    boxes = np.array([[0., 0, 1, 1]])
    targets = np.array([[0., 0, .5, 1], [.5, 0, 1, 1]])
    labels = relations.seed_box_roles(boxes, np.array([[.5, .5]]), np.array([3, 7]), targets)
    assert labels.tolist() == [-2]


def test_empty_source_box_inventory_keeps_all_queries_unknown():
    labels = relations.seed_box_roles(np.array([[0., 0, 1, 1], [0, 0, 1, 1]]),
                                      np.array([[.5, .5], [.5, .5]]),
                                      np.array([], np.int64), np.empty((0, 4)))
    assert labels.tolist() == [-1, -1]


@pytest.mark.parametrize('ids', [[3, 3], [-1, 3], [3., 7.], [True, False]])
def test_source_role_namespace_requires_unique_integer_ids(ids):
    with pytest.raises(ValueError):
        relations.seed_box_roles(np.array([[0., 0, 1, 1]]), np.array([[.5, .5]]),
                                  np.array(ids), np.array([[0., 0, 1, 1], [0, 0, 1, 1]]))
