import numpy as np
import pytest

from tiny_omni_decision.track_association import (
    association_summary,
    descriptor_matches,
    group_membership,
)


def test_future_summary_excludes_self_time_and_retains_unknown_denominator():
    initial = np.array([7, 9, -1, -2])
    labels = np.array([[7, 7, -1], [9, 7, -2], [-1, 9, 9], [-2, -1, -1]])
    indices = np.broadcast_to(np.arange(4)[:, None], (4, 3))
    result = association_summary(initial, labels, indices)
    assert result["all_query_point_times"] == 12
    assert result["initial_group_queries"] == 2
    assert result["future_denominator"] == 4
    assert result["future_hits"] == 1 and result["future_same_group_bbox_rate"] == 0.25
    assert result["future_resolved_predictions"] == 2
    assert result["future_conditional_resolved_hit_rate"] == 0.5
    assert result["future_unknown_predictions"] == 1 and result["future_ambiguous_predictions"] == 1
    assert result["time0_hits"] == 2


def test_future_summary_invalid_index_fails_closed():
    with pytest.raises(ValueError):
        association_summary(np.array([7]), np.array([[7, 7]]), np.array([[0, 1]]))


def test_retrieval_routes_candidates_and_times_without_label_inputs():
    query = np.eye(2, dtype=np.float32)
    candidates = np.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]]], np.float32)
    before = candidates.copy()
    result = descriptor_matches(query, candidates)
    np.testing.assert_array_equal(result["indices"], [[0, 1], [1, 0]])
    np.testing.assert_array_equal(result["scores"], np.ones((2, 2)))
    np.testing.assert_array_equal(result["margins"], np.ones((2, 2)))
    np.testing.assert_array_equal(result["ties"], np.ones((2, 2)))
    np.testing.assert_array_equal(candidates, before)
    swapped = descriptor_matches(query, candidates[::-1])
    np.testing.assert_array_equal(swapped["indices"], 1 - result["indices"])
    repeat = descriptor_matches(query, candidates)
    assert all(repeat[k].tobytes() == result[k].tobytes() for k in result)


def test_zero_and_tie_queries_are_retained_with_explicit_flags():
    query = np.array([[0.0, 0.0], [1.0, 0.0]], np.float32)
    candidates = np.array([[[1.0, 0.0]], [[1.0, 0.0]], [[0.0, 0.0]]], np.float32)
    result = descriptor_matches(query, candidates)
    np.testing.assert_array_equal(result["indices"], [[0], [0]])
    np.testing.assert_array_equal(result["ties"], [[3], [2]])
    np.testing.assert_array_equal(result["zero_query"], [True, False])
    np.testing.assert_array_equal(result["zero_candidate"], [[False], [False], [True]])
    np.testing.assert_array_equal(result["margins"], [[0], [0]])


@pytest.mark.parametrize("kind", ["shape", "channels", "empty", "nan", "integer", "complex"])
def test_bad_descriptor_inputs_fail_closed(kind):
    q = np.eye(2, dtype=np.float32)
    c = np.ones((3, 2, 2), np.float32)
    if kind == "shape":
        c = c[..., 0]
    elif kind == "channels":
        c = c[..., :1]
    elif kind == "empty":
        c = c[:0]
    elif kind == "nan":
        q[0, 0] = np.nan
    elif kind == "integer":
        q = q.astype(np.int32)
    else:
        c = c.astype(np.complex64)
    with pytest.raises(ValueError):
        descriptor_matches(q, c)


def tracks():
    return [
        dict(id=7, frame_ids=[0, 10], bounding_boxes=[[0, 0, 0.4, 0.4], [0.4, 0, 0.8, 0.4]]),
        dict(id=9, frame_ids=[0, 10], bounding_boxes=[[0.2, 0, 0.6, 0.4], [0.6, 0, 1, 0.4]]),
    ]


def test_posthoc_groups_interpolate_only_within_ranges_and_keep_overlap_unknown():
    xy = np.array(
        [
            [[0.1, 0.1], [0.3, 0.1], [0.5, 0.1]],
            [[0.3, 0.1], [0.5, 0.1], [0.7, 0.1]],
            [[0.9, 0.9], [1.2, 0.1], [0.5, 0.9]],
        ],
        np.float32,
    )
    labels, coverage = group_membership(tracks(), [7, 9], xy, np.array([0, 5, 10]))
    np.testing.assert_array_equal(labels, [[7, 7, 7], [-2, -2, -2], [-1, -1, -1]])
    assert coverage.all()
    labels, available = group_membership(tracks(), [7, 9], xy, np.array([-1, 5, 11]))
    assert (labels[:, 0] == -1).all() and (labels[:, -1] == -1).all()
    np.testing.assert_array_equal(available, [[False, True, False], [False, True, False]])


def test_partial_group_box_coverage_cannot_resolve_identity():
    source = tracks()
    source[1]["frame_ids"] = [5, 10]
    source[1]["bounding_boxes"] = [[0.7, 0.6, 1.0, 0.9], [0.7, 0.6, 1.0, 0.9]]
    xy = np.array([[[0.1, 0.1], [0.3, 0.1], [0.5, 0.1]]], np.float32)
    labels, coverage = group_membership(source, [7, 9], xy, np.array([0, 5, 10]))
    np.testing.assert_array_equal(labels, [[-1, 7, 7]])
    np.testing.assert_array_equal(coverage, [[True, True, True], [False, True, True]])


@pytest.mark.parametrize("kind", ["duplicate", "missing", "bad_box", "bad_frames", "coordinate"])
def test_bad_posthoc_source_schema_fails_without_substitution(kind):
    source = tracks()
    ids = [7, 9]
    xy = np.ones((1, 2, 2), np.float32) * 0.3
    if kind == "duplicate":
        ids = [7, 7]
    elif kind == "missing":
        ids = [7, 10]
    elif kind == "bad_box":
        source[0]["bounding_boxes"][1][2] = 0.1
    elif kind == "bad_frames":
        source[0]["frame_ids"] = [0, 0]
    else:
        xy[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        group_membership(source, ids, xy, np.array([0, 10]))
