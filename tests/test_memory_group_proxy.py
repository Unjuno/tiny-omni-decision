import numpy as np
import pytest

from tiny_omni_decision.memory_group_proxy import evaluate_group_proxy


def fixture():
    left = [0.0, 0.0, 0.2, 0.2]
    right = [0.6, 0.6, 0.8, 0.8]
    other = [0.3, 0.3, 0.4, 0.4]
    source = np.array([[left, right], [left, right], [right, left]], np.float64)
    available = np.ones((3, 2), bool)
    available[1, 0] = False
    actual = np.array([[left, left, right], [other, other, other]], np.float64)
    repeat = np.array([[left, left, left], [other, other, other]], np.float64)
    return actual, repeat, source, available


def test_missing_and_unsupported_groups_stay_in_future_denominators():
    actual, repeat, source, available = fixture()
    r = evaluate_group_proxy(actual, repeat, source, available, np.array([7, 9], np.int64))
    assert r["anchor_counts"] == dict(total=2, eligible=1, unknown=1, ambiguous=0)
    assert r["initial_source_groups_supported"] == 1
    assert r["source_groups"] == 2
    assert r["arms"]["actual"]["eligible_seed_future_denominator"] == 2
    assert r["arms"]["actual"]["future_categories"]["target_source_unavailable"] == 1
    assert r["arms"]["actual"]["future_categories"]["same_unique_group"] == 1
    assert r["arms"]["actual"]["source_group_future_denominator"] == 4
    assert r["arms"]["actual"]["source_group_future_covered"] == 1
    assert r["arms"]["initial_repeat"]["future_categories"]["different_unique_group"] == 1
    assert r["arms"]["initial_repeat"]["source_group_future_covered"] == 0


def test_ambiguous_initial_seeds_are_retained_with_undefined_conditioned_rate():
    actual, repeat, source, available = fixture()
    source[:, 1] = source[:, 0]
    r = evaluate_group_proxy(actual[:1], repeat[:1], source, available, np.array([7, 9], np.int64))
    assert r["anchor_counts"] == dict(total=1, eligible=0, unknown=0, ambiguous=1)
    assert r["arms"]["actual"]["eligible_seed_future_rate"] is None
    assert r["arms"]["actual"]["source_group_future_denominator"] == 4
    assert r["arms"]["actual"]["source_group_future_covered"] == 0


def test_inconsistent_initial_masks_and_invalid_source_geometry_are_rejected():
    actual, repeat, source, available = fixture()
    repeat[0, 0] = [0.1, 0.1, 0.15, 0.15]
    with pytest.raises(ValueError):
        evaluate_group_proxy(actual, repeat, source, available, np.array([7, 9], np.int64))
    source[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        evaluate_group_proxy(actual, actual, source, available, np.array([7, 9], np.int64))


def test_missing_other_group_prevents_false_unique_anchor_and_future_match():
    actual, repeat, source, available = fixture()
    available[:] = True
    available[0, 1] = False
    r = evaluate_group_proxy(actual, repeat, source, available, np.array([7, 9], np.int64))
    assert r["anchor_counts"]["eligible"] == 0
    available[0, 1] = True
    available[1, 1] = False
    r = evaluate_group_proxy(actual, repeat, source, available, np.array([7, 9], np.int64))
    assert r["arms"]["actual"]["future_categories"]["incomplete_group_source"] == 1
    assert r["arms"]["actual"]["eligible_seed_future_denominator"] == 2
    assert r["arms"]["actual"]["source_group_future_covered"] == 1
