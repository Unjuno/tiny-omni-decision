import pytest

from scripts.audit_grid_quartet_distances import distances


def test_counts_distances_for_each_query_role():
    values = distances(['aaaaaa', 'bbaaaa', 'aaccaa', 'aaaadd'])
    assert values[0] == [2, 2, 2]
    assert values[1] == [2, 4, 4]


def test_duplicate_candidate_rejected():
    with pytest.raises(ValueError, match='quartet'):
        distances(['aaaaaa'] * 4)
