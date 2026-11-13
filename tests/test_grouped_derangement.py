import pytest

from tiny_omni_decision.grouped_derangement import grouped_derangement


def test_reproducible_group_preserving_derangement():
    ids = ["a", "b", "c", "d", "e", "f"]
    groups = ["lower"] * 3 + ["upper"] * 3
    permutation = grouped_derangement(ids, groups)
    assert permutation == grouped_derangement(ids, groups)
    assert sorted(permutation) == list(range(6))
    assert all(i != j and groups[i] == groups[j] for i, j in enumerate(permutation))
    reversed_result = grouped_derangement(ids[::-1], groups[::-1])
    assert {ids[i]: ids[j] for i, j in enumerate(permutation)} == {
        ids[::-1][i]: ids[::-1][j] for i, j in enumerate(reversed_result)
    }


@pytest.mark.parametrize("ids,groups", [([], []), (["a"], ["x"]),
                                     (["a", "a"], ["x", "x"]),
                                     (["a", "b"], ["x"]),
                                     (["a", "b"], ["x", "y"])])
def test_impossible_or_ambiguous_permutation_rejected(ids, groups):
    with pytest.raises(ValueError):
        grouped_derangement(ids, groups)
