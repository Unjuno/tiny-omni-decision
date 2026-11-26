import pytest

from scripts.freeze_vidimu_split import split_inventory


def inventory():
    return [{"name": f"dataset/videoandimus/S{s:02}/S{s:02}_A{a:02}_T01.raw"}
            for s in range(40, 56) for a in range(1, 14)]


def test_split_is_deterministic_subject_disjoint_and_order_independent():
    result = split_inventory(inventory())
    assert result == split_inventory(list(reversed(inventory())))
    groups = list(map(set, result["groups"].values()))
    assert list(map(len, groups)) == [10, 3, 3]
    assert not any(a & b for i, a in enumerate(groups) for b in groups[i + 1:])
    assert [len(r) for r in result["records"].values()] == [130, 39, 39]
    for split, records in result["records"].items():
        assert all(r["subject"] in result["groups"][split] for r in records)


def test_incomplete_or_duplicated_source_fails_closed():
    with pytest.raises(ValueError):
        split_inventory(inventory()[:-1])
    source = inventory()
    source[-1] = source[0]
    with pytest.raises(ValueError):
        split_inventory(source)
