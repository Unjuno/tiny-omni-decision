import pytest

from scripts.prepare_egoviz_train_expansion import select_expansion


def test_expansion_is_deterministic_balanced_and_excludes_all_reserved_groups():
    records = [
        {
            "group_id": f"g{i}",
            "chunk_id": f"c{i}",
            "category": f"category{i % 2}",
            "duration_s": 16.0,
            "mcap_bytes": 100 + i,
            "mcap_path": f"data/c{i}.mcap",
        }
        for i in range(30)
    ]
    a = select_expansion(records, {"g0", "g1"}, count=24, seed=17)
    assert a == select_expansion(records, {"g0", "g1"}, count=24, seed=17)
    assert len({r["group_id"] for r in a}) == 24
    assert not ({r["group_id"] for r in a} & {"g0", "g1"})
    assert all(
        sum(r["category"] == category for r in a) >= 2 for category in {"category0", "category1"}
    )


def test_expansion_fails_closed_when_distinct_groups_are_insufficient():
    row = {
        "group_id": "g1",
        "chunk_id": "c1",
        "category": "one",
        "duration_s": 16.0,
        "mcap_bytes": 100,
        "mcap_path": "data/c1.mcap",
    }
    with pytest.raises(ValueError, match="distinct"):
        select_expansion([row], set(), count=2, seed=17)
