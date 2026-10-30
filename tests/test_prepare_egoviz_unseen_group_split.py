from __future__ import annotations

import pytest

from scripts.prepare_egoviz_unseen_group_split import select_grouped_records


def _records() -> list[dict[str, object]]:
    rows = []
    categories = ("rare", "common")
    for index in range(18):
        rows.append(
            {
                "chunk_id": f"clip-{index:02d}",
                "group_id": f"group-{index:02d}",
                "category": categories[index % 2],
                "duration_s": 20.0,
                "mcap_bytes": 1000 + index,
                "mcap_path": f"data/{index}.mcap",
            }
        )
    return rows


def test_split_is_deterministic_group_disjoint_and_category_covered() -> None:
    records = _records()
    capacities = {"train": 4, "development": 3, "sealed_audit": 3}

    first = select_grouped_records(records, set(), capacities, seed=17)
    second = select_grouped_records(records, set(), capacities, seed=17)

    assert first == second
    groups = {
        split: {str(row["group_id"]) for row in rows}
        for split, rows in first.items()
    }
    assert groups["train"].isdisjoint(groups["development"])
    assert groups["train"].isdisjoint(groups["sealed_audit"])
    assert groups["development"].isdisjoint(groups["sealed_audit"])
    for rows in first.values():
        assert {str(row["category"]) for row in rows} == {"rare", "common"}


def test_excludes_previously_used_groups_and_oversize_media() -> None:
    records = _records()
    records[0]["mcap_bytes"] = 10_000
    capacities = {"train": 2, "development": 2, "sealed_audit": 2}

    selected = select_grouped_records(
        records,
        {"group-02"},
        capacities,
        seed=17,
        max_mcap_bytes=5_000,
    )

    flat = [row for rows in selected.values() for row in rows]
    assert all(row["group_id"] != "group-02" for row in flat)
    assert all(int(row["mcap_bytes"]) <= 5_000 for row in flat)
    assert len({row["group_id"] for row in flat}) == len(flat)


def test_fails_when_a_category_cannot_be_represented_in_each_split() -> None:
    rows = _records()[:3]
    capacities = {"train": 1, "development": 1, "sealed_audit": 1}

    with pytest.raises(ValueError, match="distinct eligible groups"):
        select_grouped_records(rows, set(), capacities, seed=17)
