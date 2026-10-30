from __future__ import annotations

from scripts.prepare_egoviz_fresh_group_generalization import select_unseen_group_rows


def _row(chunk_id: str, group_id: str, *, duration: float = 20, size: int = 100) -> dict:
    return {
        "chunk_id": chunk_id,
        "group_id": group_id,
        "category": "activity",
        "duration_s": duration,
        "mcap_bytes": size,
        "mcap_path": f"data/activity/{chunk_id}/{chunk_id}.mcap",
    }


def test_selects_one_deterministic_clip_per_unseen_group() -> None:
    records = [
        _row("a1", "a"),
        _row("a2", "a"),
        _row("b1", "b"),
        _row("used", "old"),
        _row("short", "short", duration=14.9),
        _row("large", "large", size=1_000_000_001),
    ]

    selected = select_unseen_group_rows(records, {"old"}, seed=17)

    assert selected == select_unseen_group_rows(records, {"old"}, seed=17)
    assert {row["group_id"] for row in selected} == {"a", "b"}
    assert len({row["group_id"] for row in selected}) == len(selected)
    assert all(row["duration_s"] >= 15 for row in selected)
    assert all(0 < row["mcap_bytes"] <= 1_000_000_000 for row in selected)


def test_rejects_duplicate_chunk_ids_and_empty_eligible_pool() -> None:
    duplicate = [_row("same", "a"), _row("same", "b")]
    try:
        select_unseen_group_rows(duplicate, set(), seed=17)
    except ValueError as exc:
        assert "duplicate chunk" in str(exc)
    else:
        raise AssertionError("duplicate chunk IDs must fail closed")

    try:
        select_unseen_group_rows([_row("short", "a", duration=10)], set(), seed=17)
    except ValueError as exc:
        assert "no eligible" in str(exc)
    else:
        raise AssertionError("an empty eligible pool must fail closed")
