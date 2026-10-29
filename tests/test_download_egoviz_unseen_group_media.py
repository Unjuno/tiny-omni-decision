from __future__ import annotations

import pytest

from scripts.download_egoviz_unseen_group_media import select_media_rows


def _plan() -> dict[str, object]:
    return {
        "format": "egoviz-unseen-session-temporal-correspondence-plan-v1",
        "splits": {
            "train": [
                {
                    "chunk_id": "t1",
                    "group_id": "g1",
                    "expected_bytes": 10,
                    "mcap_path": "data/train/t1.mcap",
                },
                {
                    "chunk_id": "t2",
                    "group_id": "g2",
                    "expected_bytes": 20,
                    "mcap_path": "data/train/t2.mcap",
                },
            ],
            "development": [
                {
                    "chunk_id": "d1",
                    "group_id": "g3",
                    "expected_bytes": 30,
                    "mcap_path": "data/development/d1.mcap",
                },
                {
                    "chunk_id": "d2",
                    "group_id": "g4",
                    "expected_bytes": 40,
                    "mcap_path": "data/development/d2.mcap",
                },
            ],
            "sealed_audit": [
                {
                    "chunk_id": "a1",
                    "group_id": "g5",
                    "expected_bytes": 50,
                    "mcap_path": "data/audit/a1.mcap",
                }
            ],
        },
    }


def test_selects_only_train_and_development_and_respects_limit() -> None:
    rows = select_media_rows(_plan(), limit_per_split=1)

    assert [(row["split"], row["chunk_id"]) for row in rows] == [
        ("train", "t1"),
        ("development", "d1"),
    ]


def test_offset_selects_next_stable_row_in_each_split() -> None:
    rows = select_media_rows(_plan(), limit_per_split=1, offset_per_split=1)

    assert [(row["split"], row["chunk_id"]) for row in rows] == [
        ("train", "t2"),
        ("development", "d2"),
    ]
    train_only = select_media_rows(
        _plan(), splits=("train",), limit_per_split=1, offset_per_split=1
    )
    assert [(row["split"], row["chunk_id"]) for row in train_only] == [("train", "t2")]


def test_rejects_sealed_audit_and_invalid_split_names() -> None:
    with pytest.raises(ValueError, match="sealed_audit cannot be downloaded"):
        select_media_rows(_plan(), splits=("train", "sealed_audit"))
    with pytest.raises(ValueError, match="unsupported split"):
        select_media_rows(_plan(), splits=("test",))


def test_rejects_overlap_and_invalid_limits() -> None:
    plan = _plan()
    plan["splits"]["development"][0]["group_id"] = "g1"  # type: ignore[index]
    with pytest.raises(ValueError, match="group overlap"):
        select_media_rows(plan, limit_per_split=1)
    with pytest.raises(ValueError, match="positive"):
        select_media_rows(_plan(), limit_per_split=0)
    with pytest.raises(ValueError, match="nonnegative"):
        select_media_rows(_plan(), offset_per_split=-1)
    with pytest.raises(ValueError, match="no media rows"):
        select_media_rows(_plan(), offset_per_split=3)
