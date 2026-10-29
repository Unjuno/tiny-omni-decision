from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import download_egoviz_unseen_group_media as downloader
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


@pytest.mark.parametrize("free_bytes", [2_500_000_000, 3_000_000_000, 3_000_000_099])
def test_default_reserves_three_gb_after_selected_downloads_before_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, free_bytes: int
) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(_plan()), encoding="utf-8")
    output = tmp_path / "new-acquisition"
    monkeypatch.setattr(
        downloader.shutil, "disk_usage", lambda _: SimpleNamespace(free=free_bytes)
    )

    def unexpected_network() -> None:
        pytest.fail("insufficient capacity must fail before creating a network session")

    monkeypatch.setattr("requests.Session", unexpected_network)
    # The selected four synthetic files total 100 bytes; the reserve remains free.
    with pytest.raises(OSError, match="insufficient disk reserve: need 3000000100"):
        downloader.download(plan_path, output)
    assert not output.exists()
