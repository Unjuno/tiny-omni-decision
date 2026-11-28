from __future__ import annotations

import pytest

from scripts.extract_physionpp_train_videoprism_features import select_train_rows


def _row(index: int, *, split: str = "train", source: str = "physionpp/readout"):
    return {
        "id": f"record-{index}",
        "split": split,
        "source": source,
        "media": [{"kind": "video", "sha256": f"{index:064x}"}],
    }


def test_select_train_rows_accepts_manifest_order_and_limit() -> None:
    rows = [_row(index) for index in range(3)]
    assert select_train_rows(rows, limit=2) == rows[:2]
    assert select_train_rows(rows) == rows


def test_select_train_rows_rejects_non_train_or_wrong_source() -> None:
    with pytest.raises(ValueError, match="TRAIN"):
        select_train_rows([_row(0, split="validation")])
    with pytest.raises(ValueError, match="non-Physion"):
        select_train_rows([_row(0, source="other")])


def test_select_train_rows_rejects_invalid_limit_and_media() -> None:
    rows = [_row(0)]
    with pytest.raises(ValueError, match="limit"):
        select_train_rows(rows, limit=2)
    with pytest.raises(ValueError, match="exactly one"):
        select_train_rows([{**rows[0], "media": []}])


def test_select_train_rows_rejects_duplicate_asset_identity() -> None:
    rows = [_row(0), {**_row(1), "media": _row(0)["media"]}]
    with pytest.raises(ValueError, match="duplicate media"):
        select_train_rows(rows)
