from __future__ import annotations

import pytest

from scripts.extract_physionpp_validation_videoprism_features import (
    select_validation_rows,
)


def _row(index: int, *, split: str = "validation", sha: str | None = None):
    return {
        "id": f"val-{index}",
        "split": split,
        "source": "physionpp/readout",
        "media": [{"kind": "video", "sha256": sha or f"{index + 100:064x}"}],
    }


def test_validation_selector_keeps_manifest_order_and_disjoint_rows() -> None:
    rows = [_row(0), _row(1)]
    assert select_validation_rows(rows, {"train-1"}, {f"{0:064x}"}) == rows


def test_validation_selector_rejects_train_rows_and_overlap() -> None:
    with pytest.raises(ValueError, match="development validation"):
        select_validation_rows([_row(0, split="train")], set(), set())
    row = _row(0)
    with pytest.raises(ValueError, match="overlap"):
        select_validation_rows([row], {row["id"]}, set())
    with pytest.raises(ValueError, match="overlap"):
        select_validation_rows([row], set(), {row["media"][0]["sha256"]})


def test_validation_selector_rejects_duplicate_media_and_missing_asset() -> None:
    first = _row(0)
    second = _row(1, sha=first["media"][0]["sha256"])
    with pytest.raises(ValueError, match="unique"):
        select_validation_rows([first, second], set(), set())
    with pytest.raises(ValueError, match="exactly one"):
        select_validation_rows([{**first, "media": []}], set(), set())
