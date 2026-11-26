import json
from pathlib import Path

import pytest

from scripts.fetch_starss23_frozen_media import check_capacity, load_splits


def test_disk_reserve_is_enforced():
    check_capacity(100, 120, reserve=20)
    with pytest.raises(ValueError, match="insufficient disk"):
        check_capacity(101, 120, reserve=20)


def test_load_exact_frozen_split_set():
    root = Path(__file__).resolve().parents[1] / "docs/pretrained_reuse"
    summary = json.loads((root / "R2_STARSS23_SPLITS_20261011.json").read_text())
    rows = load_splits(root / "starss23-splits-v1", summary)
    assert len(rows) == len({r["clip"] for r in rows}) == 78
    assert all(r["upstream_split"] == "train" for r in rows)


def test_split_tampering_fails_before_fetch(tmp_path):
    (tmp_path / "train.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_splits(tmp_path, {"splits": {"train": {"sha256": "0" * 64}}})
