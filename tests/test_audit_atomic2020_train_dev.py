from __future__ import annotations

import importlib.util
from pathlib import Path

MODULE_PATH = (
    Path(__file__).parents[1] / "scripts" / "audit_atomic2020_train_dev.py"
)
SPEC = importlib.util.spec_from_file_location("atomic2020_audit", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_audit_counts_duplicates_multitails_and_group_overlap(tmp_path: Path) -> None:
    train = tmp_path / "train.tsv"
    dev = tmp_path / "dev.tsv"
    train.write_text(
        "PersonX helps PersonY\txWant\tto be thanked\n"
        "PersonX helps PersonY\txWant\tto feel good\n"
        "PersonX helps PersonY\txWant\tto be thanked\n",
        encoding="utf-8",
    )
    dev.write_text(
        "PersonX helps PersonY\txWant\tto be thanked\n"
        "PersonZ walks\txNeed\tnone\n",
        encoding="utf-8",
    )

    report = MODULE.audit_train_dev(train, dev)

    assert report["train"]["rows"] == 3
    assert report["train"]["exact_duplicate_rows"] == 1
    assert report["train"]["head_relation_pairs_with_multiple_tails"] == 1
    assert report["dev"]["none_tail_rows"] == 1
    assert report["cross_split"]["raw_event_head_overlap"] == 1
    assert report["cross_split"]["exact_triple_overlap"] == 1


def test_audit_counts_and_excludes_malformed_rows(tmp_path: Path) -> None:
    train = tmp_path / "train.tsv"
    dev = tmp_path / "dev.tsv"
    train.write_text(
        "head\trelation\ttail\n"
        "empty-tail-head\trelation\t\n"
        "malformed\trow\n",
        encoding="utf-8",
    )
    dev.write_text(
        "head\trelation\ttail\n"
        "empty-tail-head\trelation\tdev-tail\n",
        encoding="utf-8",
    )

    report = MODULE.audit_train_dev(train, dev)

    assert report["train"]["rows"] == 3
    assert report["train"]["valid_rows"] == 1
    assert report["train"]["empty_field_rows"] == 1
    assert report["train"]["malformed_width_rows"] == 1
    assert report["train"]["normalized_event_heads"] == 2
    assert report["cross_split"]["raw_event_head_overlap"] == 2
    assert report["cross_split"]["normalized_event_head_overlap"] == 2
