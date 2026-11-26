import csv

import pytest

from scripts.audit_balance_corpus_option_shortcuts import audit_metadata


def _write_metadata(path, rows):
    fieldnames = [
        "pair_id",
        "clue_giver_condition",
        "target_word",
        "taboo_1",
        "taboo_2",
        "taboo_3",
        "taboo_4",
        "taboo_5",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _row(pair, target="coffee", first_taboo="latte"):
    return {
        "pair_id": pair,
        "clue_giver_condition": "board",
        "target_word": target,
        "taboo_1": first_taboo,
        "taboo_2": "drink",
        "taboo_3": "morning",
        "taboo_4": "beans",
        "taboo_5": "caffeine",
    }


def test_detects_deterministic_option_set_reused_across_pair_groups(tmp_path):
    metadata = tmp_path / "metadata.csv"
    _write_metadata(metadata, [_row("pair-a"), _row("pair-b")])

    result = audit_metadata(metadata)

    assert result["rows"] == 2
    assert result["pair_groups"] == 2
    assert result["unique_unordered_option_sets"] == 1
    assert result["deterministic_option_sets_shared_across_pair_groups"] == 1
    assert result["pair_group_split_has_exact_option_set_shortcut"] is True
    assert result["identifiers_or_raw_text_included"] is False


def test_different_targets_for_same_option_set_are_not_deterministic(tmp_path):
    metadata = tmp_path / "metadata.csv"
    _write_metadata(
        metadata,
        [_row("pair-a"), _row("pair-b", target="latte", first_taboo="coffee")],
    )

    result = audit_metadata(metadata)

    assert result["deterministic_option_sets"] == 0
    assert result["pair_group_split_has_exact_option_set_shortcut"] is False


def test_rejects_missing_columns(tmp_path):
    metadata = tmp_path / "metadata.csv"
    metadata.write_text("pair_id,target_word\na,coffee\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required columns"):
        audit_metadata(metadata)
