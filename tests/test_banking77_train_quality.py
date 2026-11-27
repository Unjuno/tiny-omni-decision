from __future__ import annotations

import hashlib

import pytest

from scripts.audit_banking77_train_quality import (
    audit_file,
    audit_rows,
    near_duplicate_candidates,
    normalize_text,
    summarize_candidate_components,
)


def _rows() -> list[dict[str, str]]:
    rows = [
        {"text": f"Unique query for intent {index}", "category": f"intent_{index}"}
        for index in range(76)
    ]
    rows.extend(
        [
            {"text": "Card arrival?", "category": "arrival"},
            {"text": " card ARRIVAL! ", "category": "arrival"},
        ]
    )
    return rows


def test_normalize_text_collapses_unicode_case_spacing_and_punctuation() -> None:
    assert normalize_text(" ＣＡＲＤ-arrival?! ") == "card arrival"


def test_audit_rows_counts_duplicate_groups_without_exposing_text() -> None:
    report = audit_rows(_rows())
    assert report["intent_count"] == 77
    assert report["normalized_duplicate_groups"] == 1
    assert report["normalized_duplicate_conflict_groups"] == 0
    assert "text" not in report


def test_audit_rows_flags_duplicate_target_conflicts_and_possible_pii() -> None:
    rows = _rows()
    rows[77] = {"text": "Card arrival?", "category": "intent_0"}
    rows.append({"text": "Contact me at test@example.com", "category": "intent_0"})
    report = audit_rows(rows)
    assert report["normalized_duplicate_conflict_groups"] == 1
    assert report["possible_pii_pattern_row_count"] == 1
    assert report["possible_pii_pattern_counts"] == {"email": 1}


def test_near_duplicate_screen_reports_cross_intent_pairs_without_text() -> None:
    rows = [
        {"text": "Has my top up gone through?", "category": "pending_top_up"},
        {"text": "Has my top-up gone through?", "category": "top_up_reverted"},
    ]
    candidates = near_duplicate_candidates(rows)
    assert candidates == [
        {
            "left_row_id": 0,
            "left_category": "pending_top_up",
            "right_row_id": 1,
            "right_category": "top_up_reverted",
            "token_set_jaccard": 1.0,
        }
    ]
    assert all("text" not in candidate for candidate in candidates)


def test_candidate_components_group_transitive_matches_and_report_label_mix() -> None:
    pairs = [
        {"left_row_id": 1, "left_category": "a", "right_row_id": 2, "right_category": "b"},
        {"left_row_id": 2, "left_category": "b", "right_row_id": 3, "right_category": "a"},
        {"left_row_id": 9, "left_category": "c", "right_row_id": 10, "right_category": "c"},
    ]
    summary = summarize_candidate_components(pairs)
    assert summary["candidate_row_count"] == 5
    assert summary["connected_component_count"] == 2
    assert summary["largest_component_sizes"] == [3, 2]
    assert summary["multi_label_component_count"] == 1
    assert summary["components_by_distinct_label_count"] == {"1": 1, "2": 1}


def test_audit_file_fails_closed_on_wrong_hash(tmp_path) -> None:
    data = b"text,category\nhello,intent\n"
    path = tmp_path / "train.csv"
    path.write_bytes(data)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        audit_file(path, hashlib.sha256(b"different").hexdigest())
