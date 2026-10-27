import pytest

from scripts.analyze_source_candidate_visibility import matched_rows, summarize


def row(name, correct, visible=True):
    return {
        "record_id": name,
        "video_id": name,
        "none_visible": not visible,
        "target_candidate_ids": ["x"] if visible else [],
        "top1_correct": correct,
    }


def test_pairing_uses_id_not_input_order_and_stratifies():
    a = [row("v", True), row("n", False, False)]
    b = [row("n", True, False), row("v", False)]
    pairs = matched_rows(a, b)
    assert summarize(pairs, False, repetitions=10)["joint_minus_comparison_accuracy"] == 1
    assert summarize(pairs, True, repetitions=10)["joint_minus_comparison_accuracy"] == -1


def test_pairing_rejects_duplicate_missing_or_mismatched_labels():
    a = [row("v", True)]
    with pytest.raises(ValueError):
        matched_rows(a, [])
    with pytest.raises(ValueError):
        matched_rows(a, [row("v", False), row("v", False)])
    with pytest.raises(ValueError):
        matched_rows(a, [row("v", True, False)])
