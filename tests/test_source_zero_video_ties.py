import pytest

from scripts.audit_source_zero_video_ties import summarize


def test_measures_ties_including_none_candidate():
    row = {
        "candidate_ids": ["a", "b"],
        "candidate_probabilities": {"a": 0.25, "b": 0.25},
        "none_visible_probability": 0.5,
        "predicted": "none_visible",
    }
    result = summarize([row])
    assert result["max_probability_spread"] == 0.25
    assert result["rows_with_spread_at_most_1e-6"] == 0


def test_rejects_invalid_probabilities_and_ids():
    row = {
        "candidate_ids": ["a"],
        "candidate_probabilities": {"b": 0.5},
        "none_visible_probability": 0.5,
        "predicted": "none_visible",
    }
    with pytest.raises(ValueError):
        summarize([row])
