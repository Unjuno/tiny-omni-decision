from __future__ import annotations

import pytest

from tiny_omni_decision.uci_har_validation_analysis import analyze_paired_predictions


def _row(sample_id: str, subject: int, target: str, prediction: str) -> dict[str, object]:
    activities = (
        "WALKING",
        "WALKING_UPSTAIRS",
        "WALKING_DOWNSTAIRS",
        "SITTING",
        "STANDING",
        "LAYING",
    )
    probabilities = {name: 0.01 for name in activities}
    probabilities[prediction] = 0.95
    probabilities[target] = 0.95 if prediction == target else 0.02
    total = sum(probabilities.values())
    probabilities = {name: value / total for name, value in probabilities.items()}
    return {
        "sample_id": sample_id,
        "subject_id": subject,
        "target": target,
        "prediction": prediction,
        "probabilities": probabilities,
    }


def test_paired_validation_analysis_reports_subject_class_and_error_transitions():
    static = [
        _row("a", 5, "SITTING", "SITTING"),
        _row("b", 5, "LAYING", "STANDING"),
        _row("c", 11, "WALKING", "WALKING"),
    ]
    lstm = [
        _row("a", 5, "SITTING", "STANDING"),
        _row("b", 5, "LAYING", "STANDING"),
        _row("c", 11, "WALKING", "WALKING"),
    ]

    result = analyze_paired_predictions(static, lstm)

    assert result["count"] == 3
    assert result["paired_outcomes"] == {
        "static_only_correct": 1,
        "both_correct": 1,
        "both_wrong": 1,
    }
    assert result["by_subject"]["5"]["lstm_minus_static_accuracy"] == pytest.approx(-0.5)
    assert result["by_true_activity_prediction_transition"]["SITTING"] == {"SITTING->STANDING": 1}
    assert result["confidence_intervals"] is None


def test_paired_validation_analysis_rejects_different_id_order():
    static = [_row("a", 5, "SITTING", "SITTING"), _row("b", 5, "LAYING", "LAYING")]
    lstm = list(reversed(static))

    with pytest.raises(ValueError, match="sample IDs/order differ"):
        analyze_paired_predictions(static, lstm)


def test_paired_validation_analysis_rejects_metadata_mismatch():
    static = [_row("a", 5, "SITTING", "SITTING")]
    lstm = [_row("a", 11, "SITTING", "SITTING")]

    with pytest.raises(ValueError, match="metadata differs"):
        analyze_paired_predictions(static, lstm)


def test_paired_validation_analysis_rejects_duplicate_ids():
    first = _row("a", 5, "SITTING", "SITTING")
    static = [first, first]

    with pytest.raises(ValueError, match="unique"):
        analyze_paired_predictions(static, static)
