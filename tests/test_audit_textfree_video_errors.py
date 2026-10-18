import pytest

from scripts.audit_textfree_video_errors import (
    analyze_program_signatures,
    attribute_predictions,
)


def row(sample_id, target, prediction, *, taxonomy="exist", question_type="temporal_descriptive"):
    return {
        "id": sample_id,
        "scene_group_id": f"scene:{sample_id}",
        "taxonomy": taxonomy,
        "question_type": question_type,
        "target": target,
        "prediction": prediction,
        "options": [
            {"option_id": "yes", "probability": 0.7 if prediction == "yes" else 0.3},
            {"option_id": "no", "probability": 0.7 if prediction == "no" else 0.3},
        ],
    }


def test_attribution_reports_support_majority_bias_and_paired_outcomes():
    mean = [row("a", "yes", "yes"), row("b", "no", "yes"), row("c", "yes", "no")]
    lstm = [row("a", "yes", "yes"), row("b", "no", "no"), row("c", "yes", "no")]

    result = attribute_predictions({"mean_mlp": mean, "bilstm": lstm}, expected_count=3)
    cell = result["question_taxonomy_cells"]["exist/temporal_descriptive"]

    assert cell["mean_mlp"]["accuracy"] == pytest.approx(1 / 3)
    assert cell["mean_mlp"]["empirical_majority_class_accuracy"] == pytest.approx(2 / 3)
    assert cell["mean_mlp"]["target_counts"] == {"no": 1, "yes": 2}
    assert cell["paired_accuracy_outcomes"] == {
        "bilstm_win": 1,
        "bilstm_loss": 0,
        "tie": 2,
        "net_correct_examples": 1,
    }


def test_attribution_rejects_mismatched_paired_ids():
    mean = [row("a", "yes", "yes")]
    lstm = [row("b", "yes", "yes")]

    with pytest.raises(ValueError, match="different prediction IDs"):
        attribute_predictions({"mean_mlp": mean, "bilstm": lstm})


def test_attribution_rejects_target_outside_supplied_options():
    mean = [row("a", "maybe", "yes")]
    lstm = [row("a", "maybe", "yes")]

    with pytest.raises(ValueError, match="invalid target"):
        attribute_predictions({"mean_mlp": mean, "bilstm": lstm})


def test_program_coverage_counts_unseen_compositions_without_labels():
    train = [
        {"question_type": "static", "program": ["objects", "unique", "count"]},
        {"question_type": "temporal", "program": ["events", "filter_before", "count"]},
    ]
    validation = [
        {"question_type": "static", "program": ["objects", "unique", "count"]},
        {"question_type": "temporal", "program": ["events", "filter_after", "count"]},
        {"question_type": "temporal", "program": ["events", "filter_after", "count"]},
    ]

    result = analyze_program_signatures(train, validation)

    assert result["static"]["validation_example_fraction_with_program_absent_from_train"] == 0
    assert result["temporal"]["validation_programs_absent_from_train"] == 1
    assert result["temporal"]["validation_examples_with_program_absent_from_train"] == 2
    assert result["temporal"]["validation_only_program_operators"] == ["filter_after"]
