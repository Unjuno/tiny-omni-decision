import pytest

from scripts.extract_perception_test_text_features import (
    candidate_prompts,
    expected_source_split,
    strip_option_prefix,
)


def test_text_feature_split_maps_to_frozen_corpus_split_values():
    assert expected_source_split("train") == "train"
    assert expected_source_split("validation") == "valid"


def test_option_prefix_is_removed_before_text_encoding():
    assert strip_option_prefix("A: 12") == "12"
    assert strip_option_prefix("B: a red car") == "a red car"


@pytest.mark.parametrize("value", ["12", "D: 12", "A:", "A: "])
def test_option_prefix_requires_valid_unique_label_text(value):
    with pytest.raises(ValueError):
        strip_option_prefix(value)


def test_candidate_prompts_keep_question_and_remove_position_labels():
    row = {
        "question": "Which object moved first?",
        "options": ["A: car", "B: person", "C: ball"],
    }

    assert candidate_prompts(row) == [
        "Question: Which object moved first?\nCandidate answer: car",
        "Question: Which object moved first?\nCandidate answer: person",
        "Question: Which object moved first?\nCandidate answer: ball",
    ]


def test_candidate_prompts_reject_duplicate_choice_text_after_label_removal():
    with pytest.raises(ValueError, match="unique"):
        candidate_prompts({"question": "Choose", "options": ["A: same", "B: same", "C: other"]})
