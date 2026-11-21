import pytest

from scripts.audit_native_common_train_coverage import coverage_summary


def _row(modality, index, *, count=3, source=None):
    return {
        "id": f"{modality}-{index}",
        "split": "train",
        "modality": modality,
        "cohort": "cohort",
        "task": "task",
        "source": source or modality,
        "observation": f"{modality}:{index}",
        "group": f"group:{index}",
    }


def test_coverage_summary_reports_unique_and_repeated_draws():
    rows = [
        *[_row("text", i) for i in range(3)],
        *[_row("image", i) for i in range(3)],
        *[_row("audio", i, source="starss") for i in range(3)],
        _row("video", 0),
    ]

    result = coverage_summary(rows, steps=2, seed=17)

    assert result["consumed_draws"] == 8
    assert result["unique_questions_consumed"] == 7
    assert result["repeated_question_draws"] == 1
    assert result["unique_questions_unseen"] == 3
    assert result["modalities"]["video"]["sample_draws"] == 2
    assert result["modalities"]["video"]["unique_questions_consumed"] == 1
    assert result["modalities"]["video"]["repeated_question_draws"] == 1
    assert result["modalities"]["image"]["unique_questions_unseen"] == 1
    assert result["validation_loaded"] is False
    assert result["sealed_audit_loaded"] is False


def test_coverage_summary_rejects_non_train_and_duplicate_ids():
    row = _row("text", 0)
    with pytest.raises(ValueError, match="TRAIN"):
        coverage_summary([{**row, "split": "validation"}], steps=1)
    with pytest.raises(ValueError, match="unique TRAIN"):
        coverage_summary([row, row], steps=1)
