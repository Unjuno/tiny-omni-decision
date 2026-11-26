import pytest

from scripts.audit_perception_condition_pairs import summarize


def test_pairs_compare_answer_text_not_position():
    row = dict(
        metadata={"split": "train"},
        mc_question=[
            dict(question="first?", options=["a", "b", "c"], answer_id=0, reasoning="descriptive"),
            dict(question="last?", options=["c", "a", "b"], answer_id=1, reasoning="descriptive"),
        ],
    )
    result = summarize({"v": row})["counts"]
    assert result["different_question_pairs"] == 1
    assert result.get("different_question_different_answer_pairs", 0) == 0
    row["mc_question"][1]["answer_id"] = 0
    assert summarize({"v": row})["counts"]["different_question_different_answer_pairs"] == 1
    row["metadata"]["split"] = "test"
    with pytest.raises(ValueError, match="train only"):
        summarize({"v": row})
