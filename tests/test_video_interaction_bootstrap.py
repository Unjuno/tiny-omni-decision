import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from scripts.train_video_interaction_control import paired_macro


def test_bootstrap_rejects_missing_question_type():
    rows = [{"id": "a", "scene_index": 0, "question_type": "explanatory"}]
    preds = [{"id": "a", "target_index": 0, "prediction_index": 0}]
    with pytest.raises(ValueError, match="lacks a question type"):
        paired_macro(rows, preds, preds)


def test_paired_macro_zero_difference():
    rows = [
        {"id": str(i), "scene_index": 0, "question_type": task}
        for i, task in enumerate(["explanatory", "predictive", "counterfactual"])
    ]
    preds = [{"id": row["id"], "target_index": 0, "prediction_index": 0} for row in rows]
    result = paired_macro(rows, preds, preds)
    assert result["difference"] == 0
    assert result["percentile_95_interval"] == [0, 0]
