import pytest

from scripts.video_parent_diagnostic import summarize


def row(i, parent, target, prediction, scene=1):
    return dict(
        id=str(i),
        parent_question_id=parent,
        scene_index=scene,
        question_type="predictive",
        target_index=target,
        prediction_index=prediction,
        probabilities=[0.8, 0.2] if prediction == 0 else [0.2, 0.8],
    )


def test_parent_accuracy_does_not_equal_choice_accuracy():
    result = summarize([row(1, "a", 1, 1), row(2, "a", 0, 1), row(3, "b", 0, 0)])
    assert result["all"]["choice_accuracy"] == pytest.approx(2 / 3)
    assert result["all"]["parent_exact_accuracy"] == 0.5
    assert result["all"]["parents"] == 2
    assert result["all"]["scenes"] == 1


@pytest.mark.parametrize("mutation", ["duplicate", "scene", "probability", "prediction"])
def test_rejects_invalid_evidence(mutation):
    rows = [row(1, "a", 1, 1), row(2, "a", 0, 0)]
    if mutation == "duplicate":
        rows[1]["id"] = "1"
    if mutation == "scene":
        rows[1]["scene_index"] = 2
    if mutation == "probability":
        rows[1]["probabilities"] = [0.8, 0.8]
    if mutation == "prediction":
        rows[1]["prediction_index"] = 1
    with pytest.raises(ValueError):
        summarize(rows)
