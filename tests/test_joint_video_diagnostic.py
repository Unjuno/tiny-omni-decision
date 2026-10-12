import pytest

from scripts.report_joint_video_diagnostic import align, metrics


def fixture():
    pairs = [
        dict(
            video_id="v",
            questions=[dict(condition="initial", target=0), dict(condition="final", target=1)],
        )
    ]
    rows = [
        dict(
            video_id="v",
            condition=q["condition"],
            target=q["target"],
            mode=m,
            probabilities=[0.7, 0.2, 0.1],
        )
        for q in pairs[0]["questions"]
        for m in ("normal", "black_pixels")
    ]
    return rows, pairs


def test_alignment_and_condition_metrics():
    rows, pairs = fixture()
    values, targets = align(rows[::-1], pairs)
    result = metrics(values["normal"], targets)
    assert result["accuracy"] == 0.5
    assert result["both_correct"] == 0
    assert result["predicted_condition_changes"] == 0


@pytest.mark.parametrize("failure", ["missing", "duplicate", "label", "nan"])
def test_invalid_predictions_rejected(failure):
    rows, pairs = fixture()
    if failure == "missing":
        rows.pop()
    elif failure == "duplicate":
        rows.append(rows[0])
    elif failure == "label":
        rows[0]["target"] = 2
    else:
        rows[0]["probabilities"] = [float("nan"), 0.5, 0.5]
    with pytest.raises(ValueError):
        align(rows, pairs)
