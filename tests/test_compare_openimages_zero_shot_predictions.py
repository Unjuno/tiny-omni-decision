import pytest

from scripts.compare_openimages_zero_shot_predictions import compare_rows


def _row(sample_id, prediction="a"):
    return {
        "sample_id": sample_id,
        "condition": "aligned",
        "options": ["a", "b", "c", "d"],
        "target": "a",
        "prediction": prediction,
        "probabilities": [0.7, 0.1, 0.1, 0.1] if prediction == "a" else [0.1, 0.7, 0.1, 0.1],
    }


def test_compare_requires_identical_ids_and_targets():
    left = [_row(str(i)) for i in range(168)]
    right = [_row(str(i)) for i in range(168)]
    tasks = [row | {"target_label": "/a"} for row in left]
    result = compare_rows(left, right, tasks)
    assert result["count"] == 168
    assert result["PE-Core"]["accuracy"] == 1.0
    assert result["paired"]["same_top1_prediction"] == 168
    right[-1] = _row("different-id")
    with pytest.raises(ValueError, match="identical 168 IDs"):
        compare_rows(left, right, tasks)


def test_compare_rejects_option_drift():
    left = [_row(str(i)) for i in range(168)]
    right = [_row(str(i)) for i in range(168)]
    right[0] = right[0] | {"options": ["d", "b", "c", "a"]}
    tasks = [row | {"target_label": "/a"} for row in left]
    with pytest.raises(ValueError, match="frozen options"):
        compare_rows(left, right, tasks)
