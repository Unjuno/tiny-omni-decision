import pytest

from scripts.source_set_metrics import binary_metrics, score_record


def test_binary_metrics_correct_and_clamped():
    result = binary_metrics([1.0, 0.0], [1, 0])
    assert result["nll"] == 0 and result["brier"] == 0 and result["ece_10"] == 0
    assert binary_metrics([0.0], [1])["nll"] > 20
    with pytest.raises(ValueError):
        binary_metrics([float("nan")], [1])


def test_set_exact_and_none_are_exclusive():
    result = score_record(["a", "b"], ["a", "b"], ["a"], [0.8, 0.6], 0.1)
    assert not result["exact"] and result["none_correct"]
    assert score_record(["a"], [], ["none_visible"], [0.5], 0.9)["exact"]
    with pytest.raises(ValueError):
        score_record(["a"], [], ["a", "none_visible"], [0.5], 0.9)
    with pytest.raises(ValueError):
        score_record(["a"], ["b"], ["a"], [0.5], 0.1)
