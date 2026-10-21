import pytest

from scripts.diagnose_starss_ranking import ranking_metrics


def test_perfect_and_reverse_ranking():
    assert ranking_metrics([0, 1], [0.1, 0.9])["auroc"] == 1
    assert ranking_metrics([0, 1], [0.1, 0.9])["average_precision"] == 1
    assert ranking_metrics([0, 1], [0.9, 0.1])["auroc"] == 0
    assert ranking_metrics([0, 1], [0.9, 0.1])["average_precision"] == 0.5


def test_tied_scores_use_whole_group():
    for labels in ([0, 1, 0], [1, 0, 0]):
        result = ranking_metrics(labels, [0.5, 0.5, 0.5])
        assert result["auroc"] == 0.5
        assert result["average_precision"] == pytest.approx(1 / 3)


def test_single_class_and_invalid_scores():
    assert ranking_metrics([0, 0], [0.1, 0.3])["average_precision"] is None
    assert ranking_metrics([1, 1], [0.1, 0.3])["auroc"] is None
    with pytest.raises(ValueError):
        ranking_metrics([1], [float("nan")])
