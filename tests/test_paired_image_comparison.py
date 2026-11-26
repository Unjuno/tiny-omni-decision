import pytest

from tiny_omni_decision.paired_image_comparison import paired_image_metrics


def test_paired_direction_metrics_and_constant_interval():
    a = [
        dict(
            id=str(i),
            image_id=str(i),
            target="yes",
            options=["no", "yes"],
            probabilities=[0.8, 0.2],
        )
        for i in range(4)
    ]
    b = [dict(row, probabilities=[0.2, 0.8]) for row in a]
    result = paired_image_metrics(a, b, replicates=100)
    assert result["groups"] == 4
    assert result["metrics"]["accuracy"]["percentile_95_interval"] == [1, 1]
    assert result["metrics"]["nll"]["difference"] < 0
    assert result["metrics"]["brier"]["difference"] == pytest.approx(-1.2)


def test_duplicate_and_mismatched_ids_rejected():
    row = dict(
        id="a", image_id="image", target="yes", options=["no", "yes"], probabilities=[0.8, 0.2]
    )
    with pytest.raises(ValueError):
        paired_image_metrics([row, row], [row, row])
    with pytest.raises(ValueError):
        paired_image_metrics([row], [dict(row, target="no")])
