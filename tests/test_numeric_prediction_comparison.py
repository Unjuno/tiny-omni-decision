import pytest

from scripts.compare_numeric_embedding_predictions import compare


def row(identity, correct):
    return dict(
        id=identity,
        target="A",
        options=["A", "B"],
        probabilities=[0.8, 0.2] if correct else [0.2, 0.8],
    )


def test_cluster_point_weights_questions_not_groups():
    ids = ["s1:0", "s1:1", "s1:2", "s2:0"]
    a = [row(k, False) for k in ids[:3]] + [row(ids[3], True)]
    b = [row(k, True) for k in ids[:3]] + [row(ids[3], False)]
    result = compare(a, b, draws=100)
    assert result["metrics"]["accuracy"]["difference"] == 0.5
    assert result["states"] == 2
    assert result["transitions"] == {"0_to_1": 3, "1_to_0": 1}
    assert compare(a, a, draws=100)["metrics"]["nll"]["difference"] == 0
    with pytest.raises(ValueError):
        compare(a, b[::-1], draws=100)


def test_numeric_prefix_ignores_only_timing():
    from scripts.train_numeric_embedding_control import require_numeric_prefix

    a = dict(step=128, selector=0.5, elapsed_seconds=2, evaluation_seconds=1)
    require_numeric_prefix(a, dict(a, elapsed_seconds=3))
    with pytest.raises(ValueError):
        require_numeric_prefix(a, dict(a, selector=0.6))
