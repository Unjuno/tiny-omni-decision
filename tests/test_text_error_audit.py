import pytest

from scripts.audit_text_errors import summarize, teacher_distribution


def test_teacher_order_and_boolean():
    assert teacher_distribution(
        "choice", ["b: B", "a: A"], {"probabilities": {"a": 0.8, "b": 0.2}}
    ) == [0.2, 0.8]
    assert teacher_distribution(
        "score", ["low", "high"], {"probabilities": {"1": 0.7, "0": 0.3}}
    ) == [0.3, 0.7]
    assert teacher_distribution("noul", ["true: yes", "false: no"], {"noul": 0.6}) == [0.6, 0.4]


def test_teacher_missing_invalid_fail():
    with pytest.raises(ValueError):
        teacher_distribution("choice", ["a: A", "b: B"], {"probabilities": {"a": 0.8}})
    with pytest.raises(ValueError):
        teacher_distribution("noul", ["true", "false"], {"noul": 1.2})
    with pytest.raises(ValueError):
        teacher_distribution("score", ["low", "high"], {"probabilities": {"0": 0.5, "1": 0.8}})


def test_metrics_preserve_errors_and_probability():
    rows = [
        {"probabilities": [0.8, 0.2], "target_index": 0},
        {"probabilities": [0.8, 0.2], "target_index": 1},
    ]
    r = summarize(rows)
    assert r["count"] == 2 and r["errors"] == 1
    assert r["accuracy"] == 0.5
    assert r["brier"] == pytest.approx(0.68)
    assert r["ece"] == pytest.approx(0.3)
