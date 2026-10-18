import json

import pytest

from scripts.replay_soft_targets import teacher_distribution


def source(kind, teacher, criteria=None):
    return {
        "state_id": "s",
        "questions": json.dumps({"q": {"type": kind, "criteria": criteria}}),
        "teacher": json.dumps({"q": teacher}),
    }


def test_choice_order_and_rounding():
    r = {"id": "source:s:q", "options": ["B: second", "A: first"]}
    p = teacher_distribution(r, source("choice", {"probabilities": {"A": 0.3333, "B": 0.6666}}))
    assert p == pytest.approx([2 / 3, 1 / 3])


def test_binary_and_score():
    r = {"id": "source:s:q", "options": ["false: no", "true: yes"]}
    assert teacher_distribution(r, source("noul", {"noul": 0.8})) == pytest.approx([0.2, 0.8])
    r["options"] = ["low", "high"]
    assert teacher_distribution(
        r, source("score", {"probabilities": {"0": 0.3, "1": 0.7}}, ["low", "high"])
    ) == [0.3, 0.7]
    with pytest.raises(ValueError):
        teacher_distribution(
            r, source("score", {"probabilities": {"0": 0.3, "1": 0.7}}, ["high", "low"])
        )


@pytest.mark.parametrize("p", [float("nan"), float("inf"), -0.1, 1.1, 0.4])
def test_invalid_simplex(p):
    with pytest.raises(ValueError):
        teacher_distribution(
            {"id": "source:s:q", "options": ["A: a", "B: b"]},
            source("choice", {"probabilities": {"A": p, "B": 0.5}}),
        )
