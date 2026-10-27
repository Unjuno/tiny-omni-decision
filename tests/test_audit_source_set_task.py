import pytest

pytest.importorskip("torch")
from scripts.audit_source_set_task import INSTRUCTION, score_row  # noqa: E402


def example():
    prediction = {
        "candidate_ids": ["a", "b"],
        "target_candidate_ids": ["a", "b"],
        "none_visible": False,
        "probabilities": [0.6, 0.3, 0.1],
        "correct": True,
    }
    annotation = {
        "request": {"instruction": INSTRUCTION},
        "supervision": {"target_candidate_ids": ["a", "b"], "none_visible": False},
    }
    return prediction, annotation


def test_multiple_targets_hit_does_not_mean_exact_set():
    p, a = example()
    assert score_row(p, a) == {"hit": True, "exact": False, "bound": False}


def test_none_is_single_correct_set():
    p, a = example()
    p.update(target_candidate_ids=[], none_visible=True, probabilities=[0.1, 0.2, 0.7])
    a["supervision"] = {"target_candidate_ids": [], "none_visible": True}
    assert score_row(p, a) == {"hit": True, "exact": True, "bound": True}


def test_reject_mismatched_annotation_and_recorded_hit():
    p, a = example()
    a["supervision"]["target_candidate_ids"] = ["a"]
    with pytest.raises(ValueError, match="supervision"):
        score_row(p, a)
    p, a = example()
    p["correct"] = False
    with pytest.raises(ValueError, match="recorded hit"):
        score_row(p, a)
