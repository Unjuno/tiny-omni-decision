import copy

import numpy as np
import pytest

from scripts.condition_binding_retention import cache_entry, checked_logits, retention_kl


def row():
    return dict(
        id="train:1",
        source="original",
        split="train",
        question="Choose",
        state="{}",
        options=["a", "b"],
        target="a",
    )


def test_cache_is_bound_to_complete_train_question_and_candidate_order():
    item = row()
    cached = cache_entry(item, np.array([2.0, -1.0]))
    assert checked_logits(item, cached) == [2.0, -1.0]
    for key, changed in (
        ("id", "different"),
        ("question", "Changed"),
        ("options", ["b", "a"]),
        ("state", '{"x":1}'),
    ):
        edited = dict(item, **{key: changed})
        with pytest.raises(ValueError):
            checked_logits(edited, cached)
    with pytest.raises(ValueError):
        cache_entry(dict(item, split="validation"), [2.0, -1.0])
    with pytest.raises(ValueError):
        cache_entry(dict(item, source="binding"), [2.0, -1.0])
    for invalid in ([1.0], [1.0, float("nan")]):
        with pytest.raises(ValueError):
            cache_entry(item, invalid)
    damaged = copy.deepcopy(cached)
    damaged["logits"][0] = float("inf")
    with pytest.raises(ValueError):
        checked_logits(item, damaged)


def test_kl_matches_manual_variable_choice_mean_and_detaches_teacher():
    torch = pytest.importorskip("torch")
    student = torch.tensor([0.1, 0.9, 1.0, -1.0, 0.4], dtype=torch.float64, requires_grad=True)
    teacher = torch.tensor([0.8, 0.2, -0.1, 0.3, 1.2], dtype=torch.float64, requires_grad=True)
    actual = retention_kl(student, [2, 3], teacher)
    expected = 0.0
    for a, b in zip(student.detach().split([2, 3]), teacher.detach().split([2, 3]), strict=True):
        p = b.softmax(0)
        expected += (p * (b.log_softmax(0) - a.log_softmax(0))).sum().item() / 2
    assert actual.item() == pytest.approx(expected)
    actual.backward()
    assert teacher.grad is None
    assert student.grad is not None and torch.isfinite(student.grad).all()
    assert student.grad.abs().sum() > 0
    assert retention_kl(student, [2, 3], student).item() == pytest.approx(0, abs=1e-12)


def test_kl_rejects_invalid_widths_and_nonfinite_scores():
    torch = pytest.importorskip("torch")
    scores = torch.tensor([0.0, 1.0, 2.0])
    for widths in ([2], [1, 2], []):
        with pytest.raises(ValueError):
            retention_kl(scores, widths, scores)
    with pytest.raises(ValueError):
        retention_kl(scores, [3], torch.tensor([0.0, float("nan"), 1.0]))
