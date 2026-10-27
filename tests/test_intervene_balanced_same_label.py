from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from scripts.intervene_balanced_same_label import result_row  # noqa: E402


def test_set_result_preserves_multiple_targets_and_padding():
    example = SimpleNamespace(candidate_ids=["a", "b"], target_candidate_ids={"a", "b"})
    logits = torch.tensor([[1.0, 2.0, -1e9, -2.0]])
    mask = torch.tensor([[True, True, False]])
    result = result_row(example, logits, 0, mask)
    assert result["selected"] == ["a", "b"]
    assert result["exact"] is True
    assert len(result["conditional_probabilities"]) == 2


def test_set_result_preserves_none_exclusivity():
    example = SimpleNamespace(candidate_ids=["a"], target_candidate_ids=set())
    result = result_row(example, torch.tensor([[2.0, 1.0]]), 0, torch.tensor([[True]]))
    assert result["selected"] == ["none_visible"] and result["exact"]
