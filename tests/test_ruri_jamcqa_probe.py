import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

import torch

from scripts.evaluate_ruri_jamcqa_dev import _cosine_logits


def test_cosine_option_scores_are_invariant_to_reversed_option_order():
    query = torch.tensor([1.0, 0.0])
    candidates = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])

    scores = _cosine_logits(query, candidates)
    reversed_scores = _cosine_logits(query, candidates.flip(0)).flip(0)

    torch.testing.assert_close(scores, reversed_scores, rtol=0, atol=0)


def test_cosine_scores_reject_incompatible_shapes():
    with pytest.raises(ValueError, match="incompatible shapes"):
        _cosine_logits(torch.ones(2), torch.ones(3, 4))

    with pytest.raises(ValueError, match="incompatible shapes"):
        _cosine_logits(torch.ones(1, 2), torch.ones(3, 2))
