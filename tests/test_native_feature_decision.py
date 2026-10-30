from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from tiny_omni_decision.native_feature_decision import NativeFeatureDecisionScorer  # noqa: E402


def test_text_free_native_scorer_is_candidate_permutation_equivariant():
    torch.manual_seed(17)
    model = NativeFeatureDecisionScorer(feature_width=3, option_width=2, hidden_width=8).eval()
    state = torch.tensor([[0.2, -0.3, 0.7]])
    candidates = torch.tensor([[[1., 0.], [0., 1.], [2., -1.]]])
    valid = torch.tensor([[True, True, True]])

    logits = model(state, candidates, valid)
    order = torch.tensor([2, 0, 1])
    shuffled = model(state, candidates[:, order], valid[:, order])
    torch.testing.assert_close(shuffled, logits[:, order], rtol=0, atol=0)


def test_decide_preserves_supplied_option_order_and_returns_typed_json():
    torch.manual_seed(17)
    model = NativeFeatureDecisionScorer(feature_width=3, option_width=2, hidden_width=8).eval()
    result = model.decide(
        torch.tensor([0.2, -0.3, 0.7]),
        torch.tensor([[1., 0.], [0., 1.], [2., -1.]]),
        ["caller-b", "caller-a", "caller-c"],
        ["action-2", "action-1", "action-3"],
    )
    assert result["status"] == "selected"
    assert [row["option_id"] for row in result["options"]] == [
        "caller-b", "caller-a", "caller-c"
    ]
    assert sum(row["probability"] for row in result["options"]) == pytest.approx(1.0)
    assert result["selected_option_id"] in {"caller-a", "caller-b", "caller-c"}


def test_invalid_padding_is_ignored_but_available_values_must_be_finite():
    model = NativeFeatureDecisionScorer(feature_width=2, option_width=1, hidden_width=4)
    state = torch.tensor([[1., 2.]])
    candidates = torch.tensor([[[0.], [1.], [float("nan")]]])
    valid = torch.tensor([[True, True, False]])
    output = model(state, candidates, valid)
    assert torch.isfinite(output[valid]).all()
    assert torch.isneginf(output[~valid]).all()

    valid[0, 2] = True
    with pytest.raises(ValueError, match="finite"):
        model(state, candidates, valid)


def test_rejects_candidate_set_or_metadata_mismatch():
    model = NativeFeatureDecisionScorer(feature_width=2, option_width=1, hidden_width=4)
    with pytest.raises(ValueError, match="at least two"):
        model(torch.zeros(1, 2), torch.zeros(1, 2, 1), torch.tensor([[True, False]]))
    with pytest.raises(ValueError, match="unique nonempty"):
        model.decide(torch.zeros(2), torch.zeros(2, 1), ["same", "same"], ["a", "b"])
