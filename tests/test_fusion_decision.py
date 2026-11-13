import pytest

pytest.importorskip("torch")
import torch

from tiny_omni_decision.fusion_decision import FusionDecisionScorer


def fixture():
    torch.manual_seed(17)
    model = FusionDecisionScorer({"video": 4, "sensor": 2}, 3, 2, width=8)
    features = {"video": torch.randn(2, 4), "sensor": torch.randn(2, 2)}
    masks = {"video": torch.tensor([True, False]), "sensor": torch.ones(2).bool()}
    options = torch.randn(2, 4, 3)
    valid = torch.tensor([[True, True, True, False], [True, True, False, False]])
    return model, features, masks, options, valid


def test_option_permutation_and_padding():
    model, features, masks, options, valid = fixture()
    options[~valid] = float("nan")
    logits = model(features, masks, options, valid)
    order = torch.tensor([2, 0, 3, 1])
    swapped = model(features, masks, options[:, order], valid[:, order])
    torch.testing.assert_close(logits[:, order], swapped, rtol=0, atol=0)
    assert (logits.softmax(-1)[~valid] == 0).all()
    assert torch.isfinite(logits[valid]).all()


def test_criterion_and_absent_features_have_explicit_masks():
    model, features, masks, options, valid = fixture()
    absent = torch.zeros(2, dtype=torch.bool)
    first = model(features, masks, options, valid)
    features["video"][1] = float("inf")
    second = model(features, masks, options, valid,
                   torch.full((2, 2), float("nan")), absent)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    with pytest.raises(ValueError, match="criterion"):
        model(features, masks, options, valid, torch.zeros(2, 2))


def test_present_criterion_receives_gradient_and_all_sources_missing_rejected():
    model, features, masks, options, valid = fixture()
    criterion = torch.randn(2, 2, requires_grad=True)
    logits = model(features, masks, options, valid, criterion, torch.ones(2).bool())
    logits[valid].sum().backward()
    assert torch.isfinite(criterion.grad).all() and criterion.grad.abs().sum() > 0
    masks["sensor"][1] = False
    with pytest.raises(ValueError, match="abstain"):
        model(features, masks, options, valid)


def test_reject_insufficient_candidates_and_nonfinite_available_option():
    model, features, masks, options, valid = fixture()
    valid[1, 1] = False
    with pytest.raises(ValueError, match="two"):
        model(features, masks, options, valid)
    valid[1, 1] = True
    options[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        model(features, masks, options, valid)
