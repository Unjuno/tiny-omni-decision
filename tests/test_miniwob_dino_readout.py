import pytest

torch = pytest.importorskip("torch")

from tiny_omni_decision.miniwob_dino_readout import (  # noqa: E402
    MiniWoBOptionScorer,
    classification_metrics,
    deranged_indices,
    permute_candidate_axis,
)


def test_candidate_scorer_is_permutation_equivariant():
    torch.manual_seed(17)
    model = MiniWoBOptionScorer().eval()
    features = torch.randn(3, 4, 1154)
    permutation = torch.tensor([2, 0, 3, 1])
    reordered, inverse = permute_candidate_axis(features, permutation)
    with torch.inference_mode():
        expected = model(features).index_select(1, permutation)
        actual = model(reordered).index_select(1, inverse)
    torch.testing.assert_close(actual, model(features), rtol=0, atol=0)
    assert torch.equal(expected.index_select(1, inverse), model(features))


def test_metrics_match_four_choice_reference_values():
    probabilities = torch.tensor([[0.7, 0.1, 0.1, 0.1], [0.25, 0.25, 0.25, 0.25]])
    targets = torch.tensor([0, 1])
    metrics = classification_metrics(probabilities, targets)
    assert metrics["count"] == 2
    assert metrics["accuracy"] == 0.5
    assert metrics["brier"] == pytest.approx((0.12 + 0.75) / 2)
    expected_nll = (-torch.log(torch.tensor(0.7)) + torch.log(torch.tensor(4.0))) / 2
    assert metrics["nll"] == pytest.approx(expected_nll.item())


def test_scorer_rejects_wrong_shapes_and_nonfinite_inputs():
    model = MiniWoBOptionScorer()
    with pytest.raises(ValueError):
        model(torch.zeros(2, 1154))
    invalid = torch.zeros(2, 4, 1154)
    invalid[0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        model(invalid)


def test_metric_validation_rejects_non_normalized_probabilities():
    with pytest.raises(ValueError):
        classification_metrics(torch.tensor([[0.5, 0.2, 0.2, 0.2]]), torch.tensor([0]))


def test_screen_derangement_is_deterministic_complete_and_has_no_fixed_points():
    first = deranged_indices(16, seed=17)
    assert first == deranged_indices(16, seed=17)
    assert sorted(first) == list(range(16))
    assert all(index != donor for index, donor in enumerate(first))
    with pytest.raises(ValueError):
        deranged_indices(1, seed=17)
