import torch

from scripts.train_inaturalist_frozen_image_readout import (
    _class_weights,
    _metrics,
    _option_order_max_delta,
)
from tiny_omni_decision.textfree_image import TextFreeImageReadout


def test_inverse_frequency_weights_equalize_class_total_and_mean_one():
    targets = torch.tensor([0, 0, 0, 0, 1, 1])
    weights = _class_weights(targets, 2)

    assert torch.isclose(weights.mean(), torch.tensor(1.0))
    assert torch.isclose(weights[0] * 4, weights[1] * 2)


def test_metrics_handle_uniform_probability_baseline():
    logits = torch.zeros((4, 2))
    targets = torch.tensor([0, 1, 0, 1])

    result = _metrics(logits, targets)

    assert result["count"] == 4
    assert result["accuracy"] == 0.5
    assert result["nll"] == 0.6931471824645996
    assert result["brier"] == 0.5


def test_probabilities_follow_option_identity_when_order_changes():
    torch.manual_seed(17)
    model = TextFreeImageReadout(4, {"tax": ("101", "202", "303")})
    features = torch.randn((5, 4))

    delta = _option_order_max_delta(model, "tax", features, [101, 202, 303])

    assert delta == 0.0
