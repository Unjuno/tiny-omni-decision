import pytest

torch = pytest.importorskip("torch")

from tiny_omni_decision.dcase2025_readout import (  # noqa: E402
    DCASE2025LinearReadout,
    DCASE2025Readout,
    binary_probability_metrics,
)


@pytest.mark.parametrize("mode", ["audio", "video", "joint"])
def test_dcase_readout_scores_fixed_binary_options_with_event_context(mode):
    torch.manual_seed(17)
    model = DCASE2025Readout(mode, hidden_width=8)
    logits = model(torch.zeros(3, 768), torch.ones(3, 768), torch.tensor([0, 4, 12]))
    assert logits.shape == (3, 2)
    assert torch.isfinite(logits).all()


def test_dcase_readout_rejects_out_of_range_event_context():
    model = DCASE2025Readout("joint", hidden_width=8)
    with pytest.raises(ValueError, match="taxonomy"):
        model(torch.zeros(1, 768), torch.zeros(1, 768), torch.tensor([13]))


@pytest.mark.parametrize(
    "mode,expected_parameters", [("audio", 1564), ("video", 1564), ("joint", 3100)]
)
def test_linear_readout_is_affine_and_has_expected_parameter_budget(mode, expected_parameters):
    model = DCASE2025LinearReadout(mode)
    logits = model(torch.zeros(2, 768), torch.ones(2, 768), torch.tensor([0, 12]))

    assert logits.shape == (2, 2)
    assert torch.isfinite(logits).all()
    assert sum(parameter.numel() for parameter in model.parameters()) == expected_parameters
    assert len(list(model.modules())) == 2


def test_binary_metrics_report_balanced_accuracy_and_calibration():
    probabilities = [[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.4, 0.6]]
    metrics = binary_probability_metrics(probabilities, [0, 1, 1, 0])
    assert metrics["accuracy"] == 0.5
    assert metrics["balanced_accuracy"] == 0.5
    assert metrics["recall_offscreen"] == 0.5
    assert metrics["recall_onscreen"] == 0.5
    assert metrics["nll"] > 0
    assert metrics["brier"] > 0
    assert 0 <= metrics["ece_15"] <= 1


def test_binary_metrics_reject_unnormalized_or_non_binary_targets():
    with pytest.raises(ValueError, match="normalized"):
        binary_probability_metrics([[0.6, 0.6]], [0])
    with pytest.raises(ValueError, match="normalized"):
        binary_probability_metrics([[0.5, 0.5]], [2])
