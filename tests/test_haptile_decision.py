from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from tiny_omni_decision.haptile_action import ACTION_LABELS  # noqa: E402
from tiny_omni_decision.haptile_decision import (  # noqa: E402
    HAPTILE_MODES,
    HapTileActionDecisionScorer,
)


def _features():
    torch.manual_seed(17)
    return (
        torch.randn(2, 4, 8, 768),
        torch.arange(8, dtype=torch.float32).repeat(2, 1) * 1000 / 7,
        torch.randn(2, 8, 25),
        torch.arange(8, dtype=torch.float32).repeat(2, 1) * 1000 / 7,
        torch.randn(2, 384),
        torch.randn(2, 7, 1536),
        torch.ones(2, 7, dtype=torch.bool),
    )


def test_haptile_decision_supports_frozen_input_ablation_modes():
    model = HapTileActionDecisionScorer()
    inputs = _features()

    for mode in HAPTILE_MODES:
        logits = model(*inputs, mode=mode)
        assert logits.shape == (2, 7)
        assert torch.isfinite(logits).all()

    assert model.parameter_count == 519_683
    assert model.comparison_parameter_count == 716_548


def test_haptile_decision_is_equivariant_to_supplied_option_order():
    model = HapTileActionDecisionScorer().eval()
    inputs = _features()
    with torch.inference_mode():
        logits = model(*inputs, mode="full")
        order = torch.tensor([6, 2, 0, 4, 1, 5, 3])
        changed = list(inputs)
        changed[5] = inputs[5][:, order]
        changed[6] = inputs[6][:, order]
        permuted = model(*changed, mode="full")

    torch.testing.assert_close(permuted, logits[:, order], rtol=0, atol=1e-6)


def test_haptile_multimodal_decision_reaches_temporal_and_fusion_parameters():
    model = HapTileActionDecisionScorer()
    inputs = _features()
    logits = model(*inputs, mode="full")
    torch.nn.functional.cross_entropy(logits, torch.tensor([1, 5])).backward()

    assert model.readout.sensor_stem.lstm.weight_ih_l0.grad is not None
    assert model.readout.fusion.attention.in_proj_weight.grad is not None
    assert model.readout.scorer.network[0].weight.grad is not None


def test_haptile_decide_returns_option_ordered_probability_json():
    model = HapTileActionDecisionScorer().eval()
    inputs = _features()
    order = [3, 0, 6, 2, 5, 1, 4]
    result = model.decide(
        inputs[0][0],
        inputs[1][0],
        inputs[2][0],
        inputs[3][0],
        inputs[4][0],
        inputs[5][0, order],
        [f"choice-{index}" for index in range(7)],
        [ACTION_LABELS[index] for index in order],
        mode="full",
    )

    assert result["status"] == "selected"
    assert [row["option_id"] for row in result["options"]] == [
        f"choice-{index}" for index in range(7)
    ]
    assert sum(row["probability"] for row in result["options"]) == pytest.approx(1.0)
