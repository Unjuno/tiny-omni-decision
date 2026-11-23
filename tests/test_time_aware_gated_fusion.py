# ruff: noqa: E402

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", exc_type=ImportError)

from tiny_omni_decision.time_aware_gated_fusion import TimeAwareGatedFusion


def fixture():
    torch.manual_seed(17)
    model = TimeAwareGatedFusion(hidden_size=8, modality_count=3, time_scale_seconds=2.0)
    features = torch.randn(2, 4, 3, 8)
    available = torch.tensor(
        [
            [[True, False, False], [True, True, False], [False, False, True], [False] * 3],
            [[False, True, False], [True, True, False], [True, False, False], [False, False, True]],
        ],
        dtype=torch.bool,
    )
    timestamps = torch.tensor([[0.0, 0.5, 2.0, float("nan")], [0.0, 0.7, 1.5, 4.0]])
    features[~available] = float("nan")
    return model, features, timestamps, available


def test_time_aware_fusion_masks_missing_modalities_and_padding():
    model, features, timestamps, available = fixture()
    result = model(features, timestamps, available)

    assert result.fused.shape == (2, 8)
    assert result.states.shape == (2, 4, 8)
    assert result.modality_weights.shape == (2, 4, 3)
    assert result.update_gates.shape == (2, 4, 8)
    assert torch.isfinite(result.fused).all()
    assert torch.isfinite(result.states).all()
    assert (result.modality_weights[~available] == 0).all()
    torch.testing.assert_close(
        result.modality_weights.sum(-1),
        available.any(-1).to(result.modality_weights.dtype),
    )
    assert result.states[0, 3].count_nonzero() == 0


def test_fusion_is_causal_and_differentiable():
    model, features, timestamps, available = fixture()
    features = features.clone().requires_grad_(True)
    baseline = model(features, timestamps, available)

    changed_future = features.detach().clone()
    changed_future[0, 2, 2] += 3.0
    counterfactual = model(changed_future, timestamps, available)
    torch.testing.assert_close(baseline.states[0, :2], counterfactual.states[0, :2])
    assert not torch.equal(baseline.fused[0], counterfactual.fused[0])

    baseline.fused.square().sum().backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad[available]).all()
    assert features.grad[available].abs().sum() > 0


def test_masked_payload_values_cannot_change_fusion():
    model, features, timestamps, available = fixture()
    ignored_as_nan = model(features, timestamps, available)
    zeroed = torch.where(available[..., None], features, torch.zeros_like(features))
    ignored_as_zero = model(zeroed, timestamps, available)

    torch.testing.assert_close(ignored_as_nan.fused, ignored_as_zero.fused, rtol=0, atol=0)
    torch.testing.assert_close(ignored_as_nan.states, ignored_as_zero.states, rtol=0, atol=0)


def test_out_of_order_valid_timestamps_are_rejected():
    model, features, timestamps, available = fixture()
    timestamps[1, 2] = 0.2
    with pytest.raises(ValueError, match="nondecreasing"):
        model(features, timestamps, available)


def test_all_missing_sample_is_rejected_and_parameter_budget_is_small():
    model, features, timestamps, available = fixture()
    available[0] = False
    with pytest.raises(ValueError, match="at least one available"):
        model(features, timestamps, available)

    assert sum(parameter.numel() for parameter in model.parameters()) < 100_000
