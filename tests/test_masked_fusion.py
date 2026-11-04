import pytest

pytest.importorskip('torch')
import torch

from tiny_omni_decision.masked_fusion import MaskedModalityFusion


def test_absent_values_cannot_change_output_or_receive_gradient():
    torch.manual_seed(17)
    model = MaskedModalityFusion({'video': 4, 'sensor': 2}, width=8)
    video = torch.randn(2, 4, requires_grad=True)
    sensor = torch.randn(2, 2, requires_grad=True)
    masks = {'video': torch.tensor([True, False]), 'sensor': torch.tensor([True, True])}
    output, weights = model({'video': video, 'sensor': sensor}, masks)
    changed = video.detach().clone()
    changed[1] = float('nan')
    other, _ = model({'video': changed, 'sensor': sensor}, masks)
    torch.testing.assert_close(output, other, rtol=0, atol=0)
    assert weights[1, model.names.index('video')].item() == 0
    output.sum().backward()
    assert torch.equal(video.grad[1], torch.zeros(4))
    assert torch.isfinite(sensor.grad).all()


def test_all_missing_rejected():
    model = MaskedModalityFusion({'audio': 3})
    with pytest.raises(ValueError, match='abstain'):
        model({'audio': torch.zeros(1, 3)}, {'audio': torch.tensor([False])})


def test_present_zero_is_valid_and_nonfinite_present_rejected():
    model = MaskedModalityFusion({'audio': 3})
    mask = {'audio': torch.tensor([True])}
    out, weights = model({'audio': torch.zeros(1, 3)}, mask)
    assert torch.isfinite(out).all() and weights.item() == 1
    with pytest.raises(ValueError, match='nonfinite'):
        model({'audio': torch.full((1, 3), float('nan'))}, mask)

