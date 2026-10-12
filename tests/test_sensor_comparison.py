import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scripts.train_sensor_subjects import StaticSensor, metrics  # noqa: E402
from tiny_omni_decision.native_sensor import NativeSensorDecision  # noqa: E402


def test_center_readout_keeps_parameters_and_local_support():
    torch.manual_seed(17)
    mean = NativeSensorDecision()
    torch.manual_seed(17)
    center = NativeSensorDecision(readout="center")
    assert all(torch.equal(a, b) for a, b in zip(mean.parameters(), center.parameters()))
    x = torch.randn(2, 128, 6)
    t = torch.arange(128).expand(2, -1).double() * 20
    changed = x.clone()
    changed[:, :20] += 100
    assert torch.equal(center(x, t), center(changed, t))
    with pytest.raises(ValueError):
        NativeSensorDecision(readout="unknown")


def test_static_is_order_invariant():
    torch.manual_seed(17)
    model = StaticSensor().eval()
    x = torch.randn(3, 128, 6)
    t = torch.arange(128).expand(3, -1).double() * 20
    assert torch.allclose(model(x, t), model(x.flip(1), t), atol=1e-6)


def test_metrics_perfect_and_missing_class():
    result = metrics(np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([0, 1]))
    assert result["accuracy"] == result["balanced_accuracy"] == 1
    assert result["brier"] == result["ece"] == 0
    absent = metrics(np.array([[1.0, 0.0]]), np.array([0]))
    assert absent["classes"]["1"]["recall"] is None
