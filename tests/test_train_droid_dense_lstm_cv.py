import pytest

torch = pytest.importorskip("torch")
from scripts.train_droid_dense_lstm_cv import SmallTemporalDecision  # noqa: E402


def test_lstm_parameter_count_shapes_and_no_carried_state():
    torch.set_num_threads(1)
    torch.manual_seed(17)
    model = SmallTemporalDecision()
    assert sum(p.numel() for p in model.parameters()) == 880
    sensor = torch.randn(4, 8, 14)
    previous = torch.nn.functional.one_hot(torch.tensor([0, 1, 2, 3]), 7).float()
    first = model(sensor, previous)
    assert first.shape == (4, 7)
    assert torch.equal(first, model(sensor, previous))
    first.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_lstm_rejects_future_step_and_bad_history_shape():
    model = SmallTemporalDecision()
    with pytest.raises(ValueError, match="sensor"):
        model(torch.zeros(1, 9, 14), torch.zeros(1, 7))
    with pytest.raises(ValueError, match="one-hot"):
        model(torch.zeros(1, 8, 14), torch.zeros(1, 8))
