import numpy as np
import pytest


def test_real_frames_and_broadcast_have_identical_means_but_different_time_information():
    from scripts.lombard_temporal_core import visual_condition

    frames = np.arange(2 * 16 * 768, dtype=np.float32).reshape(2, 16, 768)
    pooled = frames.mean(axis=1)
    ordered = visual_condition(frames, pooled, "ordered")
    broadcast = visual_condition(frames, pooled, "broadcast")
    np.testing.assert_array_equal(ordered, frames)
    np.testing.assert_array_equal(broadcast.mean(1), pooled)
    assert np.all(broadcast[:, 0] == broadcast[:, -1])
    assert np.any(ordered[:, 0] != ordered[:, -1])


def test_invalid_or_misaligned_frame_banks_fail_closed():
    from scripts.lombard_temporal_core import visual_condition

    frames = np.zeros((2, 16, 768), np.float32)
    with pytest.raises(ValueError):
        visual_condition(frames[:, :8], frames.mean(1), "ordered")
    with pytest.raises(ValueError):
        visual_condition(frames, np.ones((2, 768), np.float32), "broadcast")
    with pytest.raises(ValueError):
        visual_condition(frames, frames.mean(1), "unknown")


def test_capacity_cpu_gradients_and_absent_visual_state():
    torch = pytest.importorskip("torch")
    from scripts.lombard_temporal_model import TemporalSlotHead

    torch.manual_seed(17)
    model = TemporalSlotHead()
    assert sum(p.numel() for p in model.parameters()) == 106355
    audio = torch.randn(2, 384)
    video = torch.randn(2, 16, 768)
    present = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    first = model(audio, video, present)
    second = model(audio, video.flip(1), present)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert first.shape == (2, 51)
    first.square().sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_ordered_head_can_observe_order_and_fresh_reload_is_exact():
    torch = pytest.importorskip("torch")
    from scripts.lombard_temporal_model import TemporalSlotHead

    torch.manual_seed(17)
    model = TemporalSlotHead().eval()
    audio = torch.zeros(2, 384)
    video = torch.randn(2, 16, 768)
    present = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    with torch.no_grad():
        output = model(audio, video, present)
        reverse = model(audio, video.flip(1), present)
        other = TemporalSlotHead().eval()
        other.load_state_dict(model.state_dict())
        actual = other(audio, video, present)
    assert not torch.equal(output, reverse)
    torch.testing.assert_close(output, actual, rtol=0, atol=0)


def test_sampler_independence_and_optimizer_accounting():
    torch = pytest.importorskip("torch")
    from scripts.train_lombard_temporal import orders_for

    before = torch.get_rng_state().clone()
    first = orders_for(4080, 24)
    second = orders_for(4080, 24)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert first.shape == (24, 4080)
    assert all(len(set(r.tolist())) == 4080 for r in first)
    torch.testing.assert_close(before, torch.get_rng_state(), rtol=0, atol=0)


def test_fixed_cpu_resource_guard_rejects_limits(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from scripts import train_lombard_temporal as module

    monkeypatch.setattr(module, "peak_bytes", lambda: 4 * 2**30)
    with pytest.raises(RuntimeError, match="resource"):
        module.guard(tmp_path, module.time.perf_counter())
