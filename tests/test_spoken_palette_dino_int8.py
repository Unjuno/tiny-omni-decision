from __future__ import annotations

import io

import torch
from torch import nn

from scripts.pretrained_reuse.calibrate_spoken_palette_dino_int8 import (
    probabilities_at_temperature,
)
from scripts.pretrained_reuse.evaluate_spoken_palette_dino_int8 import (
    quantize_dynamic_int8,
)


def test_dynamic_int8_is_cpu_only_non_mutating_and_reloadable() -> None:
    torch.manual_seed(17)
    source = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 2)).eval()
    example = torch.randn(3, 4)

    quantized = quantize_dynamic_int8(source)
    assert isinstance(source[0], nn.Linear)
    assert isinstance(quantized[0], torch.ao.nn.quantized.dynamic.Linear)
    expected = quantized(example)
    assert expected.shape == (3, 2)
    assert torch.isfinite(expected).all()

    buffer = io.BytesIO()
    torch.save(quantized.state_dict(), buffer)
    buffer.seek(0)
    state = torch.load(buffer, map_location="cpu", weights_only=False)
    restored = quantize_dynamic_int8(
        nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 2)).eval()
    )
    restored.load_state_dict(state, strict=True)
    torch.testing.assert_close(restored(example), expected, rtol=0, atol=0)


def test_dynamic_int8_rejects_non_cpu_weights() -> None:
    model = nn.Linear(4, 2, device="meta")
    try:
        quantize_dynamic_int8(model)
    except ValueError as exc:
        assert "CPU modules only" in str(exc)
    else:
        raise AssertionError("non-CPU module should be rejected")


def test_temperature_changes_confidence_without_changing_candidate_order() -> None:
    logits = torch.tensor([[3.0, 1.0, -1.0], [-2.0, 0.0, 2.0]]).numpy()
    cool = probabilities_at_temperature(logits, 0.5)
    warm = probabilities_at_temperature(logits, 2.0)

    assert cool.argmax(axis=1).tolist() == warm.argmax(axis=1).tolist()
    assert abs(cool[0, 0] - warm[0, 0]) > 0.1
    assert torch.isfinite(torch.from_numpy(cool)).all()
    assert torch.allclose(
        torch.from_numpy(cool.sum(1)), torch.ones(2, dtype=torch.float64)
    )
