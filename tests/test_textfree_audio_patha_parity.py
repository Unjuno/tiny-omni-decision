from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from scripts.verify_textfree_audio_patha_feature_parity import (  # noqa: E402
    _select_device,
    _tensor_delta,
)


def test_tensor_delta_reports_exact_parity() -> None:
    reference = torch.tensor([1.0, 2.0])
    result = _tensor_delta(reference, reference.clone())
    assert result == {"max_abs_delta": 0.0, "mean_abs_delta": 0.0, "allclose_1e-5": True}


def test_tensor_delta_reports_backend_drift() -> None:
    result = _tensor_delta(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.01]))
    assert result["max_abs_delta"] == pytest.approx(0.01, abs=1e-6)
    assert result["allclose_1e-5"] is False


def test_tensor_delta_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="equal shapes"):
        _tensor_delta(torch.ones(2), torch.ones(3))


def test_select_device_accepts_cpu() -> None:
    assert str(_select_device("cpu")) == "cpu"


def test_select_device_rejects_unknown_device() -> None:
    with pytest.raises(ValueError, match="device must be"):
        _select_device("mps")


def test_select_device_fails_closed_when_cuda_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA was requested but is unavailable"):
        _select_device("cuda")
