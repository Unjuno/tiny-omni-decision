from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from scripts.verify_textfree_audio_patha_feature_parity import _tensor_delta  # noqa: E402


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
