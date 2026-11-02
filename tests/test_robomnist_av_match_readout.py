import pytest
import torch

from tiny_omni_decision.robomnist_av_match_readout import (
    BilinearAVMatchScorer,
    SingleModalityMatchScorer,
)


def test_bilinear_readout_returns_binary_match_logits_with_exact_size():
    model = BilinearAVMatchScorer(rank=8)
    audio = torch.randn(3, 768)
    video = torch.randn(3, 1536)

    logits = model(audio, video)

    assert logits.shape == (3, 2)
    assert model.trainable_parameters == 8 * (768 + 1536)
    assert torch.isfinite(logits).all()
    assert torch.allclose(torch.softmax(logits, dim=-1).sum(-1), torch.ones(3))


def test_bilinear_readout_rejects_invalid_batch_and_nonfinite_inputs():
    model = BilinearAVMatchScorer(rank=4)
    with pytest.raises(ValueError, match="invalid batch"):
        model(torch.randn(2, 768), torch.randn(1, 1536))

    audio = torch.randn(1, 768)
    audio[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        model(audio, torch.randn(1, 1536))


def test_unimodal_control_has_fixed_two_option_output():
    model = SingleModalityMatchScorer(12)

    logits = model(torch.randn(5, 12))

    assert logits.shape == (5, 2)
    assert model.trainable_parameters == 13
