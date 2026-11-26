import pytest

torch = pytest.importorskip("torch")

from tiny_omni_decision.dcase2025_av_match import AVMatchReadout  # noqa: E402


def test_av_match_readout_is_candidate_permutation_equivariant_and_compact():
    torch.manual_seed(17)
    model = AVMatchReadout(rank=8)
    audio = torch.randn(3, 768)
    videos = torch.randn(3, 4, 768)
    permutation = torch.tensor([2, 0, 3, 1])

    logits = model(audio, videos)
    permuted = model(audio, videos[:, permutation])

    assert logits.shape == (3, 4)
    assert torch.isfinite(logits).all()
    assert torch.allclose(permuted, logits[:, permutation], atol=1e-7, rtol=0)
    assert sum(parameter.numel() for parameter in model.parameters()) == 12_288


def test_av_match_readout_abstains_to_uniform_when_either_stream_is_missing():
    model = AVMatchReadout(rank=8)
    audio = torch.randn(2, 768)
    videos = torch.randn(2, 4, 768)

    assert torch.allclose(model(torch.zeros_like(audio), videos), torch.zeros(2, 4))
    assert torch.allclose(model(audio, torch.zeros_like(videos)), torch.zeros(2, 4))


def test_av_match_readout_rejects_invalid_shapes():
    model = AVMatchReadout(rank=8)
    with pytest.raises(ValueError, match="shape"):
        model(torch.zeros(2, 768), torch.zeros(2, 3, 768))
