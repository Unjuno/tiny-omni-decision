import pytest
import torch

from tiny_omni_decision.perception_video_readout import PerceptionVideoDecisionReadout


def test_video_readout_returns_finite_candidate_logits():
    torch.manual_seed(17)
    model = PerceptionVideoDecisionReadout().eval()
    video = torch.randn(2, 8 * 14 * 14, 384)
    queries = torch.randn(2, 3, 384)

    logits = model(video, queries)

    assert logits.shape == (2, 3)
    assert torch.isfinite(logits).all()
    assert model.parameter_count() < 1_000_000


def test_video_readout_is_equivariant_to_option_permutation():
    torch.manual_seed(17)
    model = PerceptionVideoDecisionReadout().eval()
    video = torch.randn(1, 16, 384)
    queries = torch.randn(1, 4, 384)
    permutation = torch.tensor([2, 0, 3, 1])

    original = model(video, queries)
    reordered = model(video, queries[:, permutation])

    torch.testing.assert_close(reordered, original[:, permutation], rtol=0, atol=1e-6)


@pytest.mark.parametrize(
    "video,queries",
    [
        (torch.zeros(1, 0, 384), torch.zeros(1, 2, 384)),
        (torch.zeros(1, 2, 383), torch.zeros(1, 2, 384)),
        (torch.zeros(1, 2, 384), torch.zeros(1, 1, 384)),
        (torch.zeros(1, 2, 384), torch.full((1, 2, 384), float("nan"))),
    ],
)
def test_video_readout_rejects_invalid_inputs(video, queries):
    with pytest.raises(ValueError):
        PerceptionVideoDecisionReadout()(video, queries)
