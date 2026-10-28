import pytest
import torch

from scripts.fit_eidon_temporal_matcher import PairMatcher


@pytest.mark.parametrize("mode", ["pooled", "lstm"])
def test_matcher_scores_four_candidates_and_is_option_order_equivariant(mode):
    torch.manual_seed(17)
    model = PairMatcher(mode).eval()
    video = torch.randn(2, 8, 768)
    sensors = torch.randn(2, 4, 8, 54)
    with torch.inference_mode():
        scores = model(video, sensors)
        reordered = model(video, sensors.flip(1))
    assert scores.shape == (2, 4)
    assert torch.isfinite(scores).all()
    torch.testing.assert_close(reordered, scores.flip(1), rtol=1e-6, atol=1e-7)


def test_matcher_rejects_non_four_choice_or_wrong_temporal_shape():
    model = PairMatcher("pooled")
    with pytest.raises(ValueError, match="expected video"):
        model(torch.randn(1, 8, 768), torch.randn(1, 3, 8, 54))
