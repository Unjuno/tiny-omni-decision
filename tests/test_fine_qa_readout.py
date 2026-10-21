import pytest

torch = pytest.importorskip("torch")

from tiny_omni_decision.spatial_trajectory_readout import SpatialTrajectoryReadout  # noqa: E402


def test_dynamic_token_count_preserves_capacity_state_reuse_and_candidate_mapping():
    torch.manual_seed(17)
    model = SpatialTrajectoryReadout().eval()
    assert sum(p.numel() for p in model.parameters()) == 232385
    points = torch.randn(2, 3, 512)
    context = torch.randn(2, 4, 387)
    candidates = torch.randn(2, 6, 1536)
    with torch.no_grad():
        logits, _ = model(points, candidates, context)
        state = model.encode_state(points, context)
        torch.testing.assert_close(
            model.score_candidates(state, candidates), logits, rtol=0, atol=0
        )
        repeated = context.repeat_interleave(4, dim=1)
        widened, _ = model(points, candidates, repeated)
        torch.testing.assert_close(widened, logits, rtol=1e-5, atol=1e-6)
        flipped = candidates.reshape(2, 2, 3, 1536).flip(2).reshape(2, 6, 1536)
        reversed_scores = model.score_candidates(state, flipped).reshape(2, 2, 3)
        torch.testing.assert_close(
            reversed_scores, logits.reshape(2, 2, 3).flip(2), rtol=1e-5, atol=1e-6
        )
    model.train()
    result, _ = model(points, candidates, repeated)
    result.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
