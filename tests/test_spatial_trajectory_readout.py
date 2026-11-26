import pytest

torch = pytest.importorskip("torch")
from tiny_omni_decision.spatial_trajectory_readout import SpatialTrajectoryReadout  # noqa: E402


def inputs():
    torch.manual_seed(17)
    return torch.randn(2, 5, 512), torch.randn(2, 6, 1536), torch.randn(2, 9, 387)


def test_state_reuse_equivariance_and_all_gradients():
    points, candidates, context = inputs()
    model = SpatialTrajectoryReadout(hidden=8, latent_count=3, heads=2)
    qa, cover = model(points, candidates, context)
    state = model.encode_state(points, context)
    torch.testing.assert_close(model.score_candidates(state, candidates), qa, rtol=0, atol=0)
    reversed_qa, reversed_cover = model(points.flip(1), candidates.flip(1), context)
    torch.testing.assert_close(reversed_qa, qa.flip(1))
    torch.testing.assert_close(reversed_cover, cover.flip(1))
    torch.testing.assert_close(model(points, candidates, context.flip(1))[0], qa)
    (qa.square().mean() + cover.square().mean()).backward()
    assert all(p.grad is not None and p.grad.isfinite().all() for p in model.parameters())
    assert model.context_projection.weight.grad.abs().sum() > 0
    assert model.latents.grad.abs().sum() > 0


def test_both_point_and_context_padding_are_inert():
    points, candidates, context = inputs()
    model = SpatialTrajectoryReadout(hidden=8, latent_count=3, heads=2)
    expected = model(points, candidates, context)
    points_pad = torch.cat([points, torch.full((2, 2, 512), 999.0)], 1)
    context_pad = torch.cat([context, torch.full((2, 2, 387), 999.0)], 1)
    mask = torch.tensor([[True] * 5 + [False] * 2] * 2)
    cmask = torch.tensor([[True] * 9 + [False] * 2] * 2)
    actual = model(points_pad, candidates, context_pad, mask, cmask)
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1][:, :5], expected[1])
    assert torch.isneginf(actual[1][:, 5:]).all()


def test_checkpoint_and_transient_tensor_state_reload(tmp_path):
    points, candidates, context = inputs()
    model = SpatialTrajectoryReadout(hidden=8, latent_count=3, heads=2).eval()
    state = model.encode_state(points, context)
    weights, encoded = tmp_path / "weights.pt", tmp_path / "state.pt"
    torch.save(model.state_dict(), weights)
    torch.save(state, encoded)
    restored = SpatialTrajectoryReadout(hidden=8, latent_count=3, heads=2).eval()
    restored.load_state_dict(torch.load(weights, weights_only=True), strict=True)
    loaded_state = torch.load(encoded, weights_only=True)
    torch.testing.assert_close(
        restored(points, candidates, context)[0],
        model(points, candidates, context)[0],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        restored.score_candidates(loaded_state, candidates),
        model.score_candidates(state, candidates),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("kind", ["empty", "shape", "batch", "nan", "mask", "candidates"])
def test_invalid_context_or_candidates_fail_closed(kind):
    points, candidates, context = inputs()
    model = SpatialTrajectoryReadout(hidden=8, latent_count=3, heads=2)
    mask = None
    if kind == "empty":
        context = context[:, :0]
    elif kind == "shape":
        context = context[..., :386]
    elif kind == "batch":
        context = context[:1]
    elif kind == "nan":
        context[0, 0, 0] = float("nan")
    elif kind == "mask":
        mask = torch.zeros(2, 9, dtype=torch.bool)
    else:
        candidates = candidates[:, :0]
    with pytest.raises(ValueError):
        model(points, candidates, context, context_mask=mask)
