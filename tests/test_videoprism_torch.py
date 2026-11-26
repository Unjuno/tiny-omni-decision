import numpy as np
import pytest

torch = pytest.importorskip('torch')

from tiny_omni_decision.videoprism_torch import (  # noqa: E402
    Config,
    FrozenVideoPrism,
    capped_attention,
    expected_shapes,
    patchify,
    source_layer_norm,
)


def test_patch_order_matches_row_column_pixel_channel_source_convention():
    x = torch.arange(2 * 4 * 4 * 3).reshape(1, 2, 4, 4, 3).float()
    y = patchify(x, 2)
    assert y.shape == (2, 4, 12)
    torch.testing.assert_close(y[0, 0], x[0, 0, :2, :2].reshape(-1))
    torch.testing.assert_close(y[0, 1], x[0, 0, :2, 2:].reshape(-1))
    torch.testing.assert_close(y[1, 3], x[0, 1, 2:, 2:].reshape(-1))


def test_source_layer_norm_uses_scale_plus_one_and_epsilon():
    x = torch.tensor([[1., 3., 6.]])
    scale, bias = torch.tensor([0., .5, -1.]), torch.tensor([.2, .3, .4])
    centered = x - x.mean(-1, keepdim=True)
    expected = centered / (centered.square().mean(-1, keepdim=True) + 1e-6).sqrt()
    expected = expected * (scale + 1) + bias
    torch.testing.assert_close(source_layer_norm(x, scale, bias), expected)


def test_capped_attention_matches_independent_numpy_equations():
    rng = np.random.default_rng(17)
    q, k, v = [rng.normal(size=(1, 3, 2, 4)).astype('float32') for _ in range(3)]
    q *= 40  # distinguishes softcap from ordinary attention
    logits = np.einsum('btnh,bsnh->bnts', q / 2, k)
    logits = 50 * np.tanh(logits / 50)
    exp = np.exp(logits - logits.max(-1, keepdims=True))
    expected = np.einsum('bnts,bsnh->btnh', exp / exp.sum(-1, keepdims=True), v)
    actual = capped_attention(*[torch.from_numpy(a) for a in (q, k, v)])
    np.testing.assert_allclose(actual.numpy(), expected, atol=1e-5, rtol=1e-5)


def tiny():
    config = Config(width=8, patch=2, frames=2, grid=2, heads=2,
                    spatial_layers=1, temporal_layers=1, mlp=16)
    rng = np.random.default_rng(17)
    weights = {k: rng.normal(0, .02, shape).astype('float32')
               for k, shape in expected_shapes(config).items()}
    return weights, config


def test_complete_frozen_model_repeats_reloads_and_preserves_output_order():
    weights, config = tiny()
    model = FrozenVideoPrism(weights, config)
    x = torch.rand(1, 2, 4, 4, 3)
    with torch.inference_mode():
        y = model(x)
        assert y.shape == (1, 8, 8)
        assert torch.equal(y, model(x))
        reload = FrozenVideoPrism(weights, config)
        reload.load_state_dict(model.state_dict(), strict=True)
        assert torch.equal(y, reload(x))
    assert all(not p.requires_grad for p in model.parameters())
    assert sum(p.numel() for p in model.parameters()) == sum(v.size for v in weights.values())
    assert len(model.state_dict()) == 40


@pytest.mark.parametrize('fault', ['missing', 'extra', 'shape', 'dtype', 'nan'])
def test_checkpoint_schema_fails_closed(fault):
    weights, config = tiny()
    key = next(iter(weights))
    if fault == 'missing':
        del weights[key]
    elif fault == 'extra':
        weights['unreviewed'] = np.ones(1, dtype='float32')
    elif fault == 'shape':
        weights[key] = np.ones(1, dtype='float32')
    elif fault == 'dtype':
        weights[key] = weights[key].astype('float64')
    else:
        weights[key].flat[0] = np.nan
    with pytest.raises(ValueError):
        FrozenVideoPrism(weights, config)


def test_training_precision_and_invalid_inputs_are_not_silently_supported():
    weights, config = tiny()
    model = FrozenVideoPrism(weights, config)
    x = torch.rand(1, 2, 4, 4, 3)
    for bad in [x[:, :1], x.double(), x * 256, x * float('nan')]:
        with pytest.raises(ValueError):
            model(bad)
    model.train()
    with pytest.raises(ValueError, match='inference'):
        model(x)
    model.eval().half()
    with pytest.raises(ValueError):
        model(x)


def test_ambient_autocast_cannot_silently_reduce_native_precision():
    weights, config = tiny()
    model = FrozenVideoPrism(weights, config)
    with torch.autocast('cpu', dtype=torch.bfloat16), pytest.raises(ValueError, match='FP32'):
        model(torch.rand(1, 2, 4, 4, 3))


@pytest.mark.parametrize('fault', ['missing', 'extra', 'dtype', 'shape', 'nan'])
def test_state_reload_rejects_invalid_complete_state_without_partial_mutation(fault):
    weights, config = tiny()
    model = FrozenVideoPrism(weights, config)
    original = {k: v.clone() for k, v in model.state_dict().items()}
    state = {k: v.clone() for k, v in original.items()}
    first, last = next(iter(state)), next(reversed(state))
    state[first].add_(1)  # would mutate the model before a late default-loader failure
    if fault == 'missing':
        del state[last]
    elif fault == 'extra':
        state['unreviewed'] = torch.ones(1)
    elif fault == 'dtype':
        state[last] = state[last].double()
    elif fault == 'shape':
        state[last] = torch.ones(1)
    else:
        state[last].view(-1)[0] = float('nan')
    with pytest.raises(ValueError, match='state'):
        model.load_state_dict(state, strict=True)
    assert all(torch.equal(v, original[k]) for k, v in model.state_dict().items())


@pytest.mark.parametrize('mode', ['assign', 'non_strict', 'target_half'])
def test_reload_cannot_bypass_frozen_fp32_copy_policy(mode):
    weights, config = tiny()
    model = FrozenVideoPrism(weights, config)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    if mode == 'target_half':
        model.half()
    original = {k: v.clone() for k, v in model.state_dict().items()}
    kwargs = ({'assign': True} if mode == 'assign'
              else {'strict': False} if mode == 'non_strict' else {})
    with pytest.raises(ValueError, match='state'):
        model.load_state_dict(state, **kwargs)
    assert all(torch.equal(v, original[k]) for k, v in model.state_dict().items())
