from __future__ import annotations

import importlib
import math
from pathlib import Path

import pytest

torch = pytest.importorskip('torch')

def api():
    return importlib.import_module('tiny_omni_decision.postquant_loss').option_distillation_loss

def weights(**overrides):
    from tiny_omni_decision.postquant_policy import LossWeights
    return LossWeights(stage='recovery',
         **{'option_kl': 1.0,
         'cross_entropy': 0.2,
         'brier': 0.2} | overrides)

def test_loss_module_exists():
    assert importlib.util.find_spec('tiny_omni_decision.postquant_loss') is not None

@pytest.mark.parametrize('count', [2, 4, 10, 60, 62])
def test_uniform_loss_math(count):
    s = torch.zeros(2, count, requires_grad=True)
    t = s.detach().clone().requires_grad_()
    result = api()(s,
         t,
         torch.tensor([0,
         count - 1]),
         torch.ones_like(s,
         dtype=torch.bool),
         weights())
    assert set(result) == {'total', 'option_kl', 'cross_entropy', 'brier'}
    assert result['option_kl'].item() == pytest.approx(0, abs=1e-06)
    assert result['cross_entropy'].item() == pytest.approx(math.log(count), abs=1e-06)
    assert result['brier'].item() == pytest.approx(1 - 1 / count, abs=1e-06)
    result['total'].backward()
    assert torch.isfinite(s.grad).all() and t.grad is None

def test_kl_is_sum_options_not_mean_options():
    s = torch.tensor([[0.0, 1.0, 2.0, 3.0]], requires_grad=True)
    t = torch.tensor([[3.0, 2.0, 1.0, 0.0]])
    actual = api()(s, t, torch.tensor([1]), torch.ones_like(s, dtype=torch.bool), weights())
    expected = torch.nn.functional.kl_div(s.log_softmax(-1),
         t.log_softmax(-1),
         log_target=True,
         reduction='batchmean')
    torch.testing.assert_close(actual['option_kl'], expected)

@pytest.mark.parametrize('pad', [1e+30, float('inf'), float('nan')])
def test_padding_cannot_change_loss_or_gradients(pad):
    s = torch.tensor([[1.0, -2.0, pad, pad], [0.0, 1.0, -1.0, 2.0]], requires_grad=True)
    t = torch.tensor([[-1.0, 2.0, pad, pad], [1.0, 2.0, 3.0, 4.0]], requires_grad=True)
    mask = torch.tensor([[True, True, False, False], [True] * 4])
    y = torch.tensor([1, 2])
    got = api()(s, t, y, mask, weights())
    items = [api()(s[:1,
         :2],
         t[:1,
         :2],
         y[:1],
         mask[:1,
         :2],
         weights()),
         api()(s[1:],
         t[1:],
         y[1:],
         mask[1:],
         weights())]
    for key in got:
        torch.testing.assert_close(got[key], (items[0][key] + items[1][key]) / 2)
    got['total'].backward()
    assert torch.equal(s.grad[~mask], torch.zeros_like(s.grad[~mask]))
    assert torch.isfinite(s.grad).all() and t.grad is None

def test_permutations_preserve_losses_and_gradients():
    s = torch.tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
    t = torch.tensor([[2.0, 4.0, 1.0, 3.0]])
    mask = torch.tensor([[True, False, True, True]])
    y = torch.tensor([2])
    perm = torch.tensor([3, 2, 0, 1])
    orig = api()(s, t, y, mask, weights())
    moved = api()(s[:, perm], t[:, perm], torch.tensor([1]), mask[:, perm], weights())
    for key in orig:
        torch.testing.assert_close(orig[key], moved[key])
    g1 = torch.autograd.grad(orig['total'], s, retain_graph=True)[0]
    g2 = torch.autograd.grad(moved['total'], s)[0]
    torch.testing.assert_close(g1, g2)

@pytest.mark.parametrize('fault',
     ['nan',
     'inf',
     'shape',
     'target',
     'mask_type',
     'one_option',
     'too_many',
     'target_pad',
     'float_target',
     'empty'])
def test_bad_inputs_rejected(fault):
    s = torch.zeros(2, 4)
    t = s.clone()
    mask = torch.ones_like(s, dtype=torch.bool)
    y = torch.tensor([0, 1])
    if fault == 'nan':
        s[0, 0] = float('nan')
    if fault == 'inf':
        t[0, 0] = float('inf')
    if fault == 'shape':
        t = t[:1]
    if fault == 'target':
        y[0] = 4
    if fault == 'mask_type':
        mask = mask.int()
    if fault == 'one_option':
        mask[:, 1:] = False
    if fault == 'too_many':
        s = t = torch.zeros(2, 63)
        mask = torch.ones_like(s, dtype=torch.bool)
    if fault == 'target_pad':
        mask[0, 0] = False
    if fault == 'float_target':
        y = y.float()
    if fault == 'empty':
        s = t = torch.zeros(0, 4)
        mask = torch.ones_like(s, dtype=torch.bool)
        y = torch.empty(0, dtype=torch.long)
    with pytest.raises((ValueError, TypeError)):
        api()(s, t, y, mask, weights())

def test_gradient_accumulation_matches_one_batch():
    g = torch.Generator().manual_seed(17)
    s = torch.randn(4, 10, generator=g, requires_grad=True)
    t = torch.randn(4, 10, generator=g)
    y = torch.tensor([1, 3, 5, 7])
    mask = torch.ones_like(s, dtype=torch.bool)
    api()(s, t, y, mask, weights())['total'].backward()
    full = s.grad.clone()
    s.grad = None
    for i in range(4):
        (api()(s[i:i + 1],
             t[i:i + 1],
             y[i:i + 1],
             mask[i:i + 1],
             weights())['total'] / 4).backward()
    torch.testing.assert_close(s.grad, full)

def test_stage_weights_reach_real_gradient_not_only_yaml():
    from tiny_omni_decision.postquant_policy import (
        PostQuantConfig,
        load_postquant_config,
        loss_weights_for_update,
    )
    path = Path(__file__).parents[1] / 'configs/recovery/decision_respecialization_v2.example.yaml'
    c = load_postquant_config(path)
    c = PostQuantConfig.model_validate(c.model_dump())
    s = torch.tensor([[0.0, 0.0]], requires_grad=True)
    t = torch.tensor([[3.0, -3.0]])
    mask = torch.ones_like(s, dtype=torch.bool)
    y = torch.tensor([1])
    first = api()(s, t, y, mask, loss_weights_for_update(0, c))['total']
    last = api()(s, t, y, mask, loss_weights_for_update(1023, c))['total']
    g1 = torch.autograd.grad(first, s)[0]
    g2 = torch.autograd.grad(last, s)[0]
    assert g1[0, 0] < 0 and g2[0, 0] > 0

def test_bfloat16_inputs_still_compute_fp32_loss():
    s = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16, requires_grad=True)
    r = api()(s, s.detach(), torch.tensor([1]), torch.ones_like(s, dtype=torch.bool), weights())
    assert all(v.dtype == torch.float32 for v in r.values())
    r['total'].backward()
    assert torch.isfinite(s.grad).all()
