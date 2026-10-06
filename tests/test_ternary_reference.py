from __future__ import annotations

import importlib
import io

import pytest
torch = pytest.importorskip('torch')

def module():
    return importlib.import_module('tiny_omni_decision.ternary_reference')

def test_quantizer_module_exists():
    assert importlib.util.find_spec('tiny_omni_decision.ternary_reference') is not None

def test_group_scale_tail_and_no_mutation():
    m = module()
    w = torch.tensor([[0.0, -1.0, 2.0, 0.1, 3.0]], requires_grad=True)
    old = w.detach().clone()
    q = m.quantize_reference(w, group_size=4)
    assert q.codes.tolist() == [[0, -1, 1, 0, 1]]
    assert q.scales.tolist() == [[1.5, 3.0]] and q.tail_lengths == (4, 1)
    assert q.shape == (1, 5) and q.codes.dtype == torch.int8 and (q.scales.dtype == torch.float32)
    torch.testing.assert_close(m.dequantize_reference(q,
         dtype=torch.float32),
         torch.tensor([[0.0,
         -1.5,
         1.5,
         0.0,
         3.0]]))
    assert not q.scales.requires_grad and w.grad is None
    torch.testing.assert_close(w, old)

@pytest.mark.parametrize('cols', [1, 127, 128, 129, 257])
def test_zeros_and_metadata_roundtrip(cols):
    m = module()
    w = torch.zeros(2, cols)
    q = m.quantize_reference(w)
    assert torch.count_nonzero(q.codes) == 0 and torch.count_nonzero(q.scales) == 0
    buffer = io.BytesIO()
    torch.save(q.to_state_dict(), buffer)
    buffer.seek(0)
    copy = m.TernaryTensor.from_state_dict(torch.load(buffer, weights_only=True))
    torch.testing.assert_close(m.dequantize_reference(copy, dtype=torch.bfloat16).float(), w)
    assert copy.tail_lengths == q.tail_lengths and copy.recipe_id == q.recipe_id

def test_threshold_ties_are_zero():
    m = module()
    q = m.quantize_reference(torch.tensor([[-1.0, 1.0]]), threshold_multiplier=1.0)
    assert q.codes.tolist() == [[0, 0]] and q.scales.tolist() == [[0.0]]

def test_quantizer_is_deterministic_and_group_local():
    m = module()
    w = torch.randn(3, 259, generator=torch.Generator().manual_seed(17))
    a = m.quantize_reference(w)
    b = m.quantize_reference(w)
    assert torch.equal(a.codes, b.codes) and torch.equal(a.scales, b.scales)
    assert set(a.codes.unique().tolist()).issubset({-1, 0, 1})
    restored = m.dequantize_reference(a, dtype=torch.float32)
    for j, n in enumerate(a.tail_lengths):
        sl = restored[:, j * 128:j * 128 + n]
        assert torch.all((sl == 0) | (sl.abs() == a.scales[:, j:j + 1]))

@pytest.mark.parametrize('kw',
     [{'group_size': 0},
     {'group_size': True},
     {'group_size': 1.5},
     {'threshold_multiplier': -0.1},
     {'threshold_multiplier': True},
     {'threshold_multiplier': float('inf')}])
def test_invalid_recipe_rejected(kw):
    with pytest.raises((ValueError, TypeError)):
        module().quantize_reference(torch.ones(2, 3), **kw)

@pytest.mark.parametrize('weight',
     [lambda: torch.tensor([[float('nan')]]),
     lambda: torch.tensor([[float('inf')]]),
     lambda: torch.empty(0,
     2),
     lambda: torch.zeros(2),
     lambda: torch.ones(2,
     2,
     dtype=torch.int32)])
def test_invalid_weights_rejected(weight):
    with pytest.raises((ValueError, TypeError)):
        module().quantize_reference(weight())

def test_corrupted_code_scale_and_metadata_rejected():
    m = module()
    q = m.quantize_reference(torch.ones(2, 3))
    for field, value in [('codes',
         torch.full((2,
         3),
         2,
         dtype=torch.int8)),
         ('scales',
         torch.full((2,
         1),
         float('nan'))),
         ('recipe_id',
         'unknown'),
         ('shape',
         [2,
         4]),
         ('tail_lengths',
         [2]),
         ('extra',
         True)]:
        data = q.to_state_dict()
        data[field] = value
        with pytest.raises((ValueError, TypeError)):
            m.TernaryTensor.from_state_dict(data)

def test_reference_storage_is_not_packed_and_backward_uses_constant_weight():
    m = module()
    q = m.quantize_reference(torch.ones(2, 5), group_size=4)
    info = q.storage_report()
    assert info['code_bytes'] == 10 and info['scale_bytes'] == 16
    assert info['packed'] is False and info['zero_count'] == 0
    x = torch.ones(1, 5, requires_grad=True)
    w = m.dequantize_reference(q, dtype=torch.float32)
    torch.nn.functional.linear(x, w).sum().backward()
    assert torch.equal(x.grad, torch.full_like(x, 2.0)) and (not w.requires_grad)
