"""Arithmetic/gradient contracts for group-broadcast quantization."""

import math

import pytest

torch = pytest.importorskip("torch")

from tiny_omni_decision.ternary import (  # noqa: E402
    dequantize_groupwise_ternary,
    fake_quantize_ternary,
    quantize_groupwise_ternary,
)


@pytest.mark.parametrize("group_size", [1, 3, 16, 64])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_groups_match_scalar_oracle_on_noncontiguous_input(group_size, dtype):
    # Includes zeros, asymmetric signs, a partial group, and a group bigger
    # than the tensor. Integer inputs make oracle reductions unambiguous.
    weights = ((torch.arange(35).reshape(5, 7) % 9) - 4).to(dtype).T
    assert not weights.is_contiguous()
    values = weights.reshape(-1).tolist()
    expected_codes, expected_scales = [], []
    for start in range(0, len(values), group_size):
        group = values[start:start + group_size]
        threshold = 0.5 * math.fsum(abs(v) for v in group) / len(group)
        active = [abs(v) > 0 and abs(v) >= threshold for v in group]
        scale = math.fsum(abs(v) for v, keep in zip(group, active, strict=True) if keep)
        scale /= max(1, sum(active))
        expected_scales.append(scale)
        expected_codes.extend(
            (1 if v > 0 else -1) if keep else 0
            for v, keep in zip(group, active, strict=True)
        )
    codes, scales = quantize_groupwise_ternary(
        weights, group_size=group_size, threshold_factor=0.5
    )
    assert codes.reshape(-1).tolist() == expected_codes
    torch.testing.assert_close(scales, torch.tensor(expected_scales), rtol=0, atol=0)
    restored = dequantize_groupwise_ternary(codes, scales, group_size=group_size, dtype=dtype)
    expected = torch.tensor([
        code * float(scales[i // group_size]) for i, code in enumerate(expected_codes)
    ], dtype=dtype).reshape(weights.shape)
    torch.testing.assert_close(restored, expected, rtol=0, atol=0)
    weights.requires_grad_()
    upstream = torch.arange(weights.numel(), dtype=dtype).reshape(weights.shape)
    quantized = fake_quantize_ternary(weights, group_size=group_size, threshold_factor=0.5)
    quantized.backward(upstream)
    torch.testing.assert_close(weights.grad, upstream, rtol=0, atol=0)


def test_dequantization_scale_gradient_excludes_tail_padding():
    codes = torch.tensor([1, -1, 0, 1, 1], dtype=torch.int8)
    scales = torch.tensor([2.0, 3.0], requires_grad=True)
    restored = dequantize_groupwise_ternary(codes, scales, group_size=3)
    restored.backward(torch.tensor([1., 2., 3., 4., 5.]))
    torch.testing.assert_close(scales.grad, torch.tensor([-1., 9.]), rtol=0, atol=0)
