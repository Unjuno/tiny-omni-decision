from __future__ import annotations

import itertools
from types import SimpleNamespace

import pytest


@pytest.fixture
def ternary():
    torch = pytest.importorskip("torch")
    from tiny_omni_decision.ternary import (
        apply_ternary_qat,
        dequantize_groupwise_ternary,
        fake_quantize_ternary,
        pack_ternary_codes,
        quantize_groupwise_ternary,
        select_ternary_parameter_names,
        unpack_ternary_codes,
    )

    return SimpleNamespace(
        torch=torch,
        apply_qat=apply_ternary_qat,
        dequantize=dequantize_groupwise_ternary,
        fake_quantize=fake_quantize_ternary,
        pack=pack_ternary_codes,
        quantize=quantize_groupwise_ternary,
        select_targets=select_ternary_parameter_names,
        unpack=unpack_ternary_codes,
    )


def test_groupwise_quantization_uses_active_mean_scale_and_handles_tail(ternary) -> None:
    torch = ternary.torch
    dequantize = ternary.dequantize
    quantize = ternary.quantize
    weights = torch.tensor([-3.0, -1.0, 1.0, 3.0, 9.0])

    codes, scales = quantize(weights, group_size=4, threshold_factor=0.7)
    restored = dequantize(codes, scales, group_size=4)

    assert codes.tolist() == [-1, 0, 0, 1, 1]
    torch.testing.assert_close(scales, torch.tensor([3.0, 9.0]))
    torch.testing.assert_close(restored, torch.tensor([-3.0, 0.0, 0.0, 3.0, 9.0]))


def test_groupwise_quantization_maps_zero_group_to_zero_scale(ternary) -> None:
    torch = ternary.torch
    quantize = ternary.quantize

    codes, scales = quantize(torch.zeros(6), group_size=4)

    assert codes.tolist() == [0, 0, 0, 0, 0, 0]
    torch.testing.assert_close(scales, torch.tensor([0.0, 0.0]))


def test_ste_preserves_the_forward_codes_and_identity_gradient(ternary) -> None:
    torch = ternary.torch
    fake_quantize = ternary.fake_quantize
    weights = torch.tensor([-3.0, -1.0, 1.0, 3.0], requires_grad=True)

    quantized = fake_quantize(weights, group_size=4, threshold_factor=0.7)
    quantized.sum().backward()

    torch.testing.assert_close(quantized.detach(), torch.tensor([-3.0, 0.0, 0.0, 3.0]))
    torch.testing.assert_close(weights.grad, torch.ones_like(weights))


def test_qat_parametrization_uses_ternary_weights_and_backpropagates(ternary) -> None:
    torch = ternary.torch
    apply_qat = ternary.apply_qat

    class LinearModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(4, 2, bias=False)

        def forward(self, inputs):
            return self.linear(inputs)

    model = LinearModel()
    with torch.no_grad():
        model.linear.weight.copy_(torch.tensor([[-3.0, -1.0, 1.0, 3.0], [1.0, 2.0, 3.0, 4.0]]))

    targets = apply_qat(model, group_size=4, threshold_factor=0.7)
    output = model(torch.tensor([[1.0, 2.0, 3.0, 4.0]]))
    output.sum().backward()

    assert targets == ("linear.weight",)
    torch.testing.assert_close(
        model.linear.weight,
        torch.tensor([[-3.0, 0.0, 0.0, 3.0], [0.0, 3.0, 3.0, 3.0]]),
    )
    torch.testing.assert_close(output.detach(), torch.tensor([[9.0, 27.0]]))
    grad = model.linear.parametrizations.weight.original.grad
    assert grad is not None and torch.isfinite(grad).all()
    torch.testing.assert_close(grad, torch.tensor([[1.0, 2.0, 3.0, 4.0]] * 2))


def test_five_trit_byte_round_trips_all_code_patterns(ternary) -> None:
    torch = ternary.torch
    pack = ternary.pack
    unpack = ternary.unpack
    patterns = list(itertools.product((-1, 0, 1), repeat=5))
    codes = torch.tensor(patterns, dtype=torch.int8)

    packed = pack(codes)
    restored = unpack(packed, num_elements=codes.numel())

    assert packed.dtype == torch.uint8
    assert packed.numel() == len(patterns)
    torch.testing.assert_close(restored, codes.reshape(-1))


def test_five_trit_byte_encoding_is_little_endian_base_three(ternary) -> None:
    torch = ternary.torch
    pack = ternary.pack
    unpack = ternary.unpack
    codes = torch.tensor([-1, 0, 1, -1, 1, 0], dtype=torch.int8)

    packed = pack(codes)

    assert packed.tolist() == [183, 121]
    torch.testing.assert_close(unpack(packed, num_elements=6), codes)


def test_packed_decoder_rejects_invalid_bytes_and_nonzero_padding(ternary) -> None:
    torch = ternary.torch
    unpack = ternary.unpack

    with pytest.raises(ValueError, match="invalid packed trit byte"):
        unpack(torch.tensor([243], dtype=torch.uint8), num_elements=5)
    with pytest.raises(ValueError, match="padding trits must encode zero"):
        unpack(torch.tensor([162], dtype=torch.uint8), num_elements=4)


def test_pack_and_dequantize_reject_fractional_codes(ternary) -> None:
    torch = ternary.torch
    dequantize = ternary.dequantize
    pack = ternary.pack
    codes = torch.tensor([0.5])

    with pytest.raises(ValueError, match="codes must contain only"):
        pack(codes)
    with pytest.raises(ValueError, match="codes must contain only"):
        dequantize(codes, torch.tensor([1.0]), group_size=1)


def test_target_selector_covers_loaded_weight_families_and_fails_closed(ternary) -> None:
    torch = ternary.torch
    select_targets = ternary.select_targets

    class EmbeddingGemma2TextScaledWordEmbedding(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(4, 2))

    class Gemma4AudioCausalConv1d(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2, 1, 3))

    class Gemma4VisionPatchEmbedder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.position_embedding_table = torch.nn.Parameter(torch.ones(2, 4, 3))

    class Vision(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.patch_embedder = Gemma4VisionPatchEmbedder()

    class ToyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.text = torch.nn.Linear(2, 2, bias=False)
            self.embedding = torch.nn.Embedding(4, 2)
            self.scaled_embedding = EmbeddingGemma2TextScaledWordEmbedding()
            self.audio_conv = Gemma4AudioCausalConv1d()
            self.vision_tower = Vision()
            self.norm = torch.nn.LayerNorm(2)

    model = ToyModel()
    targets = select_targets(model)

    assert targets == (
        "audio_conv.weight",
        "embedding.weight",
        "scaled_embedding.weight",
        "text.weight",
        "vision_tower.patch_embedder.position_embedding_table",
    )

    class UnknownMatrix(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.kernel = torch.nn.Parameter(torch.ones(2, 2))

    model.unsupported = UnknownMatrix()
    with pytest.raises(ValueError, match="unclassified matrix parameter.*unsupported.kernel"):
        select_targets(model)
