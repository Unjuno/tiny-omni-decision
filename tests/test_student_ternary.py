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


def test_bfloat16_ste_forward_is_exactly_the_dequantized_ternary_value(ternary) -> None:
    torch = ternary.torch
    weights = torch.tensor([0.30078125, 3.0], dtype=torch.bfloat16, requires_grad=True)
    codes, scales = ternary.quantize(weights, group_size=2, threshold_factor=0.01)
    expected = ternary.dequantize(codes, scales, group_size=2, dtype=torch.bfloat16)

    actual = ternary.fake_quantize(weights, group_size=2, threshold_factor=0.01)
    actual.sum().backward()

    torch.testing.assert_close(actual.detach(), expected, rtol=0, atol=0)
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


def test_packed_overlay_round_trips_quantized_weights_and_checks_base_revision(
    tmp_path,
) -> None:
    torch = pytest.importorskip("torch")
    from tiny_omni_decision.ternary import (
        export_packed_ternary_overlay,
        load_packed_ternary_overlay,
    )

    class LinearModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(4, 1, bias=False)

        def forward(self, inputs):
            return self.linear(inputs)

    model = LinearModel()
    with torch.no_grad():
        model.linear.weight.copy_(torch.tensor([[-3.0, -1.0, 1.0, 3.0]]))
    expected = torch.tensor([[-3.0, 0.0, 0.0, 3.0]])
    expected_output = torch.tensor([[9.0]])

    overlay = tmp_path / "student-ternary"
    metadata = export_packed_ternary_overlay(
        model,
        overlay,
        base_model_id="google/embeddinggemma-2",
        base_revision="immutable-revision",
        group_size=4,
        threshold_factor=0.7,
    )
    restored = LinearModel()
    with torch.no_grad():
        restored.linear.weight.fill_(9.0)
    loaded = load_packed_ternary_overlay(
        restored,
        overlay,
        expected_base_model_id="google/embeddinggemma-2",
        expected_base_revision="immutable-revision",
    )

    torch.testing.assert_close(restored.linear.weight, expected)
    torch.testing.assert_close(
        restored(torch.tensor([[1.0, 2.0, 3.0, 4.0]])).detach(), expected_output
    )
    assert metadata["target_parameter_count"] == 1
    assert metadata["packed_code_bytes"] == 1
    assert loaded["base_revision"] == "immutable-revision"
    with pytest.raises(ValueError, match="base revision mismatch"):
        load_packed_ternary_overlay(
            restored,
            overlay,
            expected_base_model_id="google/embeddinggemma-2",
            expected_base_revision="different-revision",
        )


def test_registering_qat_weights_does_not_quantize_full_weights(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    import tiny_omni_decision.ternary as ternary

    model = torch.nn.Sequential(torch.nn.Linear(4, 2, bias=False))

    def forbidden(*args, **kwargs):
        raise AssertionError("quantization during parametrization registration")

    monkeypatch.setattr(ternary, "quantize_groupwise_ternary", forbidden)
    targets = ternary.apply_ternary_qat(model, group_size=4)
    assert targets == ("0.weight",)


def test_validation_uses_one_quantization_per_target_and_restores_shadows(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    import tiny_omni_decision.ternary as ternary

    model = torch.nn.Sequential(torch.nn.Linear(4, 2, bias=False)).eval()
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[-3., -1., 1., 3.], [2., 3., 4., 5.]]))
    targets = ternary.apply_ternary_qat(model, group_size=4)
    shadow = model[0].parametrizations.weight.original
    original = shadow.detach().clone()
    x = torch.tensor([[1., 2., 3., 4.]])
    with torch.no_grad():
        expected = model(x)
    quantize = ternary.quantize_groupwise_ternary
    calls = []

    def counted(weights, **kwargs):
        calls.append(tuple(weights.shape))
        return quantize(weights, **kwargs)

    monkeypatch.setattr(ternary, "quantize_groupwise_ternary", counted)
    with ternary.cached_ternary_validation(model, targets):
        with torch.inference_mode():
            assert torch.equal(model(x), expected)
            assert torch.equal(model(x), expected)
        assert calls == [tuple(original.shape)]
        assert shadow.device == original.device
    assert torch.equal(shadow.detach(), original)
    with torch.no_grad():
        assert torch.equal(model(x), expected)
    assert len(calls) == 2
    model(x).sum().backward()
    assert shadow.grad is not None and torch.isfinite(shadow.grad).all()


def test_validation_restores_original_shadows_after_exception() -> None:
    torch = pytest.importorskip("torch")
    from tiny_omni_decision.ternary import apply_ternary_qat, cached_ternary_validation

    model = torch.nn.Sequential(torch.nn.Linear(4, 2, bias=False)).eval()
    targets = apply_ternary_qat(model, group_size=4)
    original = model[0].parametrizations.weight.original.detach().clone()
    with pytest.raises(RuntimeError, match="interrupted"):
        with cached_ternary_validation(model, targets):
            raise RuntimeError("interrupted")
    assert torch.equal(model[0].parametrizations.weight.original.detach(), original)


def test_cpu_master_update_reuses_ternary_weights_with_identical_ste_gradients(
    monkeypatch,
) -> None:
    torch = pytest.importorskip("torch")
    import tiny_omni_decision.ternary as ternary

    torch.manual_seed(17)
    baseline = torch.nn.Sequential(torch.nn.Linear(4, 2, bias=False))
    cached = torch.nn.Sequential(torch.nn.Linear(4, 2, bias=False))
    cached.load_state_dict(baseline.state_dict())
    names = ternary.apply_ternary_qat(baseline, group_size=4)
    ternary.apply_ternary_qat(cached, group_size=4)
    base_shadow = baseline[0].parametrizations.weight.original
    cached_shadow = cached[0].parametrizations.weight.original
    cpu_master = [torch.nn.Parameter(cached_shadow.detach().cpu().float().clone())]
    original = cached_shadow.detach().clone()
    examples = [torch.randn(1, 4), torch.randn(1, 4)]

    baseline_outputs = []
    for inputs in examples:
        output = baseline(inputs)
        baseline_outputs.append(output.detach().clone())
        output.sum().backward()

    quantize = ternary.quantize_groupwise_ternary
    quantizations = []

    def counted(weights, **kwargs):
        quantizations.append(tuple(weights.shape))
        return quantize(weights, **kwargs)

    monkeypatch.setattr(ternary, "quantize_groupwise_ternary", counted)
    with ternary.cached_ternary_training_update(cached, names, cpu_master):
        for inputs, expected in zip(examples, baseline_outputs, strict=True):
            output = cached(inputs)
            torch.testing.assert_close(output.detach(), expected, rtol=0, atol=0)
            output.sum().backward()
        assert quantizations == [tuple(original.shape)]

    torch.testing.assert_close(cached_shadow.grad, base_shadow.grad, rtol=0, atol=0)
    assert torch.equal(cached_shadow.detach(), original)
    assert len(quantizations) == 1
    with torch.no_grad():
        cached(examples[0])
    assert len(quantizations) == 2


def test_cpu_master_update_restores_shadows_when_microbatch_fails() -> None:
    torch = pytest.importorskip("torch")
    from tiny_omni_decision.ternary import (
        apply_ternary_qat,
        cached_ternary_training_update,
    )

    model = torch.nn.Sequential(torch.nn.Linear(4, 2, bias=False))
    names = apply_ternary_qat(model, group_size=4)
    original = model[0].parametrizations.weight.original.detach().clone()
    master = [torch.nn.Parameter(original.detach().cpu().float().clone())]
    with pytest.raises(RuntimeError, match="microbatch failed"):
        with cached_ternary_training_update(model, names, master):
            model(torch.ones(1, 4)).sum().backward()
            raise RuntimeError("microbatch failed")
    assert torch.equal(model[0].parametrizations.weight.original.detach(), original)


def test_cached_qat_update_preserves_bf16_forward_and_ste() -> None:
    torch = pytest.importorskip("torch")
    from tiny_omni_decision.ternary import (
        apply_ternary_qat,
        cached_ternary_training_update,
    )

    torch.manual_seed(7)
    reference = torch.nn.Sequential(torch.nn.Linear(4, 2, bias=False)).to(torch.bfloat16)
    candidate = torch.nn.Sequential(torch.nn.Linear(4, 2, bias=False)).to(torch.bfloat16)
    candidate.load_state_dict(reference.state_dict())
    names = apply_ternary_qat(reference, group_size=4)
    apply_ternary_qat(candidate, group_size=4)
    ref_param = reference[0].parametrizations.weight.original
    can_param = candidate[0].parametrizations.weight.original
    saved = can_param.detach().clone()
    cpu_master = [torch.nn.Parameter(saved.detach().cpu().float().clone())]
    inputs = [
        torch.tensor([[1.0, -2.0, 0.5, 3.0]], dtype=torch.bfloat16),
        torch.tensor([[-2.0, 0.5, 1.0, -1.0]], dtype=torch.bfloat16),
    ]
    originals = []
    for value in inputs:
        logits = reference(value)
        originals.append(logits.detach().clone())
        logits.sum().backward()
    with cached_ternary_training_update(candidate, names, cpu_master):
        for value, expected in zip(inputs, originals, strict=True):
            logits = candidate(value)
            torch.testing.assert_close(logits, expected, atol=0, rtol=0)
            logits.sum().backward()
    torch.testing.assert_close(can_param.grad, ref_param.grad, atol=0, rtol=0)
    assert torch.equal(can_param.detach(), saved)
