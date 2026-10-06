import importlib.util

import pytest

torch = pytest.importorskip("torch")


def test_student_module_exists():
    assert importlib.util.find_spec("tiny_omni_decision.student.ternary") is not None


def api():
    from tiny_omni_decision.student import ternary
    return ternary


def test_quantization_values_partial_groups_and_zero():
    t = api()
    w = torch.tensor([[-2.0, 0.0, 0.1, 1.0, 0.0]])
    codes, scales = t.quantize(w, 3)
    assert codes.shape == w.shape
    assert set(codes.flatten().tolist()) <= {-1, 0, 1}
    assert scales.shape == (2,)
    assert scales[1].item() == pytest.approx(0.5)
    zc, zs = t.quantize(torch.zeros(2, 5), 3)
    assert torch.isfinite(zs).all()
    assert torch.equal(t.dequantize(zc, zs, 3), torch.zeros(2, 5))


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_weights_rejected(bad):
    with pytest.raises(ValueError, match="finite"):
        api().quantize(torch.tensor([[bad, 1.0]]), 2)


@pytest.mark.parametrize("group", [0, -1, True, 1.5])
def test_invalid_groups_rejected(group):
    with pytest.raises(ValueError):
        api().quantize(torch.ones(2, 3), group)


def test_ste_has_exact_ternary_forward_and_identity_gradient():
    t = api()
    w = torch.nn.Parameter(torch.tensor([[-1.2, 0.02, 0.7, 2.0]]))
    q = t.TernaryWeight(4)
    out = q(w)
    c, s = t.quantize(w, 4)
    assert torch.equal(out, t.dequantize(c, s, 4))
    out.sum().backward()
    assert torch.equal(w.grad, torch.ones_like(w))


def test_controller_targets_embedding_linear_and_conv_without_guessing():
    t = api()
    model = torch.nn.ModuleDict({
        "embedding": torch.nn.Embedding(9, 4),
        "linear": torch.nn.Linear(4, 3),
        "conv": torch.nn.Conv1d(2, 3, 2),
        "norm": torch.nn.LayerNorm(4),
    })
    controller = t.TernaryController(model, group_size=5)
    assert set(controller.targets) == {"embedding.weight", "linear.weight", "conv.weight"}
    assert not model["norm"].weight.requires_grad
    assert "linear.bias" in controller.exceptions
    out = model["linear"](model["embedding"](torch.tensor([1, 2])))
    out.square().sum().backward()
    assert model["linear"].parametrizations.weight.original.grad is not None


def test_freeze_and_lora_never_modify_quantized_base():
    t = api()
    torch.manual_seed(9)
    model = torch.nn.Sequential(torch.nn.Linear(5, 3))
    controller = t.TernaryController(model, group_size=4)
    x = torch.randn(4, 5)
    before = model(x).detach().clone()
    controller.freeze()
    c0, s0 = controller.components("0.weight")
    assert torch.equal(before, model(x))
    controller.add_lora(rank=2, alpha=4.0)
    assert torch.equal(before, model(x))
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.2)
    for _ in range(3):
        opt.zero_grad()
        model(x).square().mean().backward()
        opt.step()
    c1, s1 = controller.components("0.weight")
    assert torch.equal(c0, c1) and torch.equal(s0, s1)
    assert not torch.equal(before, model(x))
    assert all("lora_" in n for n, p in model.named_parameters() if p.requires_grad)
    with pytest.raises(ValueError, match="already"):
        controller.add_lora(rank=2, alpha=4.0)


def test_lora_requires_frozen_base():
    c = api().TernaryController(torch.nn.Sequential(torch.nn.Linear(2, 2)))
    with pytest.raises(ValueError, match="freeze"):
        c.add_lora(rank=1, alpha=1.0)


def test_shared_weights_rejected_before_any_model_mutation():
    t = api()
    model = torch.nn.ModuleDict({"a": torch.nn.Linear(2, 2), "b": torch.nn.Linear(2, 2)})
    model["b"].weight = model["a"].weight
    with pytest.raises(ValueError, match="shared"):
        t.TernaryController(model)
    assert not hasattr(model["a"], "parametrizations")


def test_exclusion_must_be_exact_existing_parameter():
    with pytest.raises(ValueError, match="exclusion"):
        api().TernaryController(torch.nn.Linear(2, 2), exclude=["typo.weight"])


@pytest.mark.parametrize("count", [1, 2, 4, 5, 6, 127, 128, 129])
def test_base3_roundtrip_and_size(count):
    from tiny_omni_decision.student.packing import pack_trits, unpack_trits
    codes = (torch.arange(count) % 3 - 1).to(torch.int8)
    packed = pack_trits(codes)
    assert packed.dtype == torch.uint8 and packed.numel() == (count + 4) // 5
    assert torch.equal(codes, unpack_trits(packed, count))


def test_invalid_packed_codes_rejected():
    from tiny_omni_decision.student.packing import pack_trits, unpack_trits
    with pytest.raises(ValueError):
        pack_trits(torch.tensor([2], dtype=torch.int8))
    with pytest.raises(ValueError):
        unpack_trits(torch.tensor([243], dtype=torch.uint8), 5)
    with pytest.raises(ValueError):
        unpack_trits(torch.tensor([0, 0], dtype=torch.uint8), 1)


def test_empty_target_rejected_before_mutating_other_weights():
    t = api()
    model = torch.nn.ModuleDict({"a": torch.nn.Linear(2, 2), "z": torch.nn.Linear(2, 0)})
    with pytest.raises(ValueError, match="empty"):
        t.TernaryController(model)
    assert not hasattr(model["a"], "parametrizations")
