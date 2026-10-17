import copy
import io

import pytest

torch = pytest.importorskip("torch")


def fixture(*, exchange=True, gated=True, nonzero=False):
    from tiny_omni_decision.native_av_interaction import ConditionalAVReadout
    from tiny_omni_decision.native_candidate import NativeTokenBlock

    torch.manual_seed(17)
    model = ConditionalAVReadout(exchange=exchange, gated=gated)
    if nonzero:
        # A mechanics fixture, not training or a candidate checkpoint.
        with torch.no_grad():
            model.correction[-1].weight.normal_(0, 0.02)
    blocks = [
        NativeTokenBlock(k, torch.randn(n, d), torch.randn(n, 3),
                         torch.ones(n, dtype=torch.bool))
        for k, n, d in [("speech", 5, 384), ("event_audio", 2, 384), ("video", 11, 768)]
    ]
    return model, blocks, torch.randn(4, 384)


def test_initial_joint_and_missing_stream_parity():
    model, blocks, queries = fixture()
    torch.testing.assert_close(model(blocks, queries), model.baseline(blocks, queries),
                               rtol=0, atol=0)
    assert model.parameter_ledger()["reference_total_parameters"] == 197_097_949
    for remaining in [blocks[:2], blocks[2:]]:
        with torch.no_grad():
            model.correction[-1].weight.fill_(0.01)
        scores, info = model(remaining, queries, diagnostics=True)
        torch.testing.assert_close(scores, model.baseline(remaining, queries), rtol=0, atol=0)
        assert info["active"] is False and info["gates"] is None


@pytest.mark.parametrize("exchange", [False, True])
@pytest.mark.parametrize("gated", [False, True])
def test_ablations_conditional_gates_and_candidate_order(exchange, gated):
    model, blocks, queries = fixture(exchange=exchange, gated=gated, nonzero=True)
    scores, info = model(blocks, queries, diagnostics=True)
    assert info["active"] and info["gates"].shape == (4, 2)
    gates = info["gates"]
    if gated:
        assert ((gates > 0) & (gates < 1)).all()
        assert not torch.allclose(gates[0], gates[1], atol=1e-7, rtol=0)
        assert not torch.allclose(gates.sum(-1), torch.ones(4), atol=1e-7, rtol=0)
        altered = copy.deepcopy(blocks)
        altered[2].values.mul_(-3)
        _, other = model(altered, queries, diagnostics=True)
        assert not torch.allclose(gates, other["gates"], atol=1e-7, rtol=0)
        assert not torch.allclose(scores, model(altered, queries), atol=1e-7, rtol=0)
    else:
        torch.testing.assert_close(gates, torch.ones(4, 2), rtol=0, atol=0)
    reversed_scores, reverse = model(blocks, queries.flip(0), diagnostics=True)
    torch.testing.assert_close(scores.flip(0), reversed_scores, rtol=0, atol=1e-6)
    torch.testing.assert_close(gates.flip(0), reverse["gates"], rtol=0, atol=1e-6)
    ledger = model.parameter_ledger()
    assert ledger["allocated_readout_parameters"] == 1_732_420
    assert ledger["reference_total_parameters"] < 200_000_000
    assert ledger["active_joint_parameters"] == (
        1_732_420 - (0 if exchange else 297_216) - (0 if gated else 61_634)
    )


def test_masked_padding_ignored_and_bidirectional_gradients():
    from tiny_omni_decision.native_candidate import NativeTokenBlock

    model, blocks, queries = fixture(nonzero=True)
    old = blocks[-1]
    padded = NativeTokenBlock(
        "video", torch.cat([old.values, torch.full((2, 768), 1e6)]),
        torch.cat([old.positions, torch.full((2, 3), -1e6)]),
        torch.cat([old.valid, torch.zeros(2, dtype=torch.bool)]),
    )
    scores = model(blocks, queries)
    torch.testing.assert_close(scores, model([*blocks[:-1], padded], queries), rtol=0, atol=0)
    scores.log_softmax(0)[0].neg().backward()
    for name, p in model.named_parameters():
        if name.startswith(("pool", "audio_from_video", "video_from_audio",
                            "audio_norm", "video_norm", "candidate_attention", "gate",
                            "correction", "audio_latents", "video_latents")):
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
            # A scalar added to every option cancels in softmax cross entropy.
            if name != "correction.2.bias":
                assert p.grad.abs().sum() > 0, name


def test_zero_head_delays_upstream_gradient_and_checkpoint_roundtrip():
    from tiny_omni_decision.native_av_interaction import ConditionalAVReadout

    model, blocks, queries = fixture()
    model(blocks, queries).log_softmax(0)[0].neg().backward()
    assert model.correction[-1].weight.grad.abs().sum() > 0
    assert model.audio_from_video.in_proj_weight.grad.abs().sum() == 0
    model, blocks, queries = fixture(nonzero=True)
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    restored = ConditionalAVReadout()
    restored.load_state_dict(torch.load(stream, weights_only=True), strict=True)
    torch.testing.assert_close(model(blocks, queries), restored(blocks, queries), rtol=0, atol=0)


def test_checkpoint_policy_fails_before_mutation_and_baseline_ownership():
    from tiny_omni_decision.native_av_interaction import ConditionalAVReadout
    from tiny_omni_decision.native_candidate import NativeCandidateReadout

    base = NativeCandidateReadout()
    before = {k: v.clone() for k, v in base.state_dict().items()}
    model = ConditionalAVReadout(base, exchange=False)
    for key, value in base.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    foreign, _, _ = fixture()
    for strict in (False, True):
        with pytest.raises(ValueError, match="policy"):
            model.load_state_dict(foreign.state_dict(), strict=strict)
        for key, value in base.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    with pytest.raises(ValueError, match="policy"):
        model.load_state_dict({}, strict=False)


@pytest.mark.parametrize("case", ["empty", "nonfinite", "mask", "one_option"])
def test_existing_input_contract(case):
    model, blocks, queries = fixture()
    if case == "empty":
        blocks = []
    elif case == "nonfinite":
        blocks[0].values[0, 0] = float("nan")
    elif case == "mask":
        blocks[0].valid.fill_(False)
    else:
        queries = queries[:1]
    with pytest.raises(ValueError):
        model(blocks, queries)


@pytest.mark.parametrize("kwargs", [{"exchange": 1}, {"gated": "true"}, {"baseline": 3}])
def test_invalid_constructor(kwargs):
    from tiny_omni_decision.native_av_interaction import ConditionalAVReadout

    with pytest.raises(ValueError):
        ConditionalAVReadout(**kwargs)


def test_exchange_uses_original_pair_in_both_directions():
    model, blocks, queries = fixture(nonzero=True)
    seen = {}
    handles = []
    for name in ("audio_from_video", "video_from_audio"):
        def capture(module, inputs, key=name):
            seen[key] = tuple(x.detach().clone() for x in inputs)
        handles.append(getattr(model, name).register_forward_pre_hook(capture))
    model(blocks, queries)
    for handle in handles:
        handle.remove()
    a_query, v_key, _ = seen["audio_from_video"]
    v_query, a_key, _ = seen["video_from_audio"]
    torch.testing.assert_close(a_query, a_key, rtol=0, atol=0)
    torch.testing.assert_close(v_query, v_key, rtol=0, atol=0)


def test_gates_can_both_open_or_close_and_same_seed_arms_share_tensors():
    model, blocks, queries = fixture(nonzero=True)
    for bias, expected in [(10., True), (-10., False)]:
        with torch.no_grad():
            model.gate[2].weight.zero_()
            model.gate[2].bias.fill_(bias)
        _, info = model(blocks, queries, diagnostics=True)
        assert ((info["gates"] > .99) if expected else (info["gates"] < .01)).all()
    original, _, _ = fixture()
    for exchange, gated in [(False, False), (True, False), (False, True)]:
        other, _, _ = fixture(exchange=exchange, gated=gated)
        for key, value in original.state_dict().items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, other.state_dict()[key], rtol=0, atol=0)


def test_candidate_output_norm_is_opt_in_parameter_bounded_and_checkpoint_bound():
    from tiny_omni_decision.native_av_interaction import ConditionalAVReadout

    raw, blocks, queries = fixture(exchange=False, gated=False, nonzero=True)
    torch.manual_seed(17)
    normalized = ConditionalAVReadout(
        exchange=False, gated=False, candidate_output_norm=True
    )
    common = {k: v for k, v in raw.state_dict().items() if k != "_extra_state"}
    state = normalized.state_dict()
    state.update(common)
    normalized.load_state_dict(state, strict=True)
    assert normalized.get_extra_state() == {
        "schema": 2, "exchange": False, "gated": False,
        "candidate_output_norm": True,
    }
    ledger = normalized.parameter_ledger()
    assert ledger["allocated_readout_parameters"] == 1_732_804
    assert ledger["reference_total_parameters"] == 197_098_333

    scores, details = normalized(blocks, queries, diagnostics=True)
    assert details["candidate_output_norm"] is True
    assert torch.isfinite(scores).all()
    scores.log_softmax(0)[0].neg().backward()
    for parameter in normalized.candidate_norm.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0

    with pytest.raises(ValueError, match="policy"):
        raw.load_state_dict(normalized.state_dict(), strict=False)


def test_candidate_output_norm_false_preserves_legacy_state_schema_and_count():
    from tiny_omni_decision.native_av_interaction import ConditionalAVReadout

    torch.manual_seed(23)
    legacy = ConditionalAVReadout(exchange=False, gated=False)
    torch.manual_seed(23)
    explicit_legacy = ConditionalAVReadout(
        exchange=False, gated=False, candidate_output_norm=False
    )
    assert legacy.get_extra_state() == {
        "schema": 1, "exchange": False, "gated": False,
    }
    assert set(legacy.state_dict()) == set(explicit_legacy.state_dict())
    for key, value in legacy.state_dict().items():
        actual = explicit_legacy.state_dict()[key]
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, actual, rtol=0, atol=0)
        else:
            assert value == actual
    assert sum(p.numel() for p in explicit_legacy.parameters()) == 1_732_420
