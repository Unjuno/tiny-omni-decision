import io

import pytest

torch = pytest.importorskip("torch")


def core(width=12):
    from tiny_omni_decision.native_temporal import TemporalResidual

    torch.manual_seed(17)
    model = TemporalResidual(width, hidden=8)
    with torch.no_grad():
        model.output.weight.normal_(std=.02)
    return model


def video():
    from tiny_omni_decision.native_candidate import NativeTokenBlock

    positions = torch.cartesian_prod(torch.arange(4.), torch.arange(2.), torch.arange(2.))
    return NativeTokenBlock("video", torch.randn(16, 768), positions,
                            torch.ones(16, dtype=torch.bool))


def test_zero_initialization_and_parameter_budget():
    from tiny_omni_decision.native_candidate import NativeCandidateReadout
    from tiny_omni_decision.native_temporal import TemporalNativeCandidate

    torch.manual_seed(17)
    baseline = NativeCandidateReadout()
    model = TemporalNativeCandidate(baseline)
    block, queries = video(), torch.randn(4, 384)
    torch.testing.assert_close(model([block], queries), baseline([block], queries),
                               rtol=0, atol=0)
    assert 195365529 + sum(p.numel() for p in model.parameters()) <= 200000000


def test_causal_features_chunking_and_explicit_state():
    model = core()
    x, valid = torch.randn(3, 7, 12), torch.ones(3, 7, dtype=torch.bool)
    whole, final = model(x, valid)
    first, state = model(x[:, :3], valid[:, :3])
    last, last_state = model(x[:, 3:], valid[:, 3:], state)
    torch.testing.assert_close(whole, torch.cat([first, last], 1), rtol=1e-5, atol=1e-6)
    for a, b in zip(final, last_state, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
    changed = x.clone()
    changed[:, 3:] += 100
    torch.testing.assert_close(first, model(changed, valid)[0][:, :3], atol=1e-6, rtol=0)
    torch.testing.assert_close(whole, model(x, valid)[0], atol=0, rtol=0)
    assert not torch.allclose(whole, model(x, valid, final)[0])


def test_mask_holes_and_empty_stream_do_not_update():
    model = core()
    x = torch.randn(3, 5, 12)
    valid = torch.tensor([[1, 0, 1, 0, 1], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]]).bool()
    state = (torch.randn(1, 3, 8), torch.randn(1, 3, 8))
    out, final = model(x, valid, state)
    changed = x.clone()
    changed[~valid] = 1e5
    out2, final2 = model(changed, valid, state)
    torch.testing.assert_close(out[valid], out2[valid], atol=0, rtol=0)
    torch.testing.assert_close(out[~valid], x[~valid], atol=0, rtol=0)
    for a, b, initial in zip(final, final2, state, strict=True):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        torch.testing.assert_close(a[:, 1], initial[:, 1], atol=0, rtol=0)
    compact, compact_state = model(x[:1, valid[0]], torch.ones(1, 3, dtype=torch.bool),
                                   tuple(s[:, :1] for s in state))
    torch.testing.assert_close(out[0, valid[0]], compact[0], atol=1e-6, rtol=1e-5)
    for a, b in zip(final, compact_state, strict=True):
        torch.testing.assert_close(a[:, :1], b, atol=1e-6, rtol=1e-5)


def test_video_layout_uses_space_as_batch_not_time():
    from tiny_omni_decision.native_temporal import temporal_layout

    block = video()
    indices = temporal_layout(block)
    assert indices.tolist() == [[0, 4, 8, 12], [1, 5, 9, 13],
                                 [2, 6, 10, 14], [3, 7, 11, 15]]


@pytest.mark.parametrize("case", ["time_reverse", "space_change", "duplicate",
                                  "ragged", "nonfinite", "mask_dtype", "no_valid"])
def test_bad_layout_rejected(case):
    from tiny_omni_decision.native_candidate import NativeTokenBlock
    from tiny_omni_decision.native_temporal import temporal_layout

    b = video()
    x, p, m = b.values.clone(), b.positions.clone(), b.valid.clone()
    if case == "time_reverse":
        p = p.flip(0)
    elif case == "space_change":
        p[4, 1] += .3
    elif case == "duplicate":
        p[1] = p[0]
    elif case == "ragged":
        x, p, m = x[:-1], p[:-1], m[:-1]
    elif case == "nonfinite":
        x[0, 0] = float("nan")
    elif case == "mask_dtype":
        m = m.float()
    elif case == "no_valid":
        m[:] = False
    with pytest.raises(ValueError):
        temporal_layout(NativeTokenBlock("video", x, p, m))


def test_wrapper_gradients_reload_order_and_passthrough():
    from tiny_omni_decision.native_candidate import NativeTokenBlock
    from tiny_omni_decision.native_temporal import TemporalNativeCandidate

    torch.manual_seed(17)
    model = TemporalNativeCandidate()
    blocks = [video(), NativeTokenBlock("speech", torch.randn(9, 384),
              torch.stack([torch.arange(9.), torch.zeros(9), torch.zeros(9)], -1),
              torch.ones(9, dtype=torch.bool)),
              NativeTokenBlock("text", torch.randn(3, 384), torch.randn(3, 3),
                               torch.ones(3, dtype=torch.bool))]
    queries = torch.randn(4, 384)
    assert model.encode(blocks)[2] is blocks[2]
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    for _ in range(2):
        optimizer.zero_grad()
        (-model(blocks, queries).log_softmax(0)[0]).backward()
        for name, adapter in model.temporal.items():
            assert torch.isfinite(adapter.output.weight.grad).all(), name
            assert adapter.output.weight.grad.abs().sum() > 0, name
        optimizer.step()
    for name, adapter in model.temporal.items():
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   and p.grad.abs().sum() > 0 for p in adapter.parameters()), name
    scores = model(blocks, queries)
    torch.testing.assert_close(scores.flip(0), model(blocks, queries.flip(0)),
                               atol=1e-6, rtol=0)
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    reloaded = TemporalNativeCandidate()
    reloaded.load_state_dict(torch.load(stream, weights_only=True))
    torch.testing.assert_close(scores, reloaded(blocks, queries), atol=0, rtol=0)


@pytest.mark.parametrize("case", ["dtype", "shape", "nan", "mask", "state", "empty"])
def test_bad_core_inputs(case):
    model = core()
    x, valid, state = torch.randn(2, 3, 12), torch.ones(2, 3, dtype=torch.bool), None
    if case == "dtype":
        x = x.double()
    elif case == "shape":
        x = x[:, :, :2]
    elif case == "nan":
        x[0, 0, 0] = float("nan")
    elif case == "mask":
        valid = valid.float()
    elif case == "state":
        state = (torch.zeros(1, 3, 8), torch.zeros(1, 3, 8))
    elif case == "empty":
        x, valid = x[:, :0], valid[:, :0]
    with pytest.raises(ValueError):
        model(x, valid, state)


def test_spatial_streams_are_independent_and_tokens_are_retained():
    from tiny_omni_decision.native_candidate import NativeTokenBlock
    from tiny_omni_decision.native_temporal import TemporalNativeCandidate

    model = TemporalNativeCandidate()
    with torch.no_grad():
        model.temporal["video"].output.weight.normal_(std=.01)
    block = video()
    original = model.encode([block])[0]
    changed = block.values.clone()
    changed[::4] += 10
    modified = model.encode([NativeTokenBlock("video", changed, block.positions, block.valid)])[0]
    assert original.values.shape == block.values.shape
    torch.testing.assert_close(original.values[1::4], modified.values[1::4], atol=0, rtol=0)
    assert not torch.equal(original.values[::4], modified.values[::4])
    assert original.positions is block.positions and original.valid is block.valid


@pytest.mark.parametrize("bad_time", [True, False])
def test_invalid_speech_timeline_rejected(bad_time):
    from tiny_omni_decision.native_candidate import NativeTokenBlock
    from tiny_omni_decision.native_temporal import temporal_layout

    pos = torch.zeros(4, 3)
    pos[:, 0] = torch.arange(4.)
    if bad_time:
        pos[2, 0] = pos[1, 0]
    else:
        pos[1, 2] = 1
    with pytest.raises(ValueError):
        temporal_layout(NativeTokenBlock("speech", torch.randn(4, 384), pos,
                                         torch.ones(4, dtype=torch.bool)))


def test_fully_masked_core_preserves_explicit_state():
    model = core()
    x = torch.randn(2, 3, 12)
    state = (torch.randn(1, 2, 8), torch.randn(1, 2, 8))
    values, final = model(x, torch.zeros(2, 3, dtype=torch.bool), state)
    assert values is x and final is state
