import io

import pytest

torch = pytest.importorskip("torch")


def setup_readout():
    from tiny_omni_decision.native_candidate import NativeCandidateReadout, NativeTokenBlock

    torch.manual_seed(17)
    model = NativeCandidateReadout()
    block = NativeTokenBlock(
        "speech", torch.randn(5, 384), torch.randn(5, 3), torch.ones(5, dtype=torch.bool)
    )
    return model, block, torch.randn(4, 384)


def test_order_reuse_and_checkpoint():
    from tiny_omni_decision.native_candidate import NativeCandidateReadout, candidate_json

    model, block, queries = setup_readout()
    model.eval()
    scores = model([block], queries)
    torch.testing.assert_close(scores, model([block], queries), rtol=0, atol=0)
    torch.testing.assert_close(scores.flip(0), model([block], queries.flip(0)), atol=1e-6, rtol=0)
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    loaded = NativeCandidateReadout().eval()
    loaded.load_state_dict(torch.load(stream, weights_only=True), strict=True)
    torch.testing.assert_close(scores, loaded([block], queries), atol=0, rtol=0)
    original = candidate_json(scores, ["a", "b", "c", "d"])
    reordered = candidate_json(scores.flip(0), ["d", "c", "b", "a"])
    assert original["probabilities"] == reordered["probabilities"]
    assert original["selected_option_id"] == reordered["selected_option_id"]
    assert sum(p.numel() for p in model.parameters()) < 4_344_775


def test_mask_and_coordinate_effects():
    from tiny_omni_decision.native_candidate import NativeTokenBlock

    model, block, queries = setup_readout()
    base = model([block], queries)
    padded = NativeTokenBlock(
        "speech", torch.cat([block.values, torch.randn(2, 384) * 1e6]),
        torch.cat([block.positions, torch.randn(2, 3)]),
        torch.tensor([True] * 5 + [False] * 2),
    )
    torch.testing.assert_close(base, model([padded], queries), atol=0, rtol=0)
    permuted = NativeTokenBlock(
        "speech", block.values.flip(0), block.positions.flip(0), block.valid
    )
    torch.testing.assert_close(base, model([permuted], queries), atol=1e-6, rtol=0)
    reversed_features = NativeTokenBlock(
        "speech", block.values.flip(0), block.positions, block.valid
    )
    assert not torch.allclose(base, model([reversed_features], queries), atol=1e-7, rtol=0)


def test_all_kinds_and_gradient_reach():
    from tiny_omni_decision.native_candidate import WIDTHS, NativeTokenBlock

    model, _, queries = setup_readout()
    blocks = [NativeTokenBlock(k, torch.randn(3, d), torch.randn(3, 3),
                               torch.ones(3, dtype=torch.bool)) for k, d in WIDTHS.items()]
    loss = -model(blocks, queries).log_softmax(0)[0]
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(model.projections[k].weight.grad.abs().sum() > 0 for k in WIDTHS)


@pytest.mark.parametrize("case", ["empty", "all_masked", "width", "position", "mask_dtype",
                                  "nonfinite", "query_width", "one_option", "too_many",
                                  "unknown", "token_cap"])
def test_reject_invalid_inputs(case):
    from tiny_omni_decision.native_candidate import NativeTokenBlock

    model, block, queries = setup_readout()
    values, positions, valid, kind = block.values, block.positions, block.valid, block.kind
    if case == "all_masked":
        valid = torch.zeros(5, dtype=torch.bool)
    elif case == "width":
        values = values[:, :5]
    elif case == "position":
        positions = positions[:4]
    elif case == "mask_dtype":
        valid = valid.float()
    elif case == "nonfinite":
        values[0, 0] = float("nan")
    elif case == "query_width":
        queries = queries[:, :5]
    elif case == "one_option":
        queries = queries[:1]
    elif case == "too_many":
        queries = queries[:1].expand(41, -1)
    elif case == "unknown":
        kind = "sensor"
    elif case == "token_cap":
        values = values[:1].expand(8193, -1)
        positions = positions[:1].expand(8193, -1)
        valid = torch.ones(8193, dtype=torch.bool)
    with pytest.raises(ValueError):
        blocks = [] if case == "empty" else [NativeTokenBlock(kind, values, positions, valid)]
        model(blocks, queries)


def test_json_validation_and_tie_stability():
    from tiny_omni_decision.native_candidate import candidate_json

    for scores, ids in [(torch.zeros(2), ["a", "a"]), (torch.zeros(2), ["", "b"]),
                        (torch.zeros(3), ["a", "b"]), (torch.tensor([0., float("inf")]),
                                                       ["a", "b"])]:
        with pytest.raises(ValueError):
            candidate_json(scores, ids)
    assert candidate_json(torch.zeros(2), ["b", "a"])["selected_option_id"] == "a"
