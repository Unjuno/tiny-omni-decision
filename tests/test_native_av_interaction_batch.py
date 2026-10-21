import torch

from tiny_omni_decision.native_av_interaction import ConditionalAVReadout
from tiny_omni_decision.native_candidate import NativeTokenBlock


def _inputs(batch=2):
    generator = torch.Generator().manual_seed(91)
    blocks = {
        "text": (
            torch.randn(batch, 1, 384, generator=generator),
            torch.zeros(batch, 1, 3),
            torch.ones(batch, 1, dtype=torch.bool),
        ),
        "video": (
            torch.randn(batch, 4, 768, generator=generator),
            torch.randn(batch, 4, 3, generator=generator),
            torch.ones(batch, 4, dtype=torch.bool),
        ),
        "event_audio": (
            torch.randn(batch, 4, 384, generator=generator),
            torch.randn(batch, 4, 3, generator=generator),
            torch.ones(batch, 4, dtype=torch.bool),
        ),
    }
    queries = torch.randn(batch, 2, 384, generator=generator)
    return blocks, queries


def _single_inputs(blocks, row):
    return [
        NativeTokenBlock(kind, values[row], positions[row], valid[row])
        for kind, (values, positions, valid) in blocks.items()
    ]


def test_batched_conditional_av_matches_individual_forward():
    torch.manual_seed(17)
    model = ConditionalAVReadout().eval()
    blocks, queries = _inputs()

    expected = torch.stack(
        [model(_single_inputs(blocks, row), queries[row]) for row in range(len(queries))]
    )
    actual = model.forward_batched(blocks, queries)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_batched_conditional_av_preserves_candidate_permutation():
    torch.manual_seed(17)
    model = ConditionalAVReadout().eval()
    blocks, queries = _inputs()

    original = model.forward_batched(blocks, queries)
    permuted = model.forward_batched(blocks, queries.flip(1))

    torch.testing.assert_close(permuted, original.flip(1), rtol=1e-5, atol=1e-6)


def test_batched_conditional_av_updates_exchange_and_gate_parameters():
    torch.manual_seed(17)
    model = ConditionalAVReadout().train()
    with torch.no_grad():
        model.correction[-1].weight.normal_(mean=0.0, std=0.01)
    blocks, queries = _inputs()

    model.forward_batched(blocks, queries).square().mean().backward()

    for parameter in (
        model.audio_from_video.in_proj_weight,
        model.video_from_audio.in_proj_weight,
        model.gate[0].weight,
        model.pool.in_proj_weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
