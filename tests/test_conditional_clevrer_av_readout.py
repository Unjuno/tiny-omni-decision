import torch

from scripts.run_conditional_clevrer_av_readout import (
    _baseline_batched,
    _batch_blocks,
    _make_positions,
)
from tiny_omni_decision.native_candidate import NativeCandidateReadout, NativeTokenBlock


def test_synthetic_event_audio_adapter_preserves_scalar_and_temporal_alignment():
    audio = torch.tensor([[[0.1], [0.2], [0.3]]])
    adapted = _batch_blocks(audio, torch.zeros(1, 384), torch.zeros(1, 3, 768))

    values, positions, valid = adapted["event_audio"]
    assert values.shape == (1, 3, 384)
    torch.testing.assert_close(values[..., 0], audio[..., 0])
    assert torch.count_nonzero(values[..., 1:]) == 0
    assert valid.all()
    torch.testing.assert_close(positions, _make_positions(1, 3))


def test_question_is_a_single_text_token_separate_from_candidate_queries():
    blocks = _batch_blocks(
        torch.ones(2, 3, 1),
        torch.arange(2 * 384, dtype=torch.float32).reshape(2, 384),
        torch.zeros(2, 3, 768),
    )

    text_values, text_positions, text_valid = blocks["text"]
    assert text_values.shape == (2, 1, 384)
    assert text_positions.shape == (2, 1, 3)
    assert text_valid.shape == (2, 1)
    torch.testing.assert_close(
        text_values[:, 0], torch.arange(2 * 384, dtype=torch.float32).reshape(2, 384)
    )


def test_batched_c0_matches_existing_native_candidate_forward():
    torch.manual_seed(17)
    model = NativeCandidateReadout().eval()
    audio = torch.randn(2, 3, 1)
    question = torch.randn(2, 384)
    video = torch.randn(2, 3, 768)
    blocks = _batch_blocks(audio, question, video)
    queries = torch.randn(2, 2, 384)

    batched = _baseline_batched(model, blocks, queries)
    single = torch.stack(
        [
            model(
                [
                    NativeTokenBlock(kind, values[index], positions[index], valid[index])
                    for kind, (values, positions, valid) in blocks.items()
                ],
                queries[index],
            )
            for index in range(len(queries))
        ]
    )

    torch.testing.assert_close(batched, single, rtol=1e-5, atol=1e-6)
