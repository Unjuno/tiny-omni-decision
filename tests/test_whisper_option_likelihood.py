import pytest

torch = pytest.importorskip("torch")

from tiny_omni_decision.whisper_option_likelihood import (
    build_decoder_batch,
    candidate_probabilities,
    mean_transcript_log_likelihoods,
    ordered_sample_ids_sha256,
)


def test_build_decoder_batch_right_pads_and_masks_candidate_tokens():
    input_ids, mask = build_decoder_batch(
        [[5, 6], [7]],
        prefix_token_ids=[1, 2, 3, 4],
        eos_token_id=8,
        pad_token_id=0,
        device="cpu",
    )

    assert input_ids.tolist() == [[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 7, 0]]
    assert mask.tolist() == [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0]]


def test_mean_transcript_log_likelihood_scores_tokens_after_prefix_and_eos():
    logits = torch.full((2, 6, 12), -3.0)
    logits[0, 3, 5] = 3.0
    logits[0, 4, 6] = 3.0
    logits[0, 5, 8] = 3.0
    logits[1, 3, 7] = 3.0
    logits[1, 4, 8] = 3.0

    scores = mean_transcript_log_likelihoods(logits, [[5, 6], [7]], prefix_length=4, eos_token_id=8)

    assert scores.shape == (2,)
    assert torch.isfinite(scores).all()
    assert scores[0].item() > -0.1
    assert scores[0].item() == pytest.approx(scores[1].item())


def test_candidate_probabilities_are_order_equivariant_and_normalized():
    scores = torch.tensor([-0.4, -0.9, -1.2])
    original = candidate_probabilities(scores)
    permutation = torch.tensor([2, 0, 1])
    reordered = candidate_probabilities(scores[permutation])

    assert torch.allclose(reordered, original[permutation])
    assert float(original.sum()) == pytest.approx(1.0)


def test_ordered_sample_id_hash_matches_newline_join_without_trailing_newline():
    assert (
        ordered_sample_ids_sha256(["sample-a", "sample-b"])
        == "4932c53bd536ecf7caed0e90abfed9fa49cdb8bd0e319a5ef1d34fa26bbc510c"
    )


def test_ordered_sample_id_hash_rejects_empty_ids():
    with pytest.raises(ValueError, match="nonempty"):
        ordered_sample_ids_sha256(["sample-a", ""])


@pytest.mark.parametrize(
    "candidate_token_ids,prefix_token_ids",
    [([], [1, 2, 3, 4]), ([[5]], [1, 2, 3]), ([[5], []], [1, 2, 3, 4])],
)
def test_decoder_batch_rejects_malformed_candidates(candidate_token_ids, prefix_token_ids):
    with pytest.raises(ValueError):
        build_decoder_batch(
            candidate_token_ids,
            prefix_token_ids=prefix_token_ids,
            eos_token_id=8,
            pad_token_id=0,
            device="cpu",
        )


def test_log_likelihood_rejects_short_logits_and_nonfinite_values():
    short_logits = torch.zeros((1, 5, 10))
    with pytest.raises(ValueError, match="do not cover"):
        mean_transcript_log_likelihoods(short_logits, [[5, 6]], prefix_length=4, eos_token_id=8)

    nonfinite_logits = torch.zeros((1, 7, 10))
    nonfinite_logits[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        mean_transcript_log_likelihoods(nonfinite_logits, [[5, 6]], prefix_length=4, eos_token_id=8)
