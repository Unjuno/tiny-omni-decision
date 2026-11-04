"""Teacher-forced candidate scoring helpers for Whisper Decision inputs."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def build_decoder_batch(
    candidate_token_ids: Sequence[Sequence[int]],
    *,
    prefix_token_ids: Sequence[int],
    eos_token_id: int,
    pad_token_id: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build right-padded Whisper decoder inputs for text candidates.

    The decoder input is ``prefix + candidate``. For candidate token ``j``,
    its next-token score is read from decoder logit position
    ``len(prefix) - 1 + j``; EOS is scored at the following position.
    """
    prefix = [int(token) for token in prefix_token_ids]
    if len(prefix) != 4 or len(set(prefix)) != 4:
        raise ValueError("Whisper English transcription prefix must have four distinct IDs")
    if not candidate_token_ids:
        raise ValueError("at least one candidate transcript is required")
    candidates = [[int(token) for token in ids] for ids in candidate_token_ids]
    if any(not ids for ids in candidates):
        raise ValueError("candidate transcripts must contain at least one token")
    lengths = [len(prefix) + len(ids) for ids in candidates]
    width = max(lengths)
    input_ids = torch.full(
        (len(candidates), width), int(pad_token_id), dtype=torch.long, device=device
    )
    attention_mask = torch.zeros_like(input_ids)
    for row_index, (tokens, length) in enumerate(zip(candidates, lengths, strict=True)):
        values = prefix + tokens
        input_ids[row_index, :length] = torch.tensor(values, dtype=torch.long, device=device)
        attention_mask[row_index, :length] = 1
    return input_ids, attention_mask


def mean_transcript_log_likelihoods(
    logits: torch.Tensor,
    candidate_token_ids: Sequence[Sequence[int]],
    *,
    prefix_length: int,
    eos_token_id: int,
) -> torch.Tensor:
    """Score each candidate by mean log P(text tokens + EOS | audio, prefix)."""
    if logits.ndim != 3 or logits.shape[0] != len(candidate_token_ids):
        raise ValueError("logits must have shape [candidate_count, decoder_steps, vocabulary]")
    if prefix_length != 4 or any(not ids for ids in candidate_token_ids):
        raise ValueError("four-token prefix and nonempty candidate token sequences are required")
    if not torch.isfinite(logits).all():
        raise ValueError("decoder logits must be finite")
    log_probabilities = torch.log_softmax(logits.float(), dim=-1)
    scores = []
    first_candidate_logit = prefix_length - 1
    for row_index, candidate in enumerate(candidate_token_ids):
        targets = torch.tensor(
            [*map(int, candidate), int(eos_token_id)],
            dtype=torch.long,
            device=logits.device,
        )
        last_logit = first_candidate_logit + len(targets)
        if last_logit > logits.shape[1]:
            raise ValueError("decoder logits do not cover candidate tokens plus EOS")
        selected = log_probabilities[row_index, first_candidate_logit:last_logit].gather(
            1, targets[:, None]
        )
        scores.append(selected.mean())
    return torch.stack(scores)


def candidate_probabilities(scores: torch.Tensor) -> torch.Tensor:
    """Convert a fixed candidate score vector to normalized probabilities."""
    if scores.ndim != 1 or scores.numel() < 2 or not torch.isfinite(scores).all():
        raise ValueError("at least two finite candidate scores are required")
    return torch.softmax(scores.float(), dim=0)
