"""Length-normalized teacher-forced transcript scoring, excluding prompt/padding."""

import numpy as np


def mean_transcript_logprob(token_logprobs, sequence_lengths, *, prefix_length):
    values = np.asarray(token_logprobs, dtype=np.float64)
    lengths = np.asarray(sequence_lengths)
    if (
        values.ndim != 2
        or lengths.shape != (values.shape[0],)
        or not np.issubdtype(lengths.dtype, np.integer)
        or prefix_length < 1
        or np.any(lengths <= prefix_length)
        or np.any(lengths > values.shape[1] + 1)
        or not np.isfinite(values).all()
    ):
        raise ValueError("invalid shifted token probabilities or sequence lengths")
    positions = np.arange(values.shape[1])[None, :]
    mask = (positions >= prefix_length - 1) & (positions < lengths[:, None] - 1)
    return np.where(mask, values, 0).sum(1) / mask.sum(1)
