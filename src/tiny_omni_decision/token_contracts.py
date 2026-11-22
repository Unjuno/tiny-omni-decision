"""Pure validation helpers for tokenizer IDs and encoder embedding tables."""

from __future__ import annotations


def embedding_vocab_padding(tokenizer_vocab_size: int, embedding_rows: int) -> int:
    """Return unused model rows, allowing safe padded embedding vocabularies."""
    if tokenizer_vocab_size <= 0 or embedding_rows <= 0:
        raise ValueError("vocabulary sizes must be positive")
    if tokenizer_vocab_size > embedding_rows:
        raise ValueError("tokenizer vocabulary exceeds the encoder embedding table")
    return embedding_rows - tokenizer_vocab_size


def validate_token_id_bounds(min_token_id: int, max_token_id: int, embedding_rows: int) -> None:
    """Ensure every ID in a tokenized batch can index the embedding table."""
    if embedding_rows <= 0:
        raise ValueError("embedding table must contain at least one row")
    if min_token_id < 0 or max_token_id < min_token_id or max_token_id >= embedding_rows:
        raise ValueError("tokenizer emitted an ID outside the encoder embedding table")
