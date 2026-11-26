from __future__ import annotations

import pytest

from tiny_omni_decision.token_contracts import (
    embedding_vocab_padding,
    validate_token_id_bounds,
)


def test_embedding_vocab_allows_padded_model_rows() -> None:
    assert embedding_vocab_padding(32_000, 32_768) == 768


def test_embedding_vocab_rejects_tokenizer_larger_than_model() -> None:
    with pytest.raises(ValueError, match="exceeds the encoder embedding table"):
        embedding_vocab_padding(32_769, 32_768)


@pytest.mark.parametrize(
    ("minimum", "maximum"),
    [(-1, 5), (4, 3), (0, 32_768)],
)
def test_token_id_bounds_reject_invalid_embedding_indices(
    minimum: int, maximum: int
) -> None:
    with pytest.raises(ValueError, match="outside the encoder embedding table"):
        validate_token_id_bounds(minimum, maximum, 32_768)


def test_token_id_bounds_accept_ids_within_padded_embedding_table() -> None:
    validate_token_id_bounds(0, 31_999, 32_768)
