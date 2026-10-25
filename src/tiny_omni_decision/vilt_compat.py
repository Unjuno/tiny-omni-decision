"""Narrow compatibility translation of a deterministic legacy ViLT buffer."""

import torch


def remove_legacy_position_ids(state, max_positions):
    key = "vilt.embeddings.text_embeddings.position_ids"
    values = state.get(key)
    expected = torch.arange(max_positions, dtype=torch.int64).unsqueeze(0)
    if values is None or values.dtype != torch.int64 or not torch.equal(values.cpu(), expected):
        raise ValueError("legacy position IDs are not the exact deterministic buffer")
    return {name: tensor for name, tensor in state.items() if name != key}


def build_vilt_question_input_ids(tokenizer, question):
    """Tokenize without truncation and add ViLT's BERT CLS/SEP boundaries."""
    if tokenizer.cls_token_id is None or tokenizer.sep_token_id is None:
        raise ValueError("ViLT question tokenizer must define CLS and SEP token IDs")
    original_limit = tokenizer.model_max_length
    tokenizer.model_max_length = max(original_limit, 1_000_000)
    try:
        tokens = tokenizer.tokenize(question)
    finally:
        tokenizer.model_max_length = original_limit
    content_ids = tokenizer.convert_tokens_to_ids(tokens)
    if isinstance(content_ids, int):
        content_ids = [content_ids]
    if any(token_id is None for token_id in content_ids):
        raise ValueError("question tokenizer returned a token without an ID")
    return [tokenizer.cls_token_id, *content_ids, tokenizer.sep_token_id]


def split_vilt_token_ids(
    input_ids,
    *,
    max_positions,
    overlap_tokens,
    cls_token_id,
    sep_token_id,
):
    """Split one fully tokenized question into special-token-safe overlapping windows."""
    if not isinstance(max_positions, int) or max_positions < 3:
        raise ValueError("max_positions must allow CLS, content, and SEP")
    content_capacity = max_positions - 2
    if not isinstance(overlap_tokens, int) or not 0 <= overlap_tokens < content_capacity:
        raise ValueError("overlap_tokens must be non-negative and smaller than content capacity")
    if len(input_ids) < 2 or input_ids[0] != cls_token_id or input_ids[-1] != sep_token_id:
        raise ValueError("input_ids must contain the expected CLS and SEP boundary tokens")

    if len(input_ids) <= max_positions:
        return [list(input_ids)]

    content = list(input_ids[1:-1])
    stride = content_capacity - overlap_tokens
    windows = []
    for start in range(0, len(content), stride):
        chunk = content[start : start + content_capacity]
        windows.append([cls_token_id, *chunk, sep_token_id])
        if start + content_capacity >= len(content):
            break
    return windows
