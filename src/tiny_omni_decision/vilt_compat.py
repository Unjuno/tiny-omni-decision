"""Narrow compatibility translation of a deterministic legacy ViLT buffer."""

import torch


def remove_legacy_position_ids(state, max_positions):
    key = "vilt.embeddings.text_embeddings.position_ids"
    values = state.get(key)
    expected = torch.arange(max_positions, dtype=torch.int64).unsqueeze(0)
    if values is None or values.dtype != torch.int64 or not torch.equal(values.cpu(), expected):
        raise ValueError("legacy position IDs are not the exact deterministic buffer")
    return {name: tensor for name, tensor in state.items() if name != key}
