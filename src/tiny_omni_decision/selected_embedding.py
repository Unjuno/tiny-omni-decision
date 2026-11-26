"""Small zero-initialized updates to explicitly selected embedding rows."""

import torch
from torch import nn


class SelectedEmbeddingDelta(nn.Module):
    def __init__(self, base, token_ids):
        super().__init__()
        ids = list(token_ids)
        if (
            not ids
            or len(set(ids)) != len(ids)
            or any(i < 0 or i >= base.num_embeddings for i in ids)
        ):
            raise ValueError("unique in-range token IDs required")
        self.base = base
        self.base.requires_grad_(False)
        self.delta = nn.Parameter(
            torch.zeros(
                len(ids), base.embedding_dim, device=base.weight.device, dtype=base.weight.dtype
            )
        )
        mapping = torch.full(
            (base.num_embeddings,), -1, dtype=torch.long, device=base.weight.device
        )
        mapping[torch.tensor(ids, device=mapping.device)] = torch.arange(
            len(ids), device=mapping.device
        )
        self.register_buffer("row_mapping", mapping)

    def forward(self, input_ids):
        rows = self.row_mapping[input_ids]
        update = torch.nn.functional.embedding(rows.clamp_min(0), self.delta)
        return self.base(input_ids) + update * (rows >= 0).unsqueeze(-1)
