"""Five ternary codes per byte (1.6 code bits/weight); scales are additional."""
from __future__ import annotations

import torch
from torch import Tensor


def pack_trits(codes: Tensor) -> Tensor:
    if codes.dtype != torch.int8 or codes.numel() == 0:
        raise ValueError("expected nonempty int8 codes")
    if ((codes < -1) | (codes > 1)).any():
        raise ValueError("invalid ternary codes")
    digits = codes.flatten().to(torch.int16) + 1
    digits = torch.nn.functional.pad(digits, (0, (-digits.numel()) % 5))
    powers = torch.tensor([1, 3, 9, 27, 81], dtype=torch.int16, device=digits.device)
    return (digits.reshape(-1, 5) * powers).sum(dim=1).to(torch.uint8)


def unpack_trits(packed: Tensor, count: int) -> Tensor:
    if type(count) is not int or count <= 0:
        raise ValueError("invalid code count")
    if packed.dtype != torch.uint8 or packed.ndim != 1:
        raise ValueError("expected a one-dimensional uint8 stream")
    if packed.numel() != (count + 4) // 5 or (packed >= 243).any():
        raise ValueError("invalid packed stream length or codes")
    powers = torch.tensor([1, 3, 9, 27, 81], dtype=torch.int16, device=packed.device)
    digits = (packed.to(torch.int16)[:, None] // powers) % 3
    if count % 5 and digits.flatten()[count:].any():
        raise ValueError("noncanonical padding")
    return (digits.flatten()[:count] - 1).to(torch.int8)
