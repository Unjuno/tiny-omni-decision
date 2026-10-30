"""Input-only condition interventions on the verified four-part candidate schema."""

import torch


def question_control(values, mode):
    if (
        values.ndim != 4
        or min(values.shape) == 0
        or values.shape[1] != 2
        or values.shape[-1] != 1536
        or not values.is_floating_point()
        or not torch.isfinite(values).all()
    ):
        raise ValueError("finite two-condition supplied-candidate features required")
    q, o = values[..., :384], values[..., 384:768]
    if (
        not torch.equal(q, q[:, :, :1].expand_as(q))
        or not torch.allclose(o, o[:, :1].expand_as(o), rtol=1e-5, atol=1e-6)
        or not torch.equal(values, torch.cat((q, o, (q - o).abs(), q * o), -1))
    ):
        raise ValueError("exact question/option factorization and paired options required")
    if mode == "normal":
        return values.clone()
    if mode == "neutral_question":
        q = q.mean(1, keepdim=True).expand_as(q)
    elif mode == "swapped_question":
        q = q.flip(1)
    else:
        raise ValueError("unknown question intervention")
    return torch.cat((q, o, (q - o).abs(), q * o), -1)
