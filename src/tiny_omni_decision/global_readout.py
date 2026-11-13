"""Algebraic export for the explicitly repeated-cell global comparison arm."""

import torch

from tiny_omni_decision.bilinear_readout import BilinearCandidateScorer


def fold_global_readout(model, *, condition):
    """Input becomes [spatial summary, cell mean, candidate], without cell /4.

    This is valid only when all sixteen original cells were the same mean.
    The summary is retained separately: floating point means need not be equal.
    Original weights and the caller's RNG state remain unchanged.
    """
    if condition != "global" or model.observation_size != 13056 or model.feature_size != 13440:
        raise ValueError("requires the fixed 17x768 repeated-global-cell representation")
    weight = model.observation.weight
    with torch.random.fork_rng(devices=[]):
        result = BilinearCandidateScorer(1536, 384, weight.shape[0]).to(
            device=weight.device, dtype=weight.dtype)
    with torch.no_grad():
        result.observation.weight[:, :768].copy_(weight[:, :768])
        result.observation.weight[:, 768:].copy_(
            weight[:, 768:].reshape(weight.shape[0], 16, 768).sum(1) / 4)
        result.observation.bias.copy_(model.observation.bias)
        result.candidate.load_state_dict(model.candidate.state_dict())
    return result.eval()
