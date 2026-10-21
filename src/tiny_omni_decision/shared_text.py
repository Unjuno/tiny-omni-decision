"""Frozen candidate features from the existing Decision scorer's backbone."""

import torch


def shared_text_features(model, encoded):
    """Mean pool attended tokens without allocating another text model.

    This is a feature extraction primitive, not a guarantee that existing
    readouts trained on a different encoder remain compatible.
    """
    if any(p.requires_grad for p in model.parameters()):
        raise ValueError("shared text feature model must be frozen")
    if any(m.training for m in model.modules()):
        raise ValueError("shared text model must be in evaluation mode")
    ids, mask = encoded["input_ids"], encoded["attention_mask"]
    if (
        ids.ndim != 2
        or mask.shape != ids.shape
        or not ids.shape[0]
        or not torch.all((mask == 0) | (mask == 1))
        or not torch.all(mask.sum(-1) > 0)
    ):
        raise ValueError("nonempty aligned binary attention mask required")
    with torch.inference_mode():
        hidden = model.base_model(**encoded, return_dict=True).last_hidden_state
        if hidden.ndim != 3 or hidden.shape[:2] != ids.shape or not torch.isfinite(hidden).all():
            raise ValueError("invalid backbone hidden states")
        weights = mask.unsqueeze(-1).to(hidden.dtype)
        features = (hidden.float() * weights).sum(1) / weights.sum(1)
    if not torch.isfinite(features).all():
        raise ValueError("nonfinite shared text features")
    return features
