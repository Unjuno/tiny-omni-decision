"""Frozen DeBERTa inputs with full token coverage, not full-context equivalence.

Single observations use 510 content tokens plus CLS/SEP per encoder window.
Later windows overlap by32; overlapping output tokens are retained only once.
Question/candidate pairs remain single-context inputs with the original layout.
"""

import torch


def content_spans(count):
    if type(count) is not int or not 0 <= count <= 8190:
        raise ValueError("0..8190 content tokens required")
    spans = [(0, min(count, 510), 0)]
    while spans[-1][1] < count:
        start = spans[-1][1] - 32
        spans.append((start, min(count, start + 510), 32))
    return spans


def _encode_input(model, tokenizer, text, pair=None):
    if (any(m.training for m in model.modules())
            or any(p.requires_grad for p in model.parameters())):
        raise ValueError("frozen evaluation encoder required")
    if not isinstance(text, str) or (pair is not None and not isinstance(pair, str)):
        raise ValueError("original text strings required")
    device = next(model.parameters()).device
    encoded = tokenizer(text, text_pair=pair, return_tensors="pt", padding=False,
                        add_special_tokens=True, truncation=False)
    ids = encoded["input_ids"]
    if (set(encoded) != {"input_ids", "attention_mask", "token_type_ids"}
            or ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < 2
            or any(v.shape != ids.shape or v.dtype != torch.long for v in encoded.values())
            or not (encoded["attention_mask"] == 1).all()
            or ids[0, 0].item() != tokenizer.cls_token_id
            or ids[0, -1].item() != tokenizer.sep_token_id):
        raise ValueError("unrecognized unpadded CLS/SEP tokenizer layout")
    if pair is None and not (encoded["token_type_ids"] == 0).all():
        raise ValueError("single-sequence token types must be zero")
    return {k: v.to(device) for k, v in encoded.items()}


def _forward(model, encoded):
    hidden = model.deberta(**encoded).last_hidden_state
    if (hidden.shape != (*encoded["input_ids"].shape, 384)
            or not torch.isfinite(hidden).all()):
        raise ValueError("finite original384-wide token states required")
    return hidden[0]


@torch.inference_mode()
def observation_tokens(model, tokenizer, text):
    encoded = _encode_input(model, tokenizer, text)
    ids = encoded["input_ids"]
    spans = content_spans(ids.shape[1] - 2)
    if len(spans) == 1:
        return _forward(model, encoded), spans
    output = []
    for index, (start, stop, keep) in enumerate(spans):
        window = torch.cat([ids[:, :1], ids[:, start + 1:stop + 1], ids[:, -1:]], 1)
        features = _forward(model, dict(input_ids=window, attention_mask=torch.ones_like(window),
                                        token_type_ids=torch.zeros_like(window)))
        if index == 0:
            output.append(features[:1])
        output.append(features[1 + keep:-1])
        if index == len(spans) - 1:
            output.append(features[-1:])
    result = torch.cat(output)
    if len(result) != ids.shape[1]:
        raise ValueError("window reconstruction lost or duplicated tokens")
    return result, spans


@torch.inference_mode()
def candidate_vector(model, tokenizer, question, candidate):
    encoded = _encode_input(model, tokenizer, question, candidate)
    if encoded["input_ids"].shape[1] > 512:
        raise ValueError("question/candidate pair exceeds512; no silent truncation")
    hidden = _forward(model, encoded)
    # Single pair has no padding; use the existing masked-sum/division convention.
    return hidden.sum(0) / encoded["attention_mask"].sum()
