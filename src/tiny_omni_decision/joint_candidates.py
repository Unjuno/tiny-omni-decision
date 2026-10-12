"""Experimental BERT joint-candidate readout, reusing existing model weights."""


def encode_joint(tokenizer, question, state, options, max_length=512):
    if len(options) < 2 or len(set(options)) != len(options):
        raise ValueError("at least two unique options required")
    if not all(isinstance(x, str) and x.strip() for x in [question, *options]):
        raise ValueError("question and options must be nonempty strings")
    if not isinstance(state, str):
        raise ValueError("state must be serialized text")

    def tokens(text):
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    cls, sep = tokenizer.cls_token_id, tokenizer.sep_token_id
    if cls is None or sep is None:
        raise ValueError("BERT special tokens required")
    ids = [cls] + tokens("Question: " + question)
    markers = []
    for option in options:
        markers.append(len(ids))
        ids += [sep] + tokens("Candidate answer: " + option)
    ids += [sep]
    room = max_length - len(ids) - 1
    if room < 1:
        raise ValueError("question/options exceed context budget")
    state_ids = tokens("State: " + state)
    types = [0] * len(ids) + [1] * (min(room, len(state_ids)) + 1)
    ids += state_ids[:room] + [sep]
    return dict(input_ids=ids, attention_mask=[1] * len(ids), token_type_ids=types), markers, {
        "state_tokens": len(state_ids), "kept_state_tokens": min(room, len(state_ids)),
        "state_truncated": len(state_ids) > room,
    }


def joint_logits(model, batch, positions):
    """One row per question, all candidate markers contextualized together.

    Positions are an unpadded list per question. Returns variable-width logits.
    This new readout is not assumed equivalent to the original CLS scorer.
    """
    import torch

    if model.config.model_type != "bert" or model.config.num_labels != 1:
        raise ValueError("single-logit BERT model required")
    if len(positions) != batch["input_ids"].shape[0]:
        raise ValueError("question count mismatch")
    for i, indices in enumerate(positions):
        if len(indices) < 2 or len(set(indices)) != len(indices):
            raise ValueError("invalid option markers")
        if any(j < 0 or j >= batch["input_ids"].shape[1]
               or not bool(batch["attention_mask"][i, j]) for j in indices):
            raise ValueError("marker outside real tokens")
    hidden = model.bert(**batch).last_hidden_state
    values = []
    for i, indices in enumerate(positions):
        pooled = torch.tanh(model.bert.pooler.dense(hidden[i, indices]))
        values.append(model.classifier(model.dropout(pooled)).squeeze(-1))
    return values
