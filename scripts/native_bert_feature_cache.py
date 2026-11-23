"""Label-free planning for a pinned 768-D Path A BERT feature overlay."""

import hashlib

from scripts.native_language_cache import pair_key


def _token_lengths(tokenizer, texts, candidates=None):
    lengths = []
    for offset in range(0, len(texts), 128):
        kwargs = {"add_special_tokens": True, "truncation": False}
        if candidates is not None:
            kwargs["text_pair"] = candidates[offset : offset + 128]
        encoded = tokenizer(texts[offset : offset + 128], **kwargs)["input_ids"]
        lengths.extend(len(ids) for ids in encoded)
    return lengths


def build_plan(
    *,
    state_texts,
    state_splits,
    pairs,
    query_map,
    text_observation_ids,
    tokenizer,
    width=768,
    max_tokens=8192,
):
    """Return metadata and storage shapes without reading labels or media.

    `state_splits` maps train/validation to the exact source state hashes.
    Query candidates are checked against canonical question/candidate hashes;
    query_map is used only as an identity-completeness gate.
    """
    if type(width) is not int or width < 1 or type(max_tokens) is not int or max_tokens < 2:
        raise ValueError("positive feature width and token budget are required")
    if set(state_splits) != {"train", "validation"} or any(
        not values for values in state_splits.values()
    ):
        raise ValueError("nonempty train/validation state groups required")
    train_states, validation_states = map(set, (state_splits["train"], state_splits["validation"]))
    if train_states & validation_states:
        raise ValueError("text state identity overlaps train and validation")
    if train_states | validation_states != set(state_texts):
        raise ValueError("split assignment must cover every unique source text state")
    if text_observation_ids != {"text:" + key for key in state_texts}:
        raise ValueError("text observation identities do not match source state hashes")
    if not pairs or set(pairs) != set(query_map):
        raise ValueError("complete query pairs/map identity required")

    states, state_lengths, split_lengths = {}, {}, {"train": [], "validation": []}
    for split, keys in state_splits.items():
        ordered = sorted(keys)
        lengths = _token_lengths(tokenizer, [state_texts[key] for key in ordered])
        if len(lengths) != len(ordered) or any(not 2 <= n <= max_tokens for n in lengths):
            raise ValueError("state tokenization exceeded the full-token budget")
        state_lengths.update(zip(ordered, lengths, strict=True))
        split_lengths[split] = list(zip(ordered, lengths, strict=True))
        offset = 0
        filename = split + "-states.npy"
        for key, count in split_lengths[split]:
            states[key] = dict(file=filename, offset=offset, tokens=count, split=split)
            offset += count

    query_keys = sorted(pairs)
    for key in query_keys:
        pair = pairs[key]
        if not isinstance(pair, list) or len(pair) != 2 or key != pair_key(*pair):
            raise ValueError("query key must hash its exact question/candidate strings")
    query_lengths = _token_lengths(
        tokenizer,
        [pairs[key][0] for key in query_keys],
        [pairs[key][1] for key in query_keys],
    )
    if len(query_lengths) != len(query_keys) or any(
        not 2 <= n <= max_tokens for n in query_lengths
    ):
        raise ValueError("query/candidate pair exceeded the full-token budget")
    queries = {
        key: dict(offset=i, tokens=query_lengths[i]) for i, key in enumerate(query_keys)
    }

    shapes = {
        split + "-states.npy": [sum(count for _, count in split_lengths[split]), width]
        for split in ("train", "validation")
    }
    shapes["queries.npy"] = [len(query_keys), width]
    state_order = sorted(states, key=lambda key: (states[key]["split"], states[key]["offset"]))
    return dict(
        states=states,
        queries=queries,
        shapes=shapes,
        tensor_bytes=sum(rows * columns * 4 for rows, columns in shapes.values()),
        width=width,
        max_tokens=max_tokens,
        state_order_sha256=hashlib.sha256("\n".join(state_order).encode()).hexdigest(),
        query_order_sha256=hashlib.sha256("\n".join(query_keys).encode()).hexdigest(),
        state_count=len(states),
        query_count=len(queries),
        state_over_512=sum(count > 512 for count in state_lengths.values()),
        query_over_512=sum(count > 512 for count in query_lengths),
        maximum_state_tokens=max(state_lengths.values()),
        maximum_query_tokens=max(query_lengths),
    )
