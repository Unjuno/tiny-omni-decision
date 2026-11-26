"""Inference batching and exact-weight reuse, independent of GPU initialization."""

from itertools import islice

import numpy as np

from scripts.condition_binding_training import pack_scores
from scripts.decision_semantic_training import render_pairs

ROLES = ("initial", "ce_endpoint", "ce_selected", "relevance_endpoint", "relevance_selected")


def evaluation_groups(roles):
    if set(roles) != set(ROLES):
        raise ValueError("all five fixed comparison roles required")
    groups = {}
    for name in ROLES:
        identity = roles[name]
        if identity["sha256"] not in groups:
            groups[identity["sha256"]] = dict(identity, canonical_role=name, roles=[])
        groups[identity["sha256"]]["roles"].append(name)
    return list(groups.values())


def score_rows(rows, tokenizer, forward, *, pair_batch, max_length):
    """Match the fit's global candidate batches without materializing every pair."""
    if not rows or type(pair_batch) is not int or pair_batch < 1 or max_length != 512:
        raise ValueError("nonempty rows and fixed token contract required")
    pairs = (pair for row in rows for pair in render_pairs(row))
    scores = []
    while batch := list(islice(pairs, pair_batch)):
        first, second = map(list, zip(*batch, strict=True))
        encoded = tokenizer(
            first,
            second,
            padding=True,
            truncation="only_second",
            max_length=max_length,
            return_tensors="pt",
        )
        values = np.asarray(forward(encoded))
        if values.shape != (len(batch),) or not np.isfinite(values).all():
            raise ValueError("finite native score for every candidate required")
        scores.extend(values.tolist())
    result, offset = [], 0
    for row in rows:
        result.append(np.asarray(scores[offset : offset + len(row["options"])]))
        offset += len(row["options"])
    if offset != len(scores):
        raise ValueError("unconsumed candidate scores")
    return pack_scores(result)
