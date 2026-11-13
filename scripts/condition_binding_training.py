"""Stable variable-choice metrics without importing a model runtime."""

import numpy as np

from scripts.condition_binding_corpus import pair_metrics
from scripts.semantic_state_diagnostic import metrics


def pack_scores(values):
    if not values or any(
        len(v) < 2 or not np.isfinite(v).all() or np.max(np.abs(v)) > 1e10 for v in values
    ):
        raise ValueError("invalid finite option scores")
    result = np.full((len(values), max(map(len, values))), -1e30, dtype=np.float32)
    for i, value in enumerate(values):
        result[i, : len(value)] = value
    return result


def score_summary(rows, scores):
    target = np.array([r["options"].index(r["target"]) for r in rows])
    result = dict(overall=metrics(scores, target), tasks={})
    for task in sorted({r["task"] for r in rows}):
        indices = [i for i, r in enumerate(rows) if r["task"] == task]
        result["tasks"][task] = metrics(scores[indices], target[indices])
    if all(r["source"] == "binding" for r in rows):
        if len(rows) % 4 or any(
            len({r["group"] for r in rows[i : i + 4]}) != 1 for i in range(0, len(rows), 4)
        ):
            raise ValueError("semantic questions must retain groups of four")
        result.update(pair_metrics(scores.argmax(-1).reshape(-1, 4), target.reshape(-1, 4)))
    return result
