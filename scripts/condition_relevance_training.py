"""Contracts for the matched condition-relevance comparison."""

import math
import random
from collections import Counter
from pathlib import Path

import numpy as np

from scripts.decision_semantic_training import row_key as old_row_key
from scripts.decision_supervision import state_key
from scripts.semantic_state_diagnostic import metrics


def sample_order(pools, sizes=(8, 8, 2), seed=17):
    if len(pools) != 3 or len(sizes) != 3 or any(type(n) is not int or n < 1 for n in sizes):
        raise ValueError("three positive cohort sizes required")
    rows = [r for pool in pools for r in pool]
    if any(r["split"] != "train" for r in rows) or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("unique TRAIN-only rows required")
    steps = len(pools[0]) // sizes[0]
    if steps < 1 or any(len(p) != steps * n for p, n in zip(pools, sizes, strict=True)):
        raise ValueError("all pools must be exhausted once in complete updates")
    rng = random.Random(seed)
    buckets = [[[p, i] for i in range(len(pool))] for p, pool in enumerate(pools)]
    for bucket in buckets:
        rng.shuffle(bucket)
    order = []
    for step in range(steps):
        batch = [
            v for b, n in zip(buckets, sizes, strict=True) for v in b[step * n : (step + 1) * n]
        ]
        rng.shuffle(batch)
        order.append(batch)
    return order


def row_key(row):
    if row["source"] == "condition_relevance_v1":
        return f"relevance:{row['field']}:q{row['query_index']}:{row['option_form']}"
    return old_row_key(row)


def metric_summary(rows, scores, schema):
    scores = np.asarray(scores)
    if (
        not schema
        or dict(Counter(row_key(r) for r in rows)) != schema
        or scores.ndim != 2
        or len(scores) != len(rows)
    ):
        raise ValueError("exact complete metric slices and aligned scores required")
    if any(
        len(r["options"]) > scores.shape[1] or np.any(scores[i, len(r["options"]) :] > -1e29)
        for i, r in enumerate(rows)
    ):
        raise ValueError("all and only supplied candidates must be scored")
    target = np.array([r["options"].index(r["target"]) for r in rows])
    result = {}
    for key in sorted(schema):
        indices = [i for i, r in enumerate(rows) if row_key(r) == key]
        result[key] = {
            k: v
            for k, v in metrics(scores[indices], target[indices]).items()
            if k != "mean_confidence"
        }
    return dict(
        overall=metrics(scores, target),
        metrics=result,
        macro={
            k: float(np.mean([m[k] for m in result.values()]))
            for k in ("accuracy", "nll", "brier", "ece")
        },
        minimum_accuracy=min(m["accuracy"] for m in result.values()),
    )


def accounting(rows):
    return dict(
        consumed=len(rows),
        unique_questions=len({r["id"] for r in rows}),
        repeated_questions=len(rows) - len({r["id"] for r in rows}),
        unique_states=len({state_key(r["state"]) for r in rows}),
        unique_states_by_source={
            s: len({state_key(r["state"]) for r in rows if r["source"] == s})
            for s in sorted({r["source"] for r in rows})
        },
        sources=dict(Counter(r["source"] for r in rows)),
        tasks=dict(Counter(row_key(r) for r in rows)),
        modalities={"text": len(rows)},
        native_assets=0,
    )


def balanced_relevance_loss(logits, labels):
    import torch
    from torch.nn import functional as functional

    if (
        logits.ndim != 1
        or len(labels) != len(logits)
        or any(type(v) is not bool for v in labels)
        or not any(labels)
        or all(labels)
        or not torch.isfinite(logits).all()
    ):
        raise ValueError("finite aligned logits and both boolean classes required")
    positive = torch.tensor(labels, device=logits.device, dtype=torch.bool)
    losses = functional.binary_cross_entropy_with_logits(
        logits, positive.to(logits.dtype), reduction="none"
    )
    return (losses[positive].mean() + losses[~positive].mean()) / 2


def training_objective(scores, relevance, row, teacher, coefficient):
    import torch
    from torch.nn import functional as functional

    from scripts.condition_binding_retention import retention_kl

    if not math.isfinite(coefficient) or coefficient < 0:
        raise ValueError("finite nonnegative auxiliary coefficient required")
    if scores.ndim != 1 or len(scores) != len(row["options"]) or not torch.isfinite(scores).all():
        raise ValueError("finite scores for every option required")
    target = torch.tensor([row["options"].index(row["target"])], device=scores.device)
    ce = functional.cross_entropy(scores[None], target)
    objective, kl, auxiliary = ce, None, None
    if teacher is not None:
        kl = retention_kl(scores, [len(scores)], teacher)
        objective = objective + kl
    if row["source"] == "condition_relevance_v1":
        auxiliary = balanced_relevance_loss(relevance, row["supervision"]["request_relevant"])
        objective = objective + coefficient * auxiliary
    stats = {
        k: None if v is None else float(v.detach())
        for k, v in dict(ce=ce, kl=kl, relevance=auxiliary).items()
    }
    return objective, stats


def save_trainable(model, head, names, path):
    import torch

    values = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
    if set(values) != set(names):
        raise ValueError("trainable core scope changed")
    values.update({"auxiliary." + n: p.detach().cpu().clone() for n, p in head.named_parameters()})
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(values, temporary)
    temporary.replace(path)


def load_trainable(model, head, names, path):
    import torch

    values = torch.load(path, map_location="cpu", weights_only=True)
    auxiliary = {"auxiliary." + n for n, _ in head.named_parameters()}
    if set(values) != set(names) | auxiliary:
        raise ValueError("checkpoint core/auxiliary scope mismatch")
    loaded = model.load_state_dict({n: v for n, v in values.items() if n in names}, strict=False)
    if loaded.unexpected_keys or set(loaded.missing_keys) != set(model.state_dict()) - set(names):
        raise ValueError("strict core replacement failed")
    head.load_state_dict({n.removeprefix("auxiliary."): values[n] for n in auxiliary}, strict=True)


def forward_with_relevance(model, head, encoded):
    hidden = model.deberta(**encoded, return_dict=True).last_hidden_state
    pooled = model.dropout(model.pooler(hidden))
    return model.classifier(pooled)[:, 0], head(pooled).squeeze(-1)


def bootstrap_relevance(rows, baseline, candidate, replicates=10000, seed=17):
    baseline, candidate = np.asarray(baseline), np.asarray(candidate)
    if (
        not rows
        or baseline.ndim != 2
        or baseline.shape != candidate.shape
        or len(rows) != len(baseline)
        or not np.isfinite(baseline).all()
        or not np.isfinite(candidate).all()
        or replicates < 1
        or any(r["source"] != "condition_relevance_v1" for r in rows)
    ):
        raise ValueError("aligned finite condition-relevance predictions required")
    target = np.array([r["options"].index(r["target"]) for r in rows])
    correct = np.stack([baseline.argmax(-1) == target, candidate.argmax(-1) == target])
    states = sorted({r["group"] for r in rows})
    tasks = sorted({row_key(r) for r in rows})
    table = []
    for state in states:
        indices = sorted(
            (i for i, r in enumerate(rows) if r["group"] == state), key=lambda i: row_key(rows[i])
        )
        if [row_key(rows[i]) for i in indices] != tasks:
            raise ValueError("every state must retain every slice exactly once")
        table.append(correct[:, indices])
    table = np.stack(table, axis=1)
    observed = table.mean(1).min(1)
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(replicates):
        minimum = table[:, rng.integers(len(states), size=len(states))].mean(1).min(1)
        differences.append(float(minimum[1] - minimum[0]))
    return dict(
        baseline_minimum=float(observed[0]),
        candidate_minimum=float(observed[1]),
        difference=float(observed[1] - observed[0]),
        interval95=np.quantile(differences, [0.025, 0.975], method="linear").tolist(),
        states=len(states),
        slices=len(tasks),
        replicates=replicates,
        seed=seed,
        method="paired whole-state percentile; recompute minimum each replicate",
        validation_selected=True,
        blind_evaluation=False,
    )
