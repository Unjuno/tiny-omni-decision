"""Pure selection, accounting and rendering for the fixed DeBERTa experiment."""

import math
import random
from collections import Counter

import numpy as np

from scripts.decision_supervision import state_key
from scripts.prepare_decision_supervision import metric_key
from scripts.semantic_state_diagnostic import metrics


def render_pairs(row):
    # No source, task, oracle, group, option-ID structure or target enters the model.
    if (
        not isinstance(row["state"], str)
        or not isinstance(row["question"], str)
        or len(row["options"]) < 2
        or len(set(row["options"])) != len(row["options"])
        or any(not isinstance(o, str) or not o for o in row["options"])
    ):
        raise ValueError("public state/question and distinct candidate texts required")
    return [
        (f"Question: {row['question']}\nCandidate answer: {o}", f"State: {row['state']}")
        for o in row["options"]
    ]


def sample_order(train, replay, per_cohort=8, replay_per_update=2, seed=17):
    if min(per_cohort, replay_per_update) < 1 or not train or not replay:
        raise ValueError("positive nonempty schedule required")
    ids = [r["id"] for r in train + replay]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate training ID")
    buckets = [
        [(0, i) for i, r in enumerate(train) if r["task"].startswith(t + ":")]
        for t in ("binding", "numeric")
    ]
    buckets.append([(1, i) for i in range(len(replay))])
    sizes = [per_cohort, per_cohort, replay_per_update]
    steps = len(buckets[0]) // per_cohort
    if (
        sum(map(len, buckets[:2])) != len(train)
        or steps < 1
        or any(len(b) != steps * n for b, n in zip(buckets, sizes, strict=True))
    ):
        raise ValueError("schedule must exhaust all buckets once without partial updates")
    rng = random.Random(seed)
    for b in buckets:
        rng.shuffle(b)
    result = []
    for step in range(steps):
        batch = [
            list(v)
            for b, n in zip(buckets, sizes, strict=True)
            for v in b[step * n : (step + 1) * n]
        ]
        rng.shuffle(batch)
        result.append(batch)
    return result


def row_key(row):
    return metric_key(row, row["source"] != "programmatic_decision_v1")


def metric_summary(rows, scores, schema):
    scores = np.asarray(scores)
    actual = dict(Counter(row_key(r) for r in rows))
    if actual != schema or scores.ndim != 2 or len(scores) != len(rows):
        raise ValueError("all exact metric slices and aligned scores required")
    target = np.array([r["options"].index(r["target"]) for r in rows])
    if any(len(r["options"]) > scores.shape[1] for r in rows):
        raise ValueError("missing candidate scores")
    if any(np.any(scores[i, len(r["options"]) :] > -1e29) for i, r in enumerate(rows)):
        raise ValueError("unsupplied candidates must be masked")
    values = {}
    for key in sorted(schema):
        indices = [i for i, r in enumerate(rows) if row_key(r) == key]
        m = metrics(scores[indices], target[indices])
        values[key] = {k: v for k, v in m.items() if k != "mean_confidence"}
    return dict(
        overall=metrics(scores, target),
        metrics=values,
        macro={
            k: float(np.mean([m[k] for m in values.values()]))
            for k in ("accuracy", "nll", "brier", "ece")
        },
        minimum_accuracy=min(m["accuracy"] for m in values.values()),
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
        modalities={"text": len(rows)},
        sources=dict(Counter(r["source"] for r in rows)),
        tasks=dict(Counter(row_key(r) for r in rows)),
        native_assets=0,
    )


def trainable_names(names):
    prefixes = ("deberta.encoder.", "pooler.", "classifier.")
    if any(not n.startswith((*prefixes, "deberta.embeddings.")) for n in names):
        raise ValueError("unexpected model module")
    selected = [n for n in names if n.startswith(prefixes)]
    if not selected or not any(n.startswith("deberta.embeddings.") for n in names):
        raise ValueError("expected frozen embeddings and trainable encoder absent")
    return selected


def learning_rate(step, total, warmup, peak):
    if not 1 <= step <= total or not 1 <= warmup < total or not peak > 0:
        raise ValueError("invalid finite update schedule")
    if step <= warmup:
        return peak * step / warmup
    return peak * (1 + math.cos(math.pi * (step - warmup) / (total - warmup))) / 2


def progress_record(train, order, completed_updates, forwards, backwards):
    planned = [train[s][i] for batch in order for s, i in batch]
    committed = sum(map(len, order[:completed_updates]))
    if not (
        0 <= completed_updates <= len(order) and committed <= backwards <= forwards <= len(planned)
    ):
        raise ValueError("inconsistent partial execution counters")
    return dict(
        optimizer_updates_completed=completed_updates,
        completed_question_backwards=backwards,
        consumed=accounting(planned[:forwards]),
        partial_update_forwards=forwards - committed,
    )


def bootstrap_minimum(rows, baseline, candidate, replicates=10000, seed=17):
    """Stratified paired whole-state bootstrap of the minimum new-slice accuracy."""
    baseline, candidate = np.asarray(baseline), np.asarray(candidate)
    if (
        baseline.shape != candidate.shape
        or baseline.ndim != 2
        or len(rows) != len(baseline)
        or not np.isfinite(baseline).all()
        or not np.isfinite(candidate).all()
        or replicates < 1
    ):
        raise ValueError("aligned finite scores required")
    target = np.array([r["options"].index(r["target"]) for r in rows])
    correct = np.stack([baseline.argmax(-1) == target, candidate.argmax(-1) == target])
    cohorts = {}
    for cohort in ("binding", "numeric"):
        subset = [i for i, r in enumerate(rows) if r["task"].startswith(cohort + ":")]
        states = sorted({rows[i]["group"] for i in subset})
        tasks = sorted({rows[i]["task"] for i in subset})
        if not states or not tasks:
            raise ValueError("both complete cohorts required")
        table = []
        for state in states:
            indices = sorted(
                (i for i in subset if rows[i]["group"] == state), key=lambda i: rows[i]["task"]
            )
            if [rows[i]["task"] for i in indices] != tasks:
                raise ValueError("each state must retain all tasks once")
            table.append(correct[:, indices])
        cohorts[cohort] = np.stack(table, axis=1)  # model x state x task
    if sum(a.shape[1] * a.shape[2] for a in cohorts.values()) != len(rows):
        raise ValueError("only new complete cohorts allowed")
    observed = np.concatenate([a.mean(1) for a in cohorts.values()], axis=1).min(1)
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(replicates):
        means = [a[:, rng.integers(a.shape[1], size=a.shape[1])].mean(1) for a in cohorts.values()]
        low = np.concatenate(means, axis=1).min(1)
        differences.append(float(low[1] - low[0]))
    return dict(
        baseline_minimum=float(observed[0]),
        candidate_minimum=float(observed[1]),
        difference=float(observed[1] - observed[0]),
        interval95=np.quantile(differences, [0.025, 0.975], method="linear").tolist(),
        replicates=replicates,
        seed=seed,
        method="paired stratified whole-state percentile",
        states_by_cohort={k: a.shape[1] for k, a in cohorts.items()},
        validation_selected=True,
        blind_evaluation=False,
    )
