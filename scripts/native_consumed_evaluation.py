"""Exact consumed-TRAIN membership and common-task descriptive gaps."""

from collections import Counter

import numpy as np

from scripts.native_fit import accounting, summarize


def consumed_plan(rows, histories):
    by_id = {r["id"]: r for r in rows}
    if not rows or len(by_id) != len(rows) or any(r["split"] != "train" for r in rows):
        raise ValueError("unique TRAIN rows required")
    counts = None
    if not histories:
        raise ValueError("actual complete histories required")
    for history in histories.values():
        if not history or [h["step"] for h in history] != list(range(1, len(history) + 1)):
            raise ValueError("contiguous complete update history required")
        current = Counter(ident for h in history for ident in h["sample_ids"])
        if (
            not current
            or not set(current) <= by_id.keys()
            or any(not h["sample_ids"] for h in history)
        ):
            raise ValueError("known consumed TRAIN IDs and nonempty updates required")
        if counts is not None and current != counts:
            raise ValueError("histories must have identical consumed multisets")
        counts = current
    ids = sorted(counts)
    return dict(
        ids=ids,
        multiplicities=[counts[i] for i in ids],
        occurrence_count=sum(counts.values()),
        unique_count=len(ids),
        unconsumed_count=len(rows) - len(ids),
        unique_accounting=accounting([by_id[i] for i in ids]),
        occurrence_accounting=accounting([by_id[i] for i in ids for _ in range(counts[i])]),
    )


def common_task_gap(train, train_scores, validation, validation_scores):
    """Shared task labels do not make TRAIN/validation distributions identical."""
    a, b = {r["id"] for r in train}, {r["id"] for r in validation}
    if (
        not train
        or not validation
        or len(a) != len(train)
        or len(b) != len(validation)
        or a & b
        or any(r["split"] != "train" for r in train)
        or any(r["split"] != "validation" for r in validation)
    ):
        raise ValueError("unique disjoint TRAIN/validation IDs required")
    train_tasks = {(r["source"], r["task"]) for r in train}
    val_tasks = {(r["source"], r["task"]) for r in validation}
    common = train_tasks & val_tasks
    if not common:
        raise ValueError("no common tasks")
    ix = [i for i, r in enumerate(train) if (r["source"], r["task"]) in common]
    jx = [i for i, r in enumerate(validation) if (r["source"], r["task"]) in common]
    first = summarize([train[i] for i in ix], np.asarray(train_scores)[ix])
    second = summarize([validation[i] for i in jx], np.asarray(validation_scores)[jx])
    return dict(
        common_tasks=[":".join(k) for k in sorted(common)],
        train_only_tasks=[":".join(k) for k in sorted(train_tasks - val_tasks)],
        validation_only_tasks=[":".join(k) for k in sorted(val_tasks - train_tasks)],
        train=first,
        validation=second,
        validation_minus_train={k: second["macro"][k] - v for k, v in first["macro"].items()},
        caveat="Descriptive gap; populations/prevalence can differ; not a causal effect.",
    )
