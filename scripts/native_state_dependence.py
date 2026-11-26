"""TRAIN-only matched-input pairs and descriptive state-dependence metrics."""

import hashlib
import json
from collections import Counter, defaultdict, deque

import numpy as np

from scripts.semantic_state_diagnostic import metrics


def _key(value, seed):
    return hashlib.sha256(
        json.dumps([seed, value], ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def _answer(row):
    return row["options"][row["target_index"]]


def _pairs(rows, seed):
    remaining = sorted(rows, key=lambda r: _key(r["id"], seed))
    counts = Counter(_answer(r) for r in remaining)
    result = []
    while remaining and len(counts) > 1:
        a = remaining.pop(0)
        counts[_answer(a)] -= 1
        if not counts[_answer(a)]:
            del counts[_answer(a)]
        match = next(
            (
                i
                for i, b in enumerate(remaining)
                if _answer(a) != _answer(b) and a["observation"] != b["observation"]
            ),
            None,
        )
        if match is None:
            continue
        b = remaining.pop(match)
        counts[_answer(b)] -= 1
        if not counts[_answer(b)]:
            del counts[_answer(b)]
        if dict(zip(a["options"], a["queries"], strict=True)) != dict(
            zip(b["options"], b["queries"], strict=True)
        ):
            raise ValueError("same question/candidate text must use identical query features")
        result.append((a, b))
    return result


def make_plan(rows, consumed, *, limit=8, seed=17):
    """No quality-dependent selection; preserve unpairable tasks in coverage.

    Exact question and complete candidate texts must match. Row order is irrelevant;
    independently hash-sort each condition, then greedily pair different answers and
    observations without reusing rows. Rotate conditions before taking the fixed cap.
    This is a contrast diagnostic, not a representative accuracy benchmark.
    """
    ids = {r["id"] for r in rows}
    if (
        not rows
        or len(ids) != len(rows)
        or not set(consumed) <= ids
        or type(limit) is not int
        or limit < 1
        or any(
            r["split"] != "train"
            or not 2 <= len(r["options"]) <= 40
            or len(set(r["options"])) != len(r["options"])
            or len(r["queries"]) != len(r["options"])
            or not 0 <= r["target_index"] < len(r["options"])
            for r in rows
        )
    ):
        raise ValueError("unique valid TRAIN rows and known consumed IDs required")
    tasks = sorted({(r["modality"], r["source"], r["task"]) for r in rows})
    grouped = defaultdict(list)
    for row in rows:
        state = "consumed" if row["id"] in consumed else "unconsumed"
        grouped[(state, row["modality"], row["source"], row["task"])].append(row)
    pairs, coverage = [], []
    for state in ("consumed", "unconsumed"):
        for modality, source, task in tasks:
            items = grouped[state, modality, source, task]
            conditions = defaultdict(list)
            for row in items:
                conditions[row["question"], tuple(sorted(row["options"]))].append(row)
            queues = [
                deque(_pairs(conditions[k], seed))
                for k in sorted(conditions, key=lambda k: _key(k, seed))
            ]
            available = sum(len(q) for q in queues)
            chosen = []
            while len(chosen) < limit and any(queues):
                for queue in queues:
                    if queue and len(chosen) < limit:
                        chosen.append(queue.popleft())
            coverage.append(
                dict(
                    stratum=state,
                    modality=modality,
                    source=source,
                    task=task,
                    rows=len(items),
                    observations=len({r["observation"] for r in items}),
                    condition_groups=len(conditions),
                    constructed_eligible_pairs=available,
                    selected_pairs=len(chosen),
                )
            )
            for a, b in chosen:
                pairs.append(
                    dict(
                        stratum=state,
                        modality=modality,
                        source=source,
                        task=task,
                        row_ids=[a["id"], b["id"]],
                        observations=[a["observation"], b["observation"]],
                        groups=[a["group"], b["group"]],
                        question=a["question"],
                        options=list(a["options"]),
                        option_ids=list(a["option_ids"]),
                        queries=list(a["queries"]),
                        targets=[a["target_index"], a["options"].index(_answer(b))],
                    )
                )
    return dict(seed=seed, max_pairs_per_task_and_stratum=limit, pairs=pairs, coverage=coverage)


def pair_metrics(pairs, scores):
    scores = np.asarray(scores)
    if (
        not pairs
        or scores.shape != (len(pairs), 2, 40)
        or not np.isfinite(scores).all()
        or any(
            len(p["targets"]) != 2
            or len(set(p["targets"])) != 2
            or not 2 <= len(p["options"]) <= 40
            or any(not 0 <= t < len(p["options"]) for t in p["targets"])
            or np.any(scores[i, :, len(p["options"]) :] > -1e29)
            for i, p in enumerate(pairs)
        )
    ):
        raise ValueError("finite complete pair scores and distinct aligned targets required")
    z = scores.astype(np.float64)
    p = np.exp(z - z.max(axis=2, keepdims=True))
    p /= p.sum(axis=2, keepdims=True)
    target = np.array([r["targets"] for r in pairs])
    prediction = scores.argmax(axis=2)
    indices = np.arange(len(pairs))
    a = p[indices, 0, target[:, 0]] - p[indices, 1, target[:, 0]]
    b = p[indices, 1, target[:, 1]] - p[indices, 0, target[:, 1]]
    return dict(
        pairs=len(pairs),
        individual=metrics(scores.reshape(-1, 40), target.reshape(-1)),
        both_correct=float(np.all(prediction == target, axis=1).mean()),
        prediction_changes=float((prediction[:, 0] != prediction[:, 1]).mean()),
        correct_direction_fraction=float(((a > 0) & (b > 0)).mean()),
        symmetric_target_effect=float(((a + b) / 2).mean()),
        probability_total_variation=float((np.abs(p[:, 0] - p[:, 1]).sum(axis=1) / 2).mean()),
    )


def summarize(plan, scores):
    pairs = plan["pairs"]
    result = dict(overall=pair_metrics(pairs, scores), strata={}, modalities={}, tasks={})
    for group, keys in (
        ("strata", ("stratum",)),
        ("modalities", ("stratum", "modality")),
        ("tasks", ("stratum", "source", "task")),
    ):
        for item in plan["coverage"]:
            key = ":".join(item[k] for k in keys)
            if key in result[group]:
                continue
            selected = [i for i, p in enumerate(pairs) if all(p[k] == item[k] for k in keys)]
            result[group][key] = (
                pair_metrics([pairs[i] for i in selected], scores[selected]) if selected else None
            )
    return result
