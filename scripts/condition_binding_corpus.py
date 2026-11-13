"""Deterministic conditional-fact data and selection rules, independent of torch."""

import hashlib
import itertools
import json
import random

import numpy as np

from scripts.semantic_state_diagnostic import FIELDS, QUESTIONS, options, oracle_state
from tiny_omni_decision.dataset import CLEVR4_TAXONOMIES

TRAIN_QUESTIONS = (
    (QUESTIONS[0], "Report the color shown by the objects."),
    (QUESTIONS[1], "Report the texture shown by the objects."),
    (QUESTIONS[2], "Report the count of objects in the scene."),
    (QUESTIONS[3], "Report the shape of the objects."),
)
NOVEL_QUESTIONS = (
    "Identify the objects' hue.",
    "Name the surface pattern or material.",
    "State the total number of objects.",
    "Identify the objects' geometric form.",
)
NOVEL_FIELDS = dict(color="Hue", texture="Surface", count="Quantity", shape="Form")


def signature(state):
    return json.dumps(oracle_state(state), sort_keys=True, separators=(",", ":"))


def identity(state):
    return hashlib.sha256(signature(state).encode()).hexdigest()


def all_states():
    return [dict(zip(FIELDS, values, strict=True))
            for values in itertools.product(*(CLEVR4_TAXONOMIES[k] for k in FIELDS))]


def ordered(values, label):
    return sorted(values, key=lambda state: hashlib.sha256(
        f"17:{label}:{identity(state)}".encode()).hexdigest())


def partition(known, observed, train_count=512, validation_count=256, transfer_count=256):
    known = {identity(s): oracle_state(s) for s in known}
    observed = {identity(s) for s in observed}
    train = ordered([s for k, s in known.items() if k not in observed], "train")
    unused = ordered([s for s in all_states()
                      if identity(s) not in known and identity(s) not in observed], "unused")
    if len(train) < train_count or len(unused) < validation_count + transfer_count:
        raise ValueError("insufficient independent state combinations")
    return dict(train=train[:train_count], validation=unused[:validation_count],
                transfer=unused[validation_count:validation_count + transfer_count])


def questions_for(state, split, novel_question=False, novel_options=False):
    group = identity(state)
    alternate = int(group[:8], 16) % 2
    ids, descriptions = [], []
    for option in options():
        field, value = option['id'].split(":", 1)
        ids.append(option['id'])
        descriptions.append(f"{NOVEL_FIELDS[field]} = {value}" if novel_options else (
            f"The {field} is {value}." if alternate else option['description']))
    result = []
    for i, field in enumerate(FIELDS):
        question = (NOVEL_QUESTIONS[i] if novel_question else
                    TRAIN_QUESTIONS[i][int(group[8 + i], 16) % 2])
        result.append(dict(
            id=f"binding:{group}:{field}:q{int(novel_question)}:o{int(novel_options)}",
            group=group, state=signature(state), question=question, split=split,
            options=descriptions.copy(), option_ids=ids.copy(),
            target_id=f"{field}:{state[field]}",
            target=descriptions[ids.index(f"{field}:{state[field]}")], task=field,
            source="binding", novel_question=novel_question, novel_options=novel_options))
    return result


def reverse_options(row):
    return dict(row, options=list(reversed(row['options'])),
                option_ids=list(reversed(row['option_ids'])))


def schedule(binding_count, replay_count):
    if binding_count != 2048 or replay_count != 2048:
        raise ValueError("fixed512 updates require2048 unique rows per source")
    rng = random.Random(17)
    orders = [list(range(binding_count)), list(range(replay_count))]
    for order in orders:
        rng.shuffle(order)
    return [[(source, index) for source in range(2)
             for index in orders[source][step * 4:(step + 1) * 4]] for step in range(512)]


def retention_ok(metrics, tasks, baseline_metrics, baseline_tasks):
    if set(tasks) != set(baseline_tasks) or set(metrics) != set(baseline_metrics):
        return False
    if any(not np.isfinite(tasks[k]) or tasks[k] < baseline_tasks[k] - 0.01 - 1e-12
           for k in tasks):
        return False
    for source in metrics:
        for key, limit in (("nll", 0.02), ("brier", 0.01), ("ece", 0.02)):
            if not np.isfinite(metrics[source][key]) or (
                    metrics[source][key] > baseline_metrics[source][key] + limit + 1e-12):
                return False
    return True


def select_checkpoint(history):
    allowed = [r for r in history if r['eligible'] and np.isfinite(r['binding_nll'])]
    if not allowed:
        raise ValueError("no eligible checkpoint")
    return min(allowed, key=lambda r: (r['binding_nll'], r['step']))['step']


def pair_metrics(prediction, target):
    prediction, target = np.asarray(prediction), np.asarray(target)
    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[1] != 4:
        raise ValueError("four aligned questions per state required")
    correct = prediction == target
    return dict(all_four_correct=float(correct.all(1).mean()), pair_both_correct=float(np.mean(
        [correct[:, a] & correct[:, b] for a, b in itertools.combinations(range(4), 2)])))
