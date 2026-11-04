"""Fixed replay coverage intervention; existing language, budgets and gates retained."""

import hashlib
import json
from collections import Counter, defaultdict

import numpy as np

from scripts.condition_binding_corpus import schedule
from scripts.train_numeric_deterministic import verify_splits

EXTRA = {
    "experiment_id",
    "replay_policy",
    "language_corpus_sha256",
    "language_fit_sha256",
    "curriculum_train_sha256",
}
TASKS = dict(extremum=4, temporal_order=2, count=1, conditional_selection=1)


def verify_split_integrity(train, validation):
    verify_splits(train, validation)
    if {r["group"] for r in train} & {r["group"] for r in validation}:
        raise ValueError("group overlap")


def verify_intervention(current, parent):
    if (
        set(current) != set(parent) | EXTRA
        or {k: v for k, v in current.items() if k not in EXTRA}
        != {k: v for k, v in parent.items() if k not in EXTRA}
        or current["replay_policy"] != "two_original_two_curriculum_per_update_v1"
    ):
        raise ValueError("unplanned replay intervention")


def rank(value, label):
    return hashlib.sha256(f"17:replay-coverage:{label}:{value}".encode()).hexdigest()


def mixed_replay(original, curriculum, order):
    expected = [[list(pair) for pair in batch] for batch in schedule(2048, 2048)]
    actual = [[list(pair) for pair in batch] for batch in order]
    if actual != expected or len(original) != 2048:
        raise ValueError("fixed schedule and replay length required")
    if (
        len({r["id"] for r in original + curriculum}) != len(original) + len(curriculum)
        or any(r["split"] != "train" for r in original + curriculum)
        or any(r["source"] != "original" for r in original)
    ):
        raise ValueError("unique TRAIN replay sources required")
    groups = defaultdict(list)
    for row in curriculum:
        if row.get("source", "curriculum") != "curriculum":
            raise ValueError("unexpected curriculum source")
        groups[row["group"]].append(row)
    if len(groups) < 128 or any(
        Counter(r["task"] for r in rows) != TASKS or len({r["state"] for r in rows}) != 1
        for rows in groups.values()
    ):
        raise ValueError("complete eight-question curriculum groups required")
    selected = [
        dict(row, source="curriculum")
        for group in sorted(groups, key=lambda g: rank(g, "group"))[:128]
        for row in groups[group]
    ]
    selected.sort(key=lambda r: rank(r["id"], "question"))
    result = original.copy()
    positions = [
        index for batch in order for index in [i for source, i in batch if source == 1][2:]
    ]
    for index, row in zip(positions, selected, strict=True):
        result[index] = row
    return result


def question_hash(row):
    if row.get("split") != "train" or row.get("source") not in ("original", "curriculum"):
        raise ValueError("retention targets require approved TRAIN sources")
    return hashlib.sha256(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def cache_entry(row, logits):
    values = np.asarray(logits)
    if values.shape != (len(row["options"]),) or not np.isfinite(values).all():
        raise ValueError("finite per-candidate logits required")
    return dict(question_sha256=question_hash(row), logits=values.tolist())


def checked_logits(row, cached):
    if cached["question_sha256"] != question_hash(row):
        raise ValueError("retention question or candidate order changed")
    return cache_entry(row, cached["logits"])["logits"]
