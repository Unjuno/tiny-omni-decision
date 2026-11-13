"""Pure helpers for a fixed TRAIN-only semantic bridge diagnosis, not a product API."""

import hashlib
import itertools
import json

import numpy as np

from tiny_omni_decision.dataset import CLEVR4_TAXONOMIES

FIELDS = tuple(CLEVR4_TAXONOMIES)
QUESTIONS = (
    "What is the color of the objects?",
    "What is the texture of the objects?",
    "How many objects are there?",
    "What is the shape of the objects?",
)


def oracle_state(row):
    state = {field: str(row[field]) for field in FIELDS}
    if any(state[field] not in CLEVR4_TAXONOMIES[field] for field in FIELDS):
        raise ValueError("unknown annotation value")
    return state


def select_train(rows, count, seed=17):
    if not 0 < count <= len(rows):
        raise ValueError("invalid cohort size")
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids) or any(r["split"] != "train" for r in rows):
        raise ValueError("only unique TRAIN identities are allowed")
    for row in rows:
        oracle_state(row)
    return sorted(rows, key=lambda r: (
        hashlib.sha256(f"{seed}:{r['id']}".encode()).hexdigest(), r["id"]))[:count]


def options():
    return [{"id": f"{field}:{value}", "description": f"{field}: {value}"}
            for field in FIELDS for value in CLEVR4_TAXONOMIES[field]]


def targets(state):
    return np.array([10 * i + CLEVR4_TAXONOMIES[field].index(str(state[field]))
                     for i, field in enumerate(FIELDS)], dtype=np.int64)


def predicted_state(logits):
    scores = np.asarray(logits)
    if scores.shape != (4, 10) or not np.isfinite(scores).all():
        raise ValueError("native scores must be finite 4x10")
    return {field: CLEVR4_TAXONOMIES[field][int(scores[i].argmax())]
            for i, field in enumerate(FIELDS)}


def model_pair(state, question, candidate):
    if state:
        state = oracle_state(state)  # Retain only the four input facts.
    return (f"Question: {question}\nCandidate answer: {candidate}",
            "State: " + json.dumps(state, sort_keys=True, separators=(",", ":")))


def log_probabilities(scores):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim < 2 or scores.shape[-1] < 2 or not np.isfinite(scores).all():
        raise ValueError("nonempty finite option scores required")
    shifted = scores - scores.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def metrics(scores, target):
    scores, target = np.asarray(scores), np.asarray(target)
    if (scores.ndim != 2 or len(scores) == 0 or target.shape != (len(scores),)
            or not np.issubdtype(target.dtype, np.integer)
            or (target < 0).any() or (target >= scores.shape[1]).any()):
        raise ValueError("invalid score/target dimensions or indices")
    logp = log_probabilities(scores)
    probability = np.exp(logp)
    prediction = scores.argmax(-1)
    correct = prediction == target
    confidence = probability.max(-1)
    truth = np.eye(scores.shape[1])[target]
    bins = np.minimum((confidence * 15).astype(int), 14)
    ece = sum(float((bins == b).mean()) * abs(
        float(correct[bins == b].mean()) - float(confidence[bins == b].mean()))
        for b in range(15) if (bins == b).any())
    return dict(count=len(scores), accuracy=float(correct.mean()),
                nll=float(-logp[np.arange(len(target)), target].mean()),
                brier=float(((probability - truth) ** 2).sum(-1).mean()),
                ece=ece, mean_confidence=float(confidence.mean()))


def summarize(scores, target):
    scores, target = np.asarray(scores), np.asarray(target)
    if scores.shape != (len(target), 4, 40) or target.shape != (len(target), 4):
        raise ValueError("expected image x four questions x forty options")
    if not np.array_equal(target // 10, np.broadcast_to(np.arange(4), target.shape)):
        raise ValueError("each question must retain its distinct typed target")
    tasks = {field: metrics(scores[:, i], target[:, i]) for i, field in enumerate(FIELDS)}
    correct = scores.argmax(-1) == target
    pairs = list(itertools.combinations(range(4), 2))
    pair_correct = np.array([correct[:, a] & correct[:, b] for a, b in pairs]).T
    restricted = np.stack([scores[:, i, i * 10:(i + 1) * 10].argmax(-1)
                           for i in range(4)], 1)
    return dict(
        overall=metrics(scores.reshape(-1, 40), target.reshape(-1)), per_question=tasks,
        macro={key: float(np.mean([m[key] for m in tasks.values()]))
               for key in ("accuracy", "nll", "brier", "ece")},
        all_four_correct=float(correct.all(1).mean()),
        changed_question_pair_both_correct=float(pair_correct.mean()),
        pair_count=int(pair_correct.size),
        per_pair_both_correct={f"{FIELDS[a]}+{FIELDS[b]}": float(pair_correct[:, i].mean())
                               for i, (a, b) in enumerate(pairs)},
        field_routing_accuracy=float((scores.argmax(-1) // 10 == np.arange(4)).mean()),
        within_correct_field_accuracy=float((restricted == target % 10).mean()),
    )


def donor_control(scores, target):
    if len(scores) < 2 or len(scores) != len(target):
        raise ValueError("donor control needs at least two aligned assets")
    return np.roll(scores, -1, axis=0).copy(), np.roll(target, -1, axis=0).copy()
