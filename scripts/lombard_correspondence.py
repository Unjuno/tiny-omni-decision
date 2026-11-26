"""Symmetric native catalogs and semantic agreement; no labels in score API."""

import hashlib
import json
from collections import Counter, defaultdict

import numpy as np

from scripts.lombard_grid_corpus import SLOTS

SIZES = tuple(len(t) for t in SLOTS.values())
BOUNDS = np.cumsum((0, *SIZES))
OPTION_IDS = ["video-A", "video-B", "video-C", "video-D"]


def stable(*values):
    return hashlib.sha256(json.dumps([17, *values], separators=(",", ":")).encode()).hexdigest()


def signature(row):
    if set(row["targets"]) != set(SLOTS):
        raise ValueError("complete native six-slot labels required for catalog audit")
    result = tuple(row["targets"][s] for s in SLOTS)
    if any(
        value not in table.values() for value, table in zip(result, SLOTS.values(), strict=True)
    ):
        raise ValueError("unknown native command label")
    return result


def build_quartets(rows, policy):
    if policy not in {"random", "hard"} or not rows:
        raise ValueError("fixed nonempty random/hard catalog required")
    if (
        len({r["source_id"] for r in rows}) != len(rows)
        or any(not isinstance(r["source_id"], str) or not r["source_id"] for r in rows)
        or len({r["split"] for r in rows}) != 1
        or rows[0]["split"] not in {"train", "calibration", "validation"}
        or any(r["condition"] not in {"p", "l"} or not r["speaker_group"] for r in rows)
    ):
        raise ValueError("unique grouped source identities in one unsealed split required")
    groups, signatures = defaultdict(list), [signature(r) for r in rows]
    for i, r in enumerate(rows):
        groups[(r["speaker_group"], r["condition"])].append(i)
    catalogs, distances = [], Counter()
    for i, row in enumerate(rows):
        candidates = defaultdict(list)
        for j in groups[(row["speaker_group"], row["condition"])]:
            if signatures[j] != signatures[i]:
                candidates[signatures[j]].append(j)
        if len(candidates) < 3:
            raise ValueError("three distinct negative commands required; no parent skipping")
        representatives = [
            min(indices, key=lambda j: stable("representative", rows[j]["source_id"]))
            for indices in candidates.values()
        ]

        def distance(j, anchor=signatures[i]):
            return sum(a != b for a, b in zip(anchor, signatures[j], strict=True))

        ranked = sorted(
            representatives,
            key=lambda j: (
                distance(j) if policy == "hard" else 0,
                stable("negative", row["source_id"], rows[j]["source_id"]),
            ),
        )
        picked = ranked[:3]
        distances.update(distance(j) for j in picked)
        quartet = sorted(
            [i, *picked], key=lambda j: stable("order", row["source_id"], rows[j]["source_id"])
        )
        catalogs.append(quartet)
    result = np.asarray(catalogs, dtype=np.int64)
    unique_sets = len({tuple(sorted(q)) for q in catalogs})
    return result, dict(
        anchors=len(rows),
        queries=len(rows) * 4,
        policy=policy,
        unique_parents=len(set(result.ravel().tolist())),
        unique_unordered_quartets=unique_sets,
        unique_query_contexts=unique_sets * 4,
        repeated_query_contexts=(len(rows) - unique_sets) * 4,
        audio_presentations=len(rows) * 4,
        video_presentations=len(rows) * 16,
        negative_slot_distance_counts=dict(sorted(distances.items())),
        speaker_groups=len({r["speaker_group"] for r in rows}),
        speaker_condition_buckets=len(groups),
        condition_anchor_counts={c: sum(r["condition"] == c for r in rows) for c in ["p", "l"]},
    )


def query_indices(quartets):
    q = np.asarray(quartets)
    if (
        q.ndim != 2
        or not len(q)
        or q.shape[1] != 4
        or not np.issubdtype(q.dtype, np.integer)
        or (q < 0).any()
        or any(len(set(r.tolist())) != 4 for r in q)
    ):
        raise ValueError("distinct nonnegative four-video catalogs required")
    return q.ravel(), np.repeat(q, 4, axis=0), np.tile(np.arange(4), len(q))


def native_logps(logits, temperatures):
    value, temps = np.asarray(logits, np.float64), np.asarray(temperatures, np.float64)
    if (
        value.ndim != 2
        or not len(value)
        or value.shape[1] != 51
        or temps.shape != (6,)
        or not np.isfinite(value).all()
        or not np.isfinite(temps).all()
        or (temps <= 0).any()
    ):
        raise ValueError("finite full51logit rows and six frozen temperatures required")
    blocks = []
    for t, start, end in zip(temps, BOUNDS[:-1], BOUNDS[1:], strict=True):
        z = value[:, start:end] / t
        z = z - z.max(1, keepdims=True)
        blocks.append(z - np.log(np.exp(z).sum(1, keepdims=True)))
    return np.concatenate(blocks, 1)


def agreement_scores(audio_logits, candidate_video_logits, audio_temperatures, video_temperatures):
    audio = native_logps(audio_logits, audio_temperatures)
    given = np.asarray(candidate_video_logits)
    if given.shape != (len(audio), 4, 51):
        raise ValueError("four candidate video feature rows per audio query required")
    video = native_logps(given.reshape(-1, 51), video_temperatures).reshape(len(audio), 4, 51)
    score = np.zeros((len(audio), 4), np.float64)
    for start, end in zip(BOUNDS[:-1], BOUNDS[1:], strict=True):
        joint = audio[:, start:end][:, None, :] + video[:, :, start:end]
        maximum = joint.max(-1)
        score += maximum + np.log(np.exp(joint - maximum[:, :, None]).sum(-1))
    if not np.isfinite(score).all():
        raise ValueError("finite full-option agreement required")
    return score


def choice_logps(scores, temperature=1):
    z = np.asarray(scores, np.float64)
    if z.ndim != 2 or not len(z) or z.shape[1] != 4 or not np.isfinite(z).all():
        raise ValueError("finite four-choice scores required")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("positive finite query temperature required")
    z = z / temperature
    z -= z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def choice_probabilities(scores, temperature=1):
    return np.exp(choice_logps(scores, temperature))


def choice_metrics(probabilities, targets, log_probabilities=None):
    p, y = np.asarray(probabilities, np.float64), np.asarray(targets)
    if (
        p.ndim != 2
        or not len(p)
        or p.shape[1] != 4
        or y.shape != (len(p),)
        or not np.issubdtype(y.dtype, np.integer)
        or (y < 0).any()
        or (y >= 4).any()
        or not np.isfinite(p).all()
        or (p < 0).any()
        or not np.allclose(p.sum(1), 1, atol=1e-12, rtol=0)
    ):
        raise ValueError("valid four-choice probabilities and targets required")
    if log_probabilities is None:
        if (p[np.arange(len(p)), y] <= 0).any():
            raise ValueError("underflowed target probability requires exact log-domain scores")
        with np.errstate(divide="ignore"):
            logp = np.log(p)
    else:
        logp = np.asarray(log_probabilities, np.float64)
        if (
            logp.shape != p.shape
            or not np.isfinite(logp).all()
            or not np.array_equal(np.exp(logp), p)
        ):
            raise ValueError("finite probability-consistent log scores required")
    good, confidence = p.argmax(1) == y, p.max(1)
    bins = np.minimum((confidence * 15).astype(int), 14)
    brier = p.copy()
    brier[np.arange(len(p)), y] -= 1
    return dict(
        count=len(p),
        accuracy=float(good.mean()),
        nll=float(-logp[np.arange(len(p)), y].mean()),
        brier=float(np.square(brier).sum(1).mean()),
        ece=float(
            sum(abs((good[bins == b] - confidence[bins == b]).sum()) / len(p) for b in range(15))
        ),
        mean_confidence=float(confidence.mean()),
        class_counts=np.bincount(y, minlength=4).tolist(),
    )


def select_query_temperature(scores, targets):
    grid = np.geomspace(0.25, 4, 17)
    losses = [
        choice_metrics(choice_probabilities(scores, t), targets, choice_logps(scores, t))["nll"]
        for t in grid
    ]
    return float(grid[int(np.argmin(losses))]), dict(
        grid=grid.tolist(), nll=losses, rows=len(scores)
    )


def choice_json(scores, option_ids, temperature):
    z = np.asarray(scores)
    if (
        z.shape != (4,)
        or len(option_ids) != 4
        or len(set(option_ids)) != 4
        or any(not isinstance(i, str) or not i for i in option_ids)
    ):
        raise ValueError("four unique opaque option IDs and scores required")
    order = sorted(range(4), key=lambda i: option_ids[i])
    ids = [option_ids[i] for i in order]
    p = choice_probabilities(z[order][None], temperature)[0]
    return json.dumps(
        dict(
            selected_option_id=ids[int(p.argmax())],
            probabilities=dict(zip(ids, p.tolist(), strict=True)),
        ),
        sort_keys=True,
    )


def bootstrap_minus_uniform(probabilities, targets, speaker_groups, log_probabilities=None):
    p, y = np.asarray(probabilities), np.asarray(targets)
    metric = choice_metrics(p, y, log_probabilities)
    if len(speaker_groups) != len(p) or any(
        not isinstance(s, str) or not s for s in speaker_groups
    ):
        raise ValueError("whole speaker identities required")
    group_ids = sorted(set(speaker_groups))
    stats = []
    for index, values in enumerate([np.full_like(p, 0.25), p]):
        good, confidence = values.argmax(1) == y, values.max(1)
        loss = (
            -np.asarray(log_probabilities)[np.arange(len(y)), y]
            if index == 1 and log_probabilities is not None
            else -np.log(values[np.arange(len(y)), y])
        )
        b = values.copy()
        b[np.arange(len(y)), y] -= 1
        b = np.square(b).sum(1)
        bins = np.minimum((confidence * 15).astype(int), 14)
        current = []
        for group in group_ids:
            indices = np.array([i for i, s in enumerate(speaker_groups) if s == group])
            current.append(
                [
                    len(indices),
                    good[indices].sum(),
                    loss[indices].sum(),
                    b[indices].sum(),
                    confidence[indices].sum(),
                    *[
                        (
                            good[indices][bins[indices] == j]
                            - confidence[indices][bins[indices] == j]
                        ).sum()
                        for j in range(15)
                    ],
                ]
            )
        stats.append(np.asarray(current, np.float64))
    rng = np.random.default_rng(17)
    draws = rng.integers(0, len(group_ids), (10000, len(group_ids)))
    weights = np.stack([(draws == j).sum(1) for j in range(len(group_ids))], 1)
    aggregates = []
    points = []
    for s in stats:
        sample = weights @ s
        aggregates.append(
            np.column_stack(
                [sample[:, 1:5] / sample[:, :1], abs(sample[:, 5:]).sum(1) / sample[:, 0]]
            )
        )
        full = s.sum(0)
        points.append(np.r_[full[1:5] / full[0], abs(full[5:]).sum() / full[0]])
    difference = aggregates[1] - aggregates[0]
    intervals = np.quantile(difference, [0.025, 0.975], axis=0, method="linear")
    return dict(
        groups=len(group_ids),
        group_ids=group_ids,
        observations=metric["count"],
        seed=17,
        replicates=10000,
        method="whole-speaker paired percentile95%, linear quantiles",
        differences={
            key: dict(
                point=float(points[1][j] - points[0][j]),
                lower=float(intervals[0, j]),
                upper=float(intervals[1, j]),
            )
            for j, key in enumerate(["accuracy", "nll", "brier", "mean_confidence", "ece"])
        },
        scope="Observed development, repeated correlated quartet roles; not blind audit",
    )
