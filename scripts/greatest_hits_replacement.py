"""Prospective label-blind input replacements and target-free sensitivity measures."""

from collections import Counter, defaultdict

import numpy as np

from scripts.greatest_hits_matched import digest
from scripts.semantic_state_diagnostic import log_probabilities


def build_inventory(rows):
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("complete unique validation identities required")
    buckets, definitions, owners = defaultdict(lambda: defaultdict(list)), {}, {}
    for r in rows:
        keys = ("id", "task", "question", "date_group", "parent", "asset_sha256", "event_id")
        if (
            r["split"] != "validation"
            or any(not isinstance(r[k], str) or not r[k] for k in keys)
            or not isinstance(r["options"], list)
            or not 2 <= len(r["options"]) <= 40
            or any(not isinstance(v, str) or not v for v in r["options"])
            or len(set(r["options"])) != len(r["options"])
        ):
            raise ValueError("explicit validation identity and complete options required")
        definition = r["question"], tuple(r["options"])
        if definitions.setdefault(r["task"], definition) != definition:
            raise ValueError("identical question/options required within each task")
        for key in (("parent", r["parent"]), ("asset", r["asset_sha256"])):
            if owners.setdefault(key, r["date_group"]) != r["date_group"]:
                raise ValueError("recording identity cannot cross dates")
        buckets[r["task"]][r["date_group"]].append(r["id"])
    for groups in buckets.values():
        if len(groups) < 2:
            raise ValueError("another validation date required for every task")
        for ids in groups.values():
            ids.sort()
    pairs = []
    by_id = {r["id"]: r for r in rows}
    for r in sorted(rows, key=lambda r: r["id"]):
        dates = sorted(d for d in buckets[r["task"]] if d != r["date_group"])
        date = dates[int(digest([17, "date", r["id"]]), 16) % len(dates)]
        eligible = buckets[r["task"]][date]
        donor_id = eligible[int(digest([17, "donor", r["id"]]), 16) % len(eligible)]
        donor = by_id[donor_id]
        if r["parent"] == donor["parent"] or r["asset_sha256"] == donor["asset_sha256"]:
            raise ValueError("cross-recording donor required; no redraw allowed")
        pairs.append(
            dict(
                id=r["id"],
                donor_id=donor_id,
                task=r["task"],
                date=r["date_group"],
                donor_date=date,
                parent=r["parent"],
                donor_parent=donor["parent"],
            )
        )
    usage = Counter(p["donor_id"] for p in pairs)
    return dict(
        seed=17,
        pairs=pairs,
        pairs_sha256=digest(pairs),
        views=["natural", "self_audio", "self_video", "donor_audio", "donor_video"],
        target_used_for_matching=False,
        counterfactual_truth_verified=False,
        original_questions=len(rows),
        unique_donor_questions=len(usage),
        max_donor_uses=max(usage.values()),
        donor_uses=dict(sorted(usage.items())),
        tasks=dict(sorted(Counter(p["task"] for p in pairs).items())),
        dates=dict(sorted(Counter(p["date"] for p in pairs).items())),
        donor_dates=dict(sorted(Counter(p["donor_date"] for p in pairs).items())),
    )


def inputs(store, original_id, donor_id, mode):
    if mode not in ("audio", "video"):
        raise ValueError("explicit one-stream replacement required")
    original = store.binding["questions"][original_id]
    donor = store.binding["questions"][donor_id]
    if any(original[k] != donor[k] for k in ("options", "query_keys")):
        raise ValueError("original query/options must agree with donor task")
    blocks, q = store.get(original_id)
    other, _ = store.get(donor_id)
    kinds = ["speech", "event_audio", "video"]
    if [b.kind for b in blocks] != kinds or [b.kind for b in other] != kinds:
        raise ValueError("complete ordered native audio/video blocks required")
    selected = other[:2] + blocks[2:] if mode == "audio" else blocks[:2] + other[2:]
    return selected, q


def sensitivity(original, replaced):
    a, b = np.asarray(original), np.asarray(replaced)
    if a.shape != b.shape or a.ndim != 2 or len(a) == 0 or a.shape[1] < 2:
        raise ValueError("aligned full paired score matrices required")
    lp, lq = log_probabilities(a), log_probabilities(b)
    if not np.isfinite(lp).all() or not np.isfinite(lq).all():
        raise ValueError("finite normalized logits required")
    p, q = np.exp(lp), np.exp(lq)
    middle = np.logaddexp(lp, lq) - np.log(2.0)
    js = np.maximum(
        0.0, 0.5 * (np.sum(p * (lp - middle), axis=1) + np.sum(q * (lq - middle), axis=1))
    )
    tv = 0.5 * np.abs(p - q).sum(axis=1)
    flip = a.argmax(1) != b.argmax(1)
    confidence = q.max(1) - p.max(1)
    return dict(
        count=len(a),
        prediction_flip_rate=float(flip.mean()),
        mean_total_variation=float(tv.mean()),
        mean_js_divergence=float(js.mean()),
        mean_confidence_change=float(confidence.mean()),
        mean_absolute_confidence_change=float(np.abs(confidence).mean()),
        per_example=dict(
            prediction_flip=flip.tolist(),
            total_variation=tv.tolist(),
            js_divergence=js.tolist(),
            confidence_change=confidence.tolist(),
        ),
    )
