"""Replace a second presentation with unused TRAIN questions at fixed task quotas."""

import hashlib
from collections import Counter, defaultdict, deque

from scripts.native_fit import accounting


def quota_key(row):
    return row["modality"], row["source"], row["task"], len(row["options"])


def make_pool(rows, original, policy, seed=17):
    if (
        not rows
        or not original
        or policy not in ("repeat", "fresh")
        or len({r["id"] for r in rows}) != len(rows)
        or any(r["split"] != "train" or not 2 <= len(r["options"]) <= 40 for r in rows)
        or any(type(i) is not int or not 0 <= i < len(rows) for i in original)
    ):
        raise ValueError("unique TRAIN rows, known occurrences and fixed policy required")
    used = set(original)
    buckets = defaultdict(list)
    for i, row in enumerate(rows):
        if i not in used:
            buckets[quota_key(row)].append(i)
    queues = {
        k: deque(
            sorted(
                v,
                key=lambda i: hashlib.sha256(f"{seed}:fresh:{rows[i]['id']}".encode()).hexdigest(),
            )
        )
        for k, v in buckets.items()
    }
    extra, counts = [], Counter({m: 0 for m in ("text", "image", "audio", "video")})
    repeats = 0
    for i in original:
        queue = queues.get(quota_key(rows[i]))
        if policy == "fresh" and queue:
            j = queue.popleft()
            counts[rows[j]["modality"]] += 1
        else:
            j = i
            repeats += 1
        extra.append(j)
    pool = original + extra
    if [quota_key(rows[i]) for i in original] != [quota_key(rows[i]) for i in extra]:
        raise ValueError("source/task/candidate cardinality sequence changed")
    return dict(
        policy=policy,
        seed=seed,
        pool=pool,
        new_unique_by_modality=dict(counts),
        extra_repeated_occurrences=repeats,
        accounting=accounting([rows[i] for i in pool]),
    )
