"""Fixed occurrence-preserving batch control; labels affect grouping, never inputs."""

import hashlib
import json
import math
from collections import Counter, defaultdict


def _hash(value, seed):
    return hashlib.sha256(
        json.dumps([seed, value], ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def _condition(row):
    return (
        row["modality"],
        row["source"],
        row["task"],
        row["question"],
        tuple(sorted(row["options"])),
    )


def eligible(a, b):
    return (
        _condition(a) == _condition(b)
        and a["observation"] != b["observation"]
        and a["options"][a["target_index"]] != b["options"][b["target_index"]]
    )


def build_plan(rows, pool, policy, seed=17):
    """Every pool occurrence is used exactly once, including unpairable examples.

    A pool entry is a corpus row index. Its position is its occurrence identity;
    repeated rows retain separate identities. No maximally paired claim is made:
    the fixed hash-ordered greedy algorithm preserves leftovers as fallback pairs.
    """
    if (
        policy not in ("independent", "matched")
        or not rows
        or not pool
        or len({r["id"] for r in rows}) != len(rows)
        or any(type(i) is not int or not 0 <= i < len(rows) for i in pool)
        or any(
            r["split"] != "train"
            or not 2 <= len(r["options"]) <= 40
            or len(set(r["options"])) != len(r["options"])
            or len(r["queries"]) != len(r["options"])
            or not 0 <= r["target_index"] < len(r["options"])
            for r in rows
        )
    ):
        raise ValueError("valid unique TRAIN rows, known pool indices and policy required")
    by_modality = defaultdict(list)
    query_identity = {}
    for occurrence, index in enumerate(pool):
        row = rows[index]
        by_modality[row["modality"]].append(occurrence)
        signature = _condition(row)
        query = dict(zip(row["options"], row["queries"], strict=True))
        if query_identity.setdefault(signature, query) != query:
            raise ValueError("same condition must use identical query features")
    modalities = ("text", "image", "audio", "video")
    if (
        set(by_modality) != set(modalities)
        or len({len(v) for v in by_modality.values()}) != 1
        or len(pool) % 8
    ):
        raise ValueError("equal even counts for all four modalities required")
    pairs_by_modality, coverage = {}, {}
    for modality in modalities:
        ordered = sorted(by_modality[modality], key=lambda o: _hash(["occurrence", o], seed))
        pairs, leftovers = [], []
        if policy == "matched":
            groups = defaultdict(list)
            for o in ordered:
                groups[_condition(rows[pool[o]])].append(o)
            for signature in sorted(groups):
                remaining = list(groups[signature])
                while remaining:
                    a = remaining.pop(0)
                    index = next(
                        (
                            i
                            for i, b in enumerate(remaining)
                            if eligible(rows[pool[a]], rows[pool[b]])
                        ),
                        None,
                    )
                    if index is None:
                        leftovers.append(a)
                    else:
                        pairs.append([a, remaining.pop(index)])
        else:
            leftovers = ordered
        leftovers.sort(key=lambda o: _hash(["fallback", o], seed))
        pairs.extend([leftovers[i : i + 2] for i in range(0, len(leftovers), 2)])
        if any(len(p) != 2 for p in pairs):
            raise ValueError("odd fallback count")
        pairs.sort(key=lambda p: _hash(["pair-order", p], seed))
        pairs_by_modality[modality] = pairs
        matched = sum(eligible(rows[pool[a]], rows[pool[b]]) for a, b in pairs)
        coverage[modality] = dict(
            pairs=len(pairs), matched_pairs=matched, fallback_pairs=len(pairs) - matched
        )
    batches = [
        [o for modality in modalities for o in pairs_by_modality[modality][step]]
        for step in range(len(pool) // 8)
    ]
    if Counter(o for batch in batches for o in batch) != Counter(range(len(pool))):
        raise ValueError("occurrence loss or duplication")
    return dict(
        policy=policy,
        seed=seed,
        occurrences=batches,
        coverage=coverage,
        matched_pairs=sum(c["matched_pairs"] for c in coverage.values()),
        fallback_pairs=sum(c["fallback_pairs"] for c in coverage.values()),
        occurrence_multiset_sha256=_hash(sorted(pool), seed),
        occurrence_order_sha256=_hash(batches, seed),
    )


def learning_rate(step):
    if type(step) is not int or not 1 <= step <= 1024:
        raise ValueError("fixed1024-step matched-batch schedule required")
    if step <= 31:
        return 0.001 * step / 31
    return 0.001 * 0.5 * (1 + math.cos(math.pi * (step - 31) / (1024 - 31)))
