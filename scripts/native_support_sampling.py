"""One TRAIN-only positive-support floor, preserving all other occurrence slots."""

import hashlib
import json
from collections import Counter, defaultdict

POLICY = "starss-positive-floor-half-unique-availability-capped-v1"


def repair_support(rows, order, seed=17):
    if (not rows or not order or any(len(b) != 8 for b in order)
            or any(type(i) is not int or not 0 <= i < len(rows) for b in order for i in b)
            or len({r["id"] for r in rows}) != len(rows)
            or any(r["split"] != "train" for r in rows)):
        raise ValueError("unique TRAIN-only corpus and complete eight-row batches required")
    used = Counter(i for batch in order for i in batch)
    available, consumed = defaultdict(list), defaultdict(list)
    for i, row in enumerate(rows):
        if row["source"] != "starss":
            continue
        if (row["options"] != ["no", "yes"] or type(row["target_index"]) is not int
                or row["target_index"] not in (0, 1)):
            raise ValueError("original STARSS binary target/order required")
        available[row["task"]].append(i)
        if i in used:
            consumed[row["task"]].append(i)

    def rank(i, purpose):
        return hashlib.sha256(f"{seed}:{POLICY}:{purpose}:{rows[i]['id']}".encode()).hexdigest()

    def record(i):
        return rows[i]["observation"].rsplit(":", 1)[0]

    def positive_info(indices):
        pos = [i for i in indices if rows[i]["target_index"] == 1]
        return dict(positive_unique=len(pos), positive_recordings=len({record(i) for i in pos}),
                    positive_rooms=len({rows[i]["group"] for i in pos}))

    replacements, mapping, events = [], {}, {}
    for event, original in sorted(consumed.items()):
        candidates = available[event]
        signatures = {(rows[i]["question"], tuple(rows[i]["options"]),
                       tuple(rows[i]["queries"])) for i in candidates}
        if len(signatures) != 1:
            raise ValueError("one exact condition/candidate/query signature per event required")
        if len({used[i] for i in original}) != 1:
            raise ValueError("uniform per-question repetition within event required")
        positives = [i for i in candidates if rows[i]["target_index"] == 1]
        old_positive = [i for i in original if rows[i]["target_index"] == 1]
        floor = min((len(original) + 1) // 2, len(positives))
        needed = max(0, floor - len(old_positive))
        # Keep every existing positive, including an already positive-heavy speech pool.
        remaining = set(positives) - used.keys()
        room_counts = Counter(rows[i]["group"] for i in old_positive)
        record_counts = Counter(record(i) for i in old_positive)
        added = []
        for _ in range(needed):
            chosen = min(remaining, key=lambda i: (room_counts[rows[i]["group"]],
                         record_counts[record(i)], rank(i, "positive")))
            remaining.remove(chosen)
            added.append(chosen)
            room_counts[rows[chosen]["group"]] += 1
            record_counts[record(chosen)] += 1
        removed = sorted((i for i in original if rows[i]["target_index"] == 0),
                         key=lambda i: rank(i, "negative-replacement"))[:needed]
        for old, new in zip(removed, added, strict=True):
            mapping[old] = new
            replacements.append(dict(old_index=old, new_index=new, old_id=rows[old]["id"],
                                     new_id=rows[new]["id"], event=event,
                                     presentations=used[old]))
        new_indices = [mapping.get(i, i) for i in original]
        events[event] = dict(unique_quota=len(original),
            available_positive_unique=len(positives),
            before_positive_unique=len(old_positive),
            target_positive_unique=floor,
            after_positive_unique=sum(rows[i]["target_index"] for i in new_indices),
            available_support=positive_info(positives), before_support=positive_info(original),
            after_support=positive_info(new_indices), replaced_unique=needed,
            status=("NO_AVAILABLE_TRAIN_POSITIVE" if not positives else
                    "ALREADY_MEETS_FLOOR" if not needed else "REPAIRED_FROM_UNUSED_TRAIN"))
    repaired = [[mapping.get(i, i) for i in batch] for batch in order]
    new_counts = Counter(i for batch in repaired for i in batch)
    if len(new_counts) != len(used) or any(new_counts[mapping.get(i, i)] != n
                                           for i, n in used.items()):
        raise ValueError("unique support or repetition multiplicity changed")
    for before, after in zip(order, repaired, strict=True):
        for i, j in zip(before, after, strict=True):
            if i != j and (rows[i]["source"] != "starss"
                           or rows[i]["task"] != rows[j]["task"]
                           or rows[i]["target_index"] != 0 or rows[j]["target_index"] != 1):
                raise ValueError("only within-event negative-to-positive substitutions allowed")
    value = dict(policy=POLICY, seed=seed, order=repaired, replacements=replacements,
                 events=events, unique_row_indices=sorted(new_counts),
                 parent_unique_row_indices=sorted(used),
                 replacement_presentations=sum(r["presentations"] for r in replacements))
    value["plan_sha256"] = hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return value
