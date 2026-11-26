"""Matched parent order; view multiplicity never changes task weights."""

import random


def schedule(pool, views, previous_ids, arm):
    if arm not in ("control", "wording"):
        raise ValueError("unknown arm")
    lookup = {r["id"]: r for r in pool}
    if len(lookup) != len(pool):
        raise ValueError("duplicate parent")
    original = [lookup[k] for k in previous_ids if lookup[k]["source"] == "original"][:4096]
    groups = sorted({r["group"] for r in pool if r["source"] == "curriculum"})
    random.Random(17).shuffle(groups)
    chosen = set(groups[:128])
    parents = sorted([r for r in pool if r["source"] == "curriculum" and r["group"] in chosen],
                     key=lambda r: r["id"])
    random.Random(17).shuffle(parents)
    if len(original) != 4096 or len(parents) != 1024:
        raise ValueError("parent budget mismatch")
    view_lookup = {(r["parent_id"], r["view"]): r for r in views}
    if len(view_lookup) != len(views):
        raise ValueError("duplicate view")
    curriculum = []
    for cycle in range(4):
        for parent in parents:
            index = cycle if arm == "wording" and parent["task"] in (
                "extremum", "temporal_order") else 0
            row = view_lookup[parent["id"], index]
            for key in ("state", "options", "target", "group", "task"):
                if row[key] != parent[key]:
                    raise ValueError("view changed task")
            curriculum.append(row)
    result = []
    for i in range(0, 4096, 4):
        result.extend(original[i:i+4])
        result.extend(curriculum[i:i+4])
    return result
