"""Deterministic TRAIN-only group splits for native text state holdouts."""

import hashlib
from collections import Counter, defaultdict

SOURCE = "text:condition_relevance_v1"
SPLIT_ID = "native-text-state-group-holdout-v1"


def build_split(
    rows,
    previously_consumed_ids,
    *,
    seed=17,
    holdout_fraction=0.5,
    source=SOURCE,
    split_id=SPLIT_ID,
):
    """Hold out half of groups never drawn by the prior common TRAIN sampler.

    The candidate vocabulary and prompt templates recur by design. Group IDs
    identify the underlying text state, so every row for a held state is kept
    out of fitting. This is state-group generalization, not lexical transfer.
    """
    if type(seed) is not int or seed < 0:
        raise ValueError("nonnegative integer seed required")
    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout fraction must be strictly between zero and one")
    if not isinstance(source, str) or not source or not isinstance(split_id, str) or not split_id:
        raise ValueError("nonempty source and split ID required")
    rows = [row for row in rows if row.get("source") == source]
    if not rows or any(row.get("split") != "train" for row in rows):
        raise ValueError("nonempty target TRAIN rows required")
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("TRAIN row IDs must be unique")
    row_by_id = {row["id"]: row for row in rows}
    consumed = set(previously_consumed_ids)
    if not consumed.issubset(row_by_id):
        raise ValueError("previously consumed IDs must belong to this source's TRAIN rows")
    groups = defaultdict(list)
    for row in rows:
        if not row.get("group") or not row.get("observation"):
            raise ValueError("stable state group and observation identities required")
        groups[row["group"]].append(row)
    previously_seen_groups = {row_by_id[sample_id]["group"] for sample_id in consumed}
    untouched_groups = sorted(set(groups) - previously_seen_groups)
    count = int(len(untouched_groups) * holdout_fraction)
    if count < 2 or count >= len(untouched_groups):
        raise ValueError("insufficient untouched groups for a nontrivial holdout")
    ranked = sorted(
        untouched_groups,
        key=lambda group: hashlib.sha256(f"{seed}:{split_id}:{group}".encode()).digest(),
    )
    holdout_groups = set(ranked[:count])
    fit = [row for row in rows if row["group"] not in holdout_groups]
    holdout = [row for row in rows if row["group"] in holdout_groups]
    _assert_disjoint(fit, holdout, previously_seen_groups)
    fit_tasks = Counter(row["task"] for row in fit)
    holdout_tasks = Counter(row["task"] for row in holdout)
    if set(fit_tasks) != set(holdout_tasks) or min(holdout_tasks.values()) < 2:
        raise ValueError("every task needs at least two held groups")
    return {
        "source": source,
        "split_id": split_id,
        "seed": seed,
        "holdout_fraction_of_previously_untouched_groups": holdout_fraction,
        "fit_rows": fit,
        "holdout_rows": holdout,
        "fit_groups": sorted({row["group"] for row in fit}),
        "holdout_groups": sorted(holdout_groups),
        "previously_seen_groups": sorted(previously_seen_groups),
        "previously_untouched_groups": untouched_groups,
        "fit_task_counts": dict(sorted(fit_tasks.items())),
        "holdout_task_counts": dict(sorted(holdout_tasks.items())),
    }


def _assert_disjoint(fit, holdout, previously_seen_groups):
    fit_ids = {row["id"] for row in fit}
    holdout_ids = {row["id"] for row in holdout}
    fit_groups = {row["group"] for row in fit}
    holdout_groups = {row["group"] for row in holdout}
    fit_observations = {row["observation"] for row in fit}
    holdout_observations = {row["observation"] for row in holdout}
    if fit_ids & holdout_ids or fit_groups & holdout_groups:
        raise ValueError("fit and holdout IDs/groups overlap")
    if fit_observations & holdout_observations:
        raise ValueError("fit and holdout text-state observations overlap")
    if holdout_groups & previously_seen_groups:
        raise ValueError("a previously consumed state group entered holdout")


def balanced_order(rows, steps, *, seed=17, split_id=SPLIT_ID):
    """Round-robin task strata with deterministic within-task permutations."""
    import random

    if type(steps) is not int or steps < 1 or not rows:
        raise ValueError("positive steps and nonempty fit rows required")
    pools = defaultdict(list)
    for index, row in enumerate(rows):
        pools[row["task"]].append(index)
    tasks = sorted(pools)
    cursors = Counter()
    rngs = {}
    for task, pool in pools.items():
        digest = hashlib.sha256(f"{seed}:{split_id}:{task}".encode()).digest()
        rngs[task] = random.Random(int.from_bytes(digest[:8], "big"))
        rngs[task].shuffle(pool)
    order = []
    for step in range(steps):
        batch = []
        for microbatch in range(4):
            task = tasks[(step * 4 + microbatch) % len(tasks)]
            if cursors[task] == len(pools[task]):
                rngs[task].shuffle(pools[task])
                cursors[task] = 0
            batch.append(pools[task][cursors[task]])
            cursors[task] += 1
        order.append(batch)
    return order


def counterfactual_donor_indices(rows, *, seed=17, split_id=SPLIT_ID):
    """Map each held row to another same-task group whose answer differs."""
    by_task = defaultdict(list)
    for index, row in enumerate(rows):
        by_task[row["task"]].append((index, row))
    donors = {}
    for task, entries in by_task.items():
        for index, row in entries:
            target_id = row["option_ids"][row["target_index"]]
            candidates = [
                (other_index, other)
                for other_index, other in entries
                if other["group"] != row["group"]
                and other["question"] == row["question"]
                and set(other["option_ids"]) == set(row["option_ids"])
                and other["option_ids"][other["target_index"]] != target_id
            ]
            if not candidates:
                raise ValueError(f"no different-label same-task state donor for {task}")
            digest = hashlib.sha256(f"{seed}:{split_id}:donor:{row['id']}".encode()).digest()
            selected = int.from_bytes(digest[:8], "big") % len(candidates)
            donors[index] = candidates[selected][0]
    if len(donors) != len(rows):
        raise ValueError("every held row must have exactly one counterfactual donor")
    return donors
