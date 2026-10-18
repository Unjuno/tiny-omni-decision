"""Label-independent, deterministic within-group observation substitution."""

import hashlib
from collections import defaultdict


def grouped_derangement(identities, groups, seed=17):
    """Return replacement indices; require unique episode IDs and groups of >=2.

    Hash-sort each group's identities and rotate once. This is a deterministic
    control, not a uniform sample from all possible derangements. No target
    labels enter the ordering, and all observations are used exactly once.
    """
    if not identities or len(identities) != len(groups):
        raise ValueError("aligned nonempty identities and groups required")
    if len(set(identities)) != len(identities):
        raise ValueError("unique episode identities required")
    members = defaultdict(list)
    for index, group in enumerate(groups):
        members[group].append(index)
    result = list(range(len(identities)))
    for indices in members.values():
        if len(indices) < 2:
            raise ValueError("cannot derange a singleton group")
        ordered = sorted(indices, key=lambda i: (
            hashlib.sha256(f"{seed}:{identities[i]}".encode()).hexdigest(), identities[i]
        ))
        for original, replacement in zip(ordered, ordered[1:] + ordered[:1], strict=True):
            result[original] = replacement
    return result
