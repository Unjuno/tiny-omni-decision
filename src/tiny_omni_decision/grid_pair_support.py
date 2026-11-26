"""Exact Cartesian support counts without materializing billions of pairs."""

import numpy as np

from tiny_omni_decision.grid_role_supervision import pair_role_counts


def grid_pair_support(labels, windows, *, hidden, group, cover):
    if (
        not isinstance(labels, np.ndarray)
        or labels.ndim != 2
        or min(labels.shape) < 1
        or labels.dtype != np.int64
        or np.any(labels < -2)
        or not isinstance(windows, list)
        or not windows
    ):
        raise ValueError("nonempty integer native cell/time labels required")
    role_zero = pair_role_counts(
        labels.ravel(), np.empty((0, 2), np.int64), hidden=hidden, group=group, cover=cover
    )
    identities = group + [hidden]
    if not np.isin(labels[labels >= 0], identities).all():
        raise ValueError("unknown role namespace")
    seen = set()
    for w in windows:
        if (
            not isinstance(w, tuple)
            or len(w) != 3
            or any(type(i) is not int for i in w)
            or not 0 <= w[0] < w[1] < len(labels)
            or w[2] not in (0, 1, 2)
            or w in seen
        ):
            raise ValueError("unique forward-time windows required")
        seen.add(w)
    zero = dict(
        all_pairs=0,
        labelled_pairs=0,
        unknown_or_ambiguous_pairs=0,
        unknown_pairs=0,
        ambiguous_pairs=0,
        unknown_and_ambiguous_pairs=0,
    )
    ledger = dict(zero)
    strata = {str(k): dict(zero) for k in range(3)}
    roles = dict(role_zero)
    role_strata = {str(k): dict(role_zero) for k in range(3)}
    hist = [{i: int((row == i).sum()) for i in identities} for row in labels]
    unknown = (labels == -1).sum(1)
    ambiguous = (labels == -2).sum(1)
    n = labels.shape[1]
    for first, last, stratum in windows:
        a, b = hist[first], hist[last]
        ga = sum(a[i] for i in group)
        gb = sum(b[i] for i in group)
        ua, ub, aa, ab = map(
            int, (unknown[first], unknown[last], ambiguous[first], ambiguous[last])
        )
        eligible = sum(a.values()) * sum(b.values())
        value = dict(
            all_pairs=n * n,
            labelled_pairs=eligible,
            unknown_or_ambiguous_pairs=n * n - eligible,
            unknown_pairs=n * ua + n * ub - ua * ub,
            ambiguous_pairs=n * aa + n * ab - aa * ab,
            unknown_and_ambiguous_pairs=ua * ab + aa * ub,
        )
        same_group = sum(a[i] * b[i] for i in group)
        role = dict(
            total=eligible,
            same_hidden=a[hidden] * b[hidden],
            same_group=same_group,
            different_hidden_group=a[hidden] * gb + ga * b[hidden],
            different_group_group=ga * gb - same_group,
            same_cover=a[cover] * b[cover],
        )
        for key, v in value.items():
            ledger[key] += v
            strata[str(stratum)][key] += v
        for key, v in role.items():
            roles[key] += v
            role_strata[str(stratum)][key] += v
    ledger["strata"] = strata
    return dict(ledger=ledger, roles=roles, role_strata=role_strata)
