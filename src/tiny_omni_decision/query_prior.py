"""Leave-group-out smoothed query-class prior for shortcut diagnosis only."""

from collections import Counter


def leave_group_out_prior(classes, groups, targets):
    if not classes or not len(classes) == len(groups) == len(targets):
        raise ValueError("nonempty aligned rows required")
    if any(type(y) is not int or y not in (0, 1) for y in targets):
        raise ValueError("binary integer targets required")
    totals, grouped = Counter(), Counter()
    for c, g, y in zip(classes, groups, targets, strict=True):
        totals[c, y] += 1
        grouped[c, g, y] += 1
    output = []
    for c, g in zip(classes, groups, strict=True):
        negative = totals[c, 0] - grouped[c, g, 0]
        positive = totals[c, 1] - grouped[c, g, 1]
        output.append(
            [(negative + 1) / (negative + positive + 2), (positive + 1) / (negative + positive + 2)]
        )
    return output
