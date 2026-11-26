"""Grouped development curriculum for irrelevant and necessary context."""

import copy

from scripts.build_conditional_curriculum import build, verify


def context_rows(seed, states, split):
    if split not in {"train", "development"}:
        raise ValueError("unsupported split")
    result = []
    for original in build(seed, states, split):
        if original["task"] not in {"extremum", "temporal_order"}:
            continue
        field = original["oracle"]["field"]
        direction = original["oracle"]["direction"]
        for kind in ("plain", "irrelevant", "necessary"):
            row = copy.deepcopy(original)
            if kind == "irrelevant":
                prefix = (
                    "Values for cost, risk, and time are listed in the state. "
                    if split == "train"
                    else "Each record contains cost, risk, and time measurements. "
                )
                row["question"] = prefix + row["question"]
            elif kind == "necessary":
                prefix = (
                    f"Compare the {field} values. "
                    if split == "train"
                    else f"The relevant attribute is {field}. "
                )
                row["question"] = prefix + f"Return the item with the {direction} value."
            row.update(base_id=original["id"], id=kind + ":" + original["id"], context=kind)
            verify(row)
            result.append(row)
    return result
