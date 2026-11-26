"""Integrity and descriptive metrics for the predeclared transfer factorial."""

import numpy as np

from scripts.condition_relevance_corpus import validate_parts
from scripts.semantic_state_diagnostic import metrics

CONDITIONS = {
    f"{query}_query_{options}_options" for query in ("seen", "new") for options in ("seen", "new")
}


def validate_transfer_rows(rows, payload_contract, development_states):
    """Validate every fixed row, complete state group and prior development exclusion."""
    expected_rows, expected_states = payload_contract["rows"], payload_contract["states"]
    if len(rows) != expected_rows or expected_rows != 64 * expected_states:
        raise ValueError("complete fixed transfer row count required")
    if any(len(r["options"]) != 40 or r["source"] != "condition_relevance_v1" for r in rows):
        raise ValueError("all40 candidates and original source required")
    result = validate_parts({"transfer": rows}, development_states)
    partition = result["partitions"]["transfer"]
    if result["unique_states"] != expected_states or partition["views"] != {
        k: expected_rows // 4 for k in CONDITIONS
    }:
        raise ValueError("complete four-cell state factorial required")
    return result


def transfer_metrics(rows, scores):
    """Report counts for actual syntax slices; do not assume shared seen-query indices."""
    scores = np.asarray(scores)
    if (
        not rows
        or scores.shape != (len(rows), 40)
        or not np.isfinite(scores).all()
        or any(len(r["options"]) != 40 for r in rows)
        or {r["condition"] for r in rows} != CONDITIONS
    ):
        raise ValueError("complete aligned finite40-candidate scores required")
    target = np.asarray([r["options"].index(r["target"]) for r in rows])

    def summarize(key):
        groups = {}
        for i, row in enumerate(rows):
            groups.setdefault(key(row), []).append(i)
        return {
            name: metrics(scores[index], target[index]) for name, index in sorted(groups.items())
        }

    fields = summarize(lambda r: f"{r['condition']}:{r['field']}")
    return dict(
        overall=metrics(scores, target),
        conditions=summarize(lambda r: r["condition"]),
        condition_fields=fields,
        actual_query_slices=summarize(
            lambda r: f"{r['condition']}:{r['field']}:q{r['query_index']}:{r['option_form']}"
        ),
        macro_condition_field={
            metric: float(np.mean([m[metric] for m in fields.values()]))
            for metric in ("accuracy", "nll", "brier", "ece")
        },
        minimum_condition_field_accuracy=min(m["accuracy"] for m in fields.values()),
        scope="authored development transfer; no calibration or checkpoint reselection",
    )
