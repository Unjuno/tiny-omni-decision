"""TRAIN-only teacher capability screen; oracle metadata never enters the prompt."""

import hashlib
import itertools
import json
from collections import Counter, defaultdict

import numpy as np

from scripts.build_conditional_curriculum import verify as verify_numeric
from scripts.condition_binding_corpus import identity
from scripts.semantic_state_diagnostic import FIELDS
from tiny_omni_decision.decision_math import (
    brier_score,
    expected_calibration_error,
    negative_log_likelihood,
)

LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmn"
BINDING = ("direct", "exclusion", "switched", "reverse_exclusion", "no_question")


def _groups(rows, expected):
    groups = defaultdict(list)
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("unique nonempty source required")
    for row in rows:
        if row["split"] != "train":
            raise ValueError("TRAIN only")
        groups[row["group"]].append(row)
    for block in groups.values():
        if Counter(r["task"] for r in block) != expected or len({r["state"] for r in block}) != 1:
            raise ValueError("complete consistent groups required")
    return groups


def build_probe(binding, numeric, count=32):
    bg = _groups(binding, dict.fromkeys(FIELDS, 1))
    ng = _groups(numeric, dict(extremum=4, temporal_order=2, count=1, conditional_selection=1))
    for row in binding:
        state = json.loads(row["state"])
        if (
            identity(state) != row["group"]
            or row["target_id"] != f"{row['task']}:{state[row['task']]}"
            or len(row["option_ids"]) != len(set(row["option_ids"]))
            or len(row["option_ids"]) != 40
            or row["options"][row["option_ids"].index(row["target_id"])] != row["target"]
        ):
            raise ValueError("binding oracle/options mismatch")
    for row in numeric:
        verify_numeric(row)
    if count < 1 or min(len(bg), len(ng)) < count:
        raise ValueError("insufficient TRAIN groups")

    def chosen(groups, source):
        return sorted(
            groups,
            key=lambda g: hashlib.sha256(
                f"17:semantic-teacher-v1:{source}:{g}".encode()
            ).hexdigest(),
        )[:count]

    result = []
    for group in chosen(bg, "binding"):
        for row in sorted(bg[group], key=lambda r: FIELDS.index(r["task"])):
            field = row["task"]
            other = [f for f in FIELDS if f != field]
            for condition in BINDING:
                question = {
                    "direct": f"Return the value of the {field} property.",
                    "exclusion": (
                        f"Do not report {', '.join(other)}. "
                        "Return the value of the sole remaining property."
                    ),
                    "switched": (
                        f"Initially {other[0]} was requested. "
                        f"That request is cancelled. Report {field} instead."
                    ),
                    "no_question": "",
                }
                q = question["exclusion" if condition == "reverse_exclusion" else condition]
                ids = row["option_ids"][:: -1 if condition == "reverse_exclusion" else 1]
                descriptions = [
                    f"This answer gives the object's {f}: {v}."
                    for f, v in (item.split(":", 1) for item in ids)
                ]
                result.append(
                    dict(
                        id=f"binding:{group}:{condition}:{field}",
                        source="binding",
                        group=group,
                        task=field,
                        condition=condition,
                        split="train",
                        state=row["state"],
                        question=q,
                        option_ids=ids,
                        options=descriptions,
                        target_id=row["target_id"],
                        target=descriptions[ids.index(row["target_id"])],
                    )
                )
    for group in chosen(ng, "numeric"):
        for row in sorted(ng[group], key=lambda r: r["id"]):
            for condition in ("original", "reverse"):
                opts = row["options"][:: -1 if condition == "reverse" else 1]
                result.append(
                    dict(
                        row,
                        id=f"numeric:{row['id']}:{condition}",
                        source="numeric",
                        condition=condition,
                        option_ids=opts,
                        options=opts,
                        target_id=row["target"],
                    )
                )
    return result


def input_text(row):
    if not 2 <= len(row["options"]) <= len(LABELS):
        raise ValueError("unsupported option count")
    choices = "\n".join(
        f"{label}. {option}"
        for label, option in zip(LABELS[: len(row["options"])], row["options"], strict=True)
    )
    return (
        f"State: {row['state']}\nQuestion: {row['question']}\nOptions:\n{choices}\n"
        "Choose the best option. Return only its case-sensitive option label, without explanation."
    )


def probabilities(logits):
    a = np.asarray(logits, dtype=np.float64)
    if a.ndim != 1 or len(a) < 2 or not np.isfinite(a).all():
        raise ValueError("finite option logits required")
    a = np.exp(a - a.max())
    return a / a.sum()


def summarize_rows(rows, outputs):
    if len(rows) != len(outputs) or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("complete unique aligned rows required")
    groups, by_id = defaultdict(list), {}
    for row, output in zip(rows, outputs, strict=True):
        if row["id"] != output["row_id"] or len(row["options"]) != len(output["logits"]):
            raise ValueError("output identity/options mismatch")
        p = probabilities(output["logits"])
        item = dict(
            row=row,
            p=p.tolist(),
            target=row["option_ids"].index(row["target_id"]),
            prediction=row["option_ids"][int(p.argmax())],
        )
        groups[f"{row['source']}/{row['condition']}"].append(item)
        by_id[row["id"]] = item

    def stats(items):
        p, t = [i["p"] for i in items], [i["target"] for i in items]
        return dict(
            count=len(items),
            accuracy=float(np.mean([np.argmax(a) == b for a, b in zip(p, t, strict=True)])),
            nll=negative_log_likelihood(p, t),
            brier=brier_score(p, t),
            ece=expected_calibration_error(p, t),
            mean_confidence=float(np.mean([max(v) for v in p])),
        )

    report = dict(conditions={}, reversal={})
    for name, items in groups.items():
        tasks = {
            task: stats([i for i in items if i["row"]["task"] == task])
            for task in sorted({i["row"]["task"] for i in items})
        }
        pair_correct = []
        for group in sorted({i["row"]["group"] for i in items}):
            for a, b in itertools.combinations([i for i in items if i["row"]["group"] == group], 2):
                if a["row"]["target_id"] != b["row"]["target_id"]:
                    pair_correct.append(
                        a["prediction"] == a["row"]["target_id"]
                        and b["prediction"] == b["row"]["target_id"]
                    )
        report["conditions"][name] = dict(
            overall=stats(items),
            tasks=tasks,
            macro={
                k: float(np.mean([m[k] for m in tasks.values()]))
                for k in ("accuracy", "nll", "brier", "ece")
            },
            pair_count=len(pair_correct),
            pair_both_correct=float(np.mean(pair_correct)) if pair_correct else None,
        )
    for source, original, _reverse in (
        ("binding", "exclusion", "reverse_exclusion"),
        ("numeric", "original", "reverse"),
    ):
        agreements, diffs = [], []
        for item in groups[f"{source}/{original}"]:
            row = item["row"]
            rid = (
                row["id"].replace(":exclusion:", ":reverse_exclusion:")
                if source == "binding"
                else row["id"].removesuffix(":original") + ":reverse"
            )
            other = by_id[rid]
            aligned = [
                other["p"][other["row"]["option_ids"].index(oid)] for oid in row["option_ids"]
            ]
            agreements.append(item["prediction"] == other["prediction"])
            diffs.append(float(np.max(np.abs(np.array(item["p"]) - aligned))))
        report["reversal"][source] = dict(
            count=len(agreements),
            answer_agreement=float(np.mean(agreements)),
            maximum_probability_difference=max(diffs),
        )
    return report
