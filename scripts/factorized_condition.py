"""Fixed representation/readout diagnostic; no gold-dependent inference routing."""

import json

import numpy as np

from scripts.condition_binding_corpus import all_states, identity, ordered, questions_for
from scripts.semantic_state_diagnostic import FIELDS, log_probabilities

FAMILIES = (
    "explicit",
    "irrelevant_first",
    "positive_then_not",
    "exclude",
    "other_than",
    "although",
    "synonym",
    "exclude_synonym",
)
SYNONYMS = dict(
    color="hue", texture="surface pattern", count="number of objects", shape="geometric form"
)


def select_states(excluded, count=32):
    blocked = {identity(s) for s in excluded}
    pool = ordered(
        [s for s in all_states() if identity(s) not in blocked], "factorized-representation-v1"
    )
    if count < 1 or len(pool) < count:
        raise ValueError("insufficient independent states")
    return pool[:count]


def make_rows(states):
    if not states or len({identity(s) for s in states}) != len(states):
        raise ValueError("unique nonempty states required")
    result = []
    for family in FAMILIES:
        for state in states:
            for base in questions_for(state, "development"):
                field = base["task"]
                others = ", ".join(f for f in FIELDS if f != field)
                synonym = SYNONYMS[field]
                templates = (
                    f"Give the value of {field}.",
                    f"For this request, {others} are irrelevant; report {field}.",
                    f"Report {field}, not {others}.",
                    f"Do not report {others}. Report the remaining property.",
                    f"I need the entry other than {others}.",
                    f"Although {others} appear in the record, answer about {field}.",
                    f"Answer about the {synonym}.",
                    f"Ignore values for {others}. My question concerns the {synonym}.",
                )
                options = [
                    f"The {f} is {v}." for f, v in (oid.split(":", 1) for oid in base["option_ids"])
                ]
                result.append(
                    dict(
                        base,
                        id=f"factor-v1:{family}:{base['group']}:{field}",
                        family=family,
                        question=templates[FAMILIES.index(family)],
                        options=options,
                        target=options[base["option_ids"].index(base["target_id"])],
                    )
                )
    return result


def render(row, fields=FIELDS):
    """Read only public state, question, supplied candidate texts and schema names."""
    state = json.dumps(json.loads(row["state"]), sort_keys=True, separators=(",", ":"))
    return dict(
        direct=[
            (
                f"Record: {state}\nRequest: {row['question']}",
                f"The correct response to the request is: {option}",
            )
            for option in row["options"]
        ],
        routing=[
            (f"Request: {row['question']}", f"The requested property is {field}.")
            for field in fields
        ],
        values=[(f"Record: {state}", option) for option in row["options"]],
    )


def combine(route, values, fields, option_ids):
    """Return log p(field|request) + log p(value|record,field), all candidates kept."""
    route, values = np.asarray(route, dtype=np.float64), np.asarray(values, dtype=np.float64)
    if (
        route.shape != (len(fields),)
        or values.shape != (len(option_ids),)
        or not len(fields)
        or len(set(fields)) != len(fields)
        or len(set(option_ids)) != len(option_ids)
        or not np.isfinite(route).all()
        or not np.isfinite(values).all()
    ):
        raise ValueError("invalid finite factor scores or identities")
    groups = [oid.split(":", 1)[0] for oid in option_ids]
    if set(groups) != set(fields):
        raise ValueError("incomplete schema coverage")
    route_lp = route - np.logaddexp.reduce(route)
    result = np.empty(len(values), dtype=np.float64)
    for i, field in enumerate(fields):
        indices = np.array([j for j, f in enumerate(groups) if f == field])
        result[indices] = route_lp[i] + values[indices] - np.logaddexp.reduce(values[indices])
    return result


def diagnostic_summary(rows, scores):
    from scripts.condition_binding_training import score_summary

    if scores.shape != (len(rows), 40):
        raise ValueError("all forty options required")
    answer = score_summary(rows, scores)
    target_fields = np.array([FIELDS.index(r["task"]) for r in rows])
    predicted_fields = np.array(
        [
            FIELDS.index(rows[i]["option_ids"][p].split(":")[0])
            for i, p in enumerate(scores.argmax(-1))
        ]
    )
    answer["routing_accuracy"] = float((predicted_fields == target_fields).mean())
    answer["families"] = {}
    for family in FAMILIES:
        ids = [i for i, r in enumerate(rows) if r["family"] == family]
        if ids:
            answer["families"][family] = score_summary([rows[i] for i in ids], scores[ids])
    answer["probability_sum_error"] = float(abs(np.exp(log_probabilities(scores)).sum(1) - 1).max())
    return answer


def control_rows(rows, condition):
    from copy import deepcopy

    if condition not in ("primary", "reverse", "donor", "no_question"):
        raise ValueError("unknown control")
    result = deepcopy(rows)
    states = dict((r["group"], r["state"]) for r in rows)
    keys = list(states)
    if len(keys) < 2:
        raise ValueError("at least two distinct states required")
    donor = {key: states[keys[(i + 1) % len(keys)]] for i, key in enumerate(keys)}
    for row in result:
        if condition == "reverse":
            row["options"].reverse()
            row["option_ids"].reverse()
        elif condition == "no_question":
            row["question"] = ""
        elif condition == "donor":
            row["state"] = donor[row["group"]]
            state = json.loads(row["state"])
            row["target_id"] = f"{row['task']}:{state[row['task']]}"
            row["target"] = row["options"][row["option_ids"].index(row["target_id"])]
    return result


def research_failures(direct, factored, interval):
    old, new = direct["overall"], factored["overall"]
    failures = []
    if new["accuracy"] - old["accuracy"] < 0.03 - 1e-12:
        failures.append("joint_accuracy_improvement")
    if interval["lower"][0] <= 0:
        failures.append("paired_accuracy_interval")
    if min(v["overall"]["accuracy"] for v in factored["families"].values()) < 0.9:
        failures.append("minimum_family_accuracy")
    if any(
        v["overall"]["accuracy"] < direct["families"][k]["overall"]["accuracy"] - 0.01 - 1e-12
        for k, v in factored["families"].items()
    ):
        failures.append("family_retention")
    for name, limit in (("nll", 0), ("brier", 0), ("ece", 0.02)):
        if new[name] > old[name] + limit + 1e-12:
            failures.append(name)
    return failures


def order_control(original, reverse, original_ids, reverse_ids):
    original, reverse = np.asarray(original), np.asarray(reverse)
    if original.shape != reverse.shape or len(original) != len(original_ids):
        raise ValueError("aligned order comparison required")
    aligned, agreement = [], []
    for a, b, ids, swapped in zip(original, reverse, original_ids, reverse_ids, strict=True):
        if len(set(ids)) != len(ids) or set(ids) != set(swapped) or len(ids) != len(a):
            raise ValueError("same unique candidate identities required")
        aligned.append([b[swapped.index(identity)] for identity in ids])
        agreement.append(ids[int(a.argmax())] == swapped[int(b.argmax())])
    return dict(
        max_probability_delta=float(abs(original - np.array(aligned)).max()),
        identity_agreement=float(np.mean(agreement)),
    )


def replay_equal(left, right):
    """Exact discrete results, finite float outputs within the frozen absolute tolerance."""
    if isinstance(left, dict):
        return (
            isinstance(right, dict)
            and left.keys() == right.keys()
            and all(replay_equal(left[k], right[k]) for k in left)
        )
    if isinstance(left, list):
        return (
            isinstance(right, list)
            and len(left) == len(right)
            and all(replay_equal(a, b) for a, b in zip(left, right, strict=True))
        )
    if isinstance(left, float):
        return (
            isinstance(right, float)
            and np.isfinite(left)
            and np.isfinite(right)
            and abs(left - right) <= 1e-6
        )
    return type(left) is type(right) and left == right
