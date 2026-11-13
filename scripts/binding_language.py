"""Fixed instruction-diversity intervention, with separate held-out syntax."""

import hashlib
import json
import random

from scripts.condition_binding_corpus import all_states, identity, ordered, questions_for
from scripts.semantic_state_diagnostic import FIELDS, QUESTIONS

CONDITIONS = (
    "known",
    "focus",
    "complement",
    "focus_descriptions",
    "complement_descriptions",
    "reverse_focus_descriptions",
    "no_question",
)
DESCRIPTION = {
    "color": "Object coloration: {value}.",
    "texture": "Surface texture category: {value}.",
    "count": "The total number of items is {value}.",
    "shape": "Geometry classification: {value}.",
}


def attributes(group, label):
    values = list(FIELDS)
    rng = random.Random(int(hashlib.sha256(f"17:{label}:{group}".encode()).hexdigest(), 16))
    rng.shuffle(values)
    return values


def diverse_train(rows):
    if not rows or len(rows) % 4 or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("complete unique TRAIN state groups required")
    result, seen = [], set()
    for start in range(0, len(rows), 4):
        block = rows[start : start + 4]
        group = block[0]["group"]
        if (
            group in seen
            or {r["group"] for r in block} != {group}
            or [r["task"] for r in block] != list(FIELDS)
            or any(r["split"] != "train" or r["source"] != "binding" for r in block)
        ):
            raise ValueError("ordered unique four-field TRAIN binding groups required")
        seen.add(group)
        family = (start // 4) % 8
        for r in block:
            field = r["task"]
            state = json.loads(r["state"])
            if identity(state) != group or r["target_id"] != f"{field}:{state[field]}":
                raise ValueError("state/target mismatch")
            other = [v for v in attributes(group, field) if v != field]
            alternatives = (
                QUESTIONS[FIELDS.index(field)],
                f"Report the {field} of the objects.",
                f"The requested attribute is {field}. State its value.",
                f"Answer with the {field}, not the {other[0]}.",
                f"Do not report {', '.join(other)}. Report the unexcluded property.",
                f"Return {field}; withhold {', '.join(other)}.",
                f"Do not return {other[0]}; return {field} instead.",
                f"Among {', '.join(attributes(group, 'all'))}, the requested one is {field}.",
            )
            result.append(
                dict(
                    r,
                    id=f"language-v1:{r['id']}",
                    underlying_id=r["id"],
                    language_family=family,
                    question=alternatives[family],
                )
            )
    return result


def unused_states(excluded, count=256):
    blocked = {identity(s) for s in excluded}
    available = [s for s in all_states() if identity(s) not in blocked]
    if count < 1 or count > len(available):
        raise ValueError("insufficient untouched states")
    return ordered(available, "language-heldout-v1")[:count]


def heldout_conditions(states):
    if not states or len({identity(s) for s in states}) != len(states):
        raise ValueError("unique nonempty heldout states required")
    result = {name: [] for name in CONDITIONS}
    for state in states:
        for base in questions_for(state, "development_language_transfer"):
            field = base["task"]
            other = ", ".join(v for v in attributes(base["group"], field) if v != field)
            for condition in CONDITIONS:
                row = dict(base, id=f"language-heldout-v1:{base['group']}:{field}:{condition}")
                row["options"], row["option_ids"] = (
                    base["options"].copy(),
                    base["option_ids"].copy(),
                )
                if "focus" in condition:
                    row["question"] = (
                        f"Relevant: {field}. Irrelevant: {other}. Select the matching value."
                    )
                elif "complement" in condition:
                    row["question"] = (
                        "Among the four recorded properties, answer for the one "
                        f"outside this list: {other}. What is its value?"
                    )
                elif condition == "no_question":
                    row["question"] = ""
                if "descriptions" in condition or condition == "no_question":
                    row["options"] = [
                        DESCRIPTION[f].format(value=v)
                        for f, v in (key.split(":", 1) for key in row["option_ids"])
                    ]
                if condition.startswith("reverse_"):
                    row["options"].reverse()
                    row["option_ids"].reverse()
                row["target"] = row["options"][row["option_ids"].index(row["target_id"])]
                result[condition].append(row)
    return result


def quality_failures(candidate, control, intervals):
    failures = []
    for condition in ("focus_descriptions", "complement_descriptions"):
        current, old = candidate[condition], control[condition]
        m, b = current["overall"], old["overall"]
        checks = dict(
            minimum_task_accuracy=min(v["accuracy"] for v in current["tasks"].values()) >= 0.95,
            pair_accuracy=current["pair_both_correct"] >= 0.90,
            improvement=m["accuracy"] - b["accuracy"] >= 0.03,
            paired_lower=intervals[condition]["lower"][0] > 0,
            nll=m["nll"] < b["nll"],
            brier=m["brier"] <= b["brier"],
            ece=m["ece"] <= b["ece"] + 0.02,
        )
        failures.extend(f"{condition}:{key}" for key, ok in checks.items() if not ok)
    if candidate["known"]["overall"]["accuracy"] < 0.99:
        failures.append("known:accuracy_below_99pct")
    return failures
