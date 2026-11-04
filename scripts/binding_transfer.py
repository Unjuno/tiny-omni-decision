"""Predeclared, state-disjoint conditional transfer; no template search."""

import numpy as np

from scripts.condition_binding_corpus import all_states, identity, ordered, questions_for
from scripts.semantic_state_diagnostic import FIELDS, log_probabilities

CONDITIONS = (
    "known",
    "question",
    "options",
    "joint",
    "distractor",
    "distractor_options",
    "no_question",
    "reverse_joint",
)
PARAPHRASES = (
    "Which offered description matches the objects' coloration?",
    "Select the description of the objects' surface finish.",
    "Which option gives the scene's object total?",
    "Select the description of the objects' solid geometry.",
)
DESCRIPTION = {
    "color": "The objects look {value}.",
    "texture": "The surface material or pattern is {value}.",
    "count": "The scene contains exactly {value} objects.",
    "shape": "Each object has the {value} form.",
}


def select_states(excluded, count=256):
    blocked = {identity(s) for s in excluded}
    available = [s for s in all_states() if identity(s) not in blocked]
    if count < 1 or len(available) < count:
        raise ValueError("insufficient untouched state combinations")
    return ordered(available, "independent-transfer-v1")[:count]


def build_conditions(states):
    if not states or len({identity(s) for s in states}) != len(states):
        raise ValueError("unique nonempty states required")
    result = {name: [] for name in CONDITIONS}
    for state in states:
        for index, base in enumerate(questions_for(state, "development_transfer")):
            field = FIELDS[index]
            distractors = ", ".join(f for f in FIELDS if f != field)
            for condition in CONDITIONS:
                row = dict(base, id=f"transfer-v1:{base['group']}:{field}:{condition}")
                row["options"], row["option_ids"] = (
                    base["options"].copy(),
                    base["option_ids"].copy(),
                )
                if condition in ("question", "joint", "reverse_joint"):
                    row["question"] = PARAPHRASES[index]
                elif condition in ("distractor", "distractor_options"):
                    row["question"] = f"Ignore {distractors}. Give only the {field} of the objects."
                elif condition == "no_question":
                    row["question"] = ""
                if condition in (
                    "options",
                    "joint",
                    "distractor_options",
                    "no_question",
                    "reverse_joint",
                ):
                    row["options"] = [
                        DESCRIPTION[f].format(value=v)
                        for f, v in (key.split(":", 1) for key in row["option_ids"])
                    ]
                if condition == "reverse_joint":
                    row["option_ids"].reverse()
                    row["options"].reverse()
                row["target"] = row["options"][row["option_ids"].index(row["target_id"])]
                result[condition].append(row)
    return result


def transfer_gates(candidate, parent, intervals):
    failures = []
    for condition in ("joint", "distractor_options"):
        current, baseline = candidate[condition], parent[condition]
        m, b = current["overall"], baseline["overall"]
        checks = {
            "minimum_task_accuracy": min(v["accuracy"] for v in current["tasks"].values()) >= 0.95,
            "pair_accuracy": current["pair_both_correct"] >= 0.90,
            "improvement": m["accuracy"] - b["accuracy"] >= 0.03,
            "paired_lower": intervals[condition]["lower"][0] > 0,
            "nll": m["nll"] < b["nll"],
            "brier": m["brier"] <= b["brier"],
            "ece": m["ece"] <= b["ece"] + 0.02,
        }
        failures.extend(f"{condition}:{k}" for k, passed in checks.items() if not passed)
    if candidate["known"]["overall"]["accuracy"] < 0.99:
        failures.append("known:accuracy_below_99pct")
    return failures


def control_summary(joint, reverse_joint, no_question):
    if (
        joint.shape != reverse_joint.shape
        or joint.shape != no_question.shape
        or joint.ndim != 2
        or len(joint) % 4
        or joint.shape[1] != 40
    ):
        raise ValueError("aligned complete four-question controls required")
    normal = np.exp(log_probabilities(joint))
    reversed_prob = np.exp(log_probabilities(reverse_joint))[:, ::-1]
    max_diff = float(np.max(np.abs(normal - reversed_prob)))
    same = bool(np.array_equal(joint.argmax(1), 39 - reverse_joint.argmax(1)))
    if max_diff > 1e-5 or not same:
        raise ValueError("candidate order violation")
    without = np.exp(log_probabilities(no_question)).reshape(-1, 4, 40)
    question_diff = float(np.max(np.abs(without - without[:, :1])))
    if question_diff > 1e-5:
        raise ValueError("question-free distributions differ for identical input")
    return dict(
        max_order_probability_difference=max_diff,
        same_argmax_identity=same,
        max_question_free_probability_difference=question_diff,
    )
