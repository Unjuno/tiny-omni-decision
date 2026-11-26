"""Condition relevance corpus and development-only loading contract."""

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from scripts.decision_supervision import canonical, digest, state_key
from scripts.semantic_state_diagnostic import FIELDS, QUESTIONS, oracle_state
from tiny_omni_decision.dataset import CLEVR4_TAXONOMIES

SPLITS = ("train", "calibration", "validation", "transfer")
DEVELOPMENT_FILES = {
    "binding_train": ("binding_train.jsonl", "train"),
    "numeric_train": ("numeric_train.jsonl", "train"),
    "legacy_train": ("legacy_train.jsonl", "train"),
    "validation": ("validation.jsonl", "validation"),
    "old_validation": ("old_validation.jsonl", "validation"),
    "legacy_validation": ("legacy_validation.jsonl", "validation"),
}

# Surface categories include materials and patterns; "material" alone is ambiguous.
QUERY_BANK = {
    "color": (
        "Which color appears on the objects?",
        "Describe the color shown in this observation.",
        "Choose the answer that names the objects' color.",
        "Set aside shape, surface category, and number. Report the color.",
        "What color should be recorded for these objects?",
        "I need the color, without their quantity, shape, or surface category.",
        "Tell me the color of the objects being observed.",
        "Choose their color rather than their shape, surface category, or count.",
        "What is their observed color?",
        "Provide the color; the other recorded properties do not answer this query.",
        "Which answer describes the coloration visible here?",
        "Disregard the object's form, surface category, and quantity; give its color.",
    ),
    "texture": (
        "Which surface category do the objects exhibit?",
        "Describe their surface appearance category.",
        "Choose the answer naming the objects' texture.",
        "Set aside color, shape, and number. Report the surface category.",
        "What texture should be recorded for these objects?",
        "I need the surface category, without their color, quantity, or shape.",
        "Tell me the surface category of the objects being observed.",
        "Choose their texture rather than their shape, color, or count.",
        "What is their observed surface category?",
        "Provide the texture; the other recorded properties do not answer this query.",
        "Which answer describes how the surfaces are categorized?",
        "Disregard the object's form, color, and quantity; give its surface category.",
    ),
    "count": (
        "How many objects are present?",
        "Give the number of objects in the observation.",
        "Choose the answer stating how many objects are visible.",
        "Set aside color, shape, and surface category. Report the number of objects.",
        "What total should be recorded when the objects are counted?",
        "I need the quantity of objects, without their color, shape, or surface category.",
        "Tell me the total number of objects being observed.",
        "Choose their count rather than their shape, color, or surface category.",
        "How many separate objects does this observation contain?",
        "Provide the object count; the other recorded properties do not answer this query.",
        "What is the cardinality of the observed collection of objects?",
        "Disregard color, form, and surface category; give the number of objects.",
    ),
    "shape": (
        "Which object shape do the objects have?",
        "Describe the object form shown in this observation.",
        "Choose the answer naming the objects' shape.",
        "Set aside color, surface category, and number. Report the object form.",
        "What shape should be recorded for these objects?",
        "I need the object form, without their color, quantity, or surface category.",
        "Tell me the object form of the objects being observed.",
        "Choose their shape rather than their surface category, color, or count.",
        "What is their observed object shape?",
        "Provide the shape; the other recorded properties do not answer this query.",
        "Which answer identifies the objects' form?",
        "Disregard color, surface category, and quantity; give the object's form.",
    ),
}


def description(field, value, form):
    return {
        "key_value": f"{field}: {value}",
        "sentence": f"The {field} is {value}.",
        "reported": f"Reported {field}: {value}.",
        "attribute": f"{value} is the observed {field}.",
    }[form]


def make_rows(state, split, index):
    if split not in SPLITS or type(index) is not int or index < 0:
        raise ValueError("known split and nonnegative assignment index required")
    state = oracle_state(state)
    group = state_key(state)
    known = (index % 3, index % 3 + 3)
    seen_forms, new_forms = ("key_value", "sentence"), ("reported", "attribute")
    if split == "transfer":
        views = [(f"{q}_query_{o}_options", queries, forms)
                 for q, queries in (("seen", known), ("new", (10, 11)))
                 for o, forms in (("seen", seen_forms), ("new", new_forms))]
    else:
        queries = {"train": known, "calibration": (6, 7), "validation": (8, 9)}[split]
        views = [(split, queries, seen_forms)]
    candidates = [(field, value) for field in FIELDS for value in CLEVR4_TAXONOMIES[field]]
    candidates.sort(key=lambda c: digest([17, "condition-relevance-options-v1", group, c]))
    ids = [f"{field}:{value}" for field, value in candidates]
    rows = []
    for condition, queries, forms in views:
        for field in FIELDS:
            for query_index in queries:
                for form in forms:
                    target = f"{field}:{state[field]}"
                    labels = dict(
                        fact_true=[state[f] == v for f, v in candidates],
                        request_relevant=[f == field for f, _ in candidates],
                        correct=[candidate == target for candidate in ids],
                    )
                    rows.append(dict(
                        id=f"condition-relevance-v1:{split}:{group}:{condition}:"
                           f"{field}:{query_index}:{form}",
                        group=group, split=split, source="condition_relevance_v1",
                        state=canonical(state), question=QUERY_BANK[field][query_index],
                        options=[description(f, v, form) for f, v in candidates],
                        option_ids=ids.copy(), target_id=target,
                        target=description(field, state[field], form),
                        field=field, query_index=query_index, option_form=form,
                        assignment_index=index, condition=condition, supervision=labels,
                    ))
    return rows


def public_pairs(row):
    return [(f"Question: {row['question']}\nCandidate answer: {option}",
             "State: " + row["state"]) for option in row["options"]]


def unused_states(universe, excluded, previous_counts):
    if (len(previous_counts) != 3 or any(type(n) is not int or n < 1
                                       for n in previous_counts)):
        raise ValueError("three positive previous state counts required")
    if len({state_key(s) for s in universe}) != len(universe):
        raise ValueError("duplicate universe states")
    excluded = set(excluded)
    pool = sorted((s for s in universe if state_key(s) not in excluded),
                  key=lambda s: digest([17, "decision-supervision-binding-v1", state_key(s)]))
    count = sum(previous_counts)
    if len(pool) < count:
        raise ValueError("insufficient states for previous reservation")
    return pool[count:], {state_key(s) for s in pool[:count]}


def content_key(row):
    # Reordering options and changing IDs cannot evade duplicate-content detection.
    value = dict(state=json.loads(row["state"]), question=row["question"],
                 options=sorted(row["options"]))
    return digest(re.sub(r"\s+", " ", canonical(value).casefold()))


def validate_parts(parts, reserved):
    if not parts or set(parts) - set(SPLITS) or any(not rows for rows in parts.values()):
        raise ValueError("nonempty named partitions required")
    reserved, states_seen, content_seen = set(reserved), set(), set()
    accounting = {}
    for split, rows in parts.items():
        states, content, ids, groups = set(), set(), set(), {}
        for row in rows:
            state = oracle_state(json.loads(row["state"]))
            group = state_key(state)
            if row["split"] != split or row["group"] != group or row["id"] in ids:
                raise ValueError("partition, state or duplicate ID mismatch")
            labels = row["supervision"]
            truth, relevance = [], []
            for candidate in row["option_ids"]:
                field, value = candidate.split(":", 1)
                truth.append(state[field] == value)
                relevance.append(row["field"] == field)
            if labels != dict(fact_true=truth, request_relevant=relevance,
                              correct=[a and b for a, b in zip(truth, relevance, strict=True)]):
                raise ValueError("supervision integrity mismatch")
            if row["question"] in QUESTIONS:
                raise ValueError("old native diagnostic question reused")
            key = content_key(row)
            if key in content or row["id"] in ids:
                raise ValueError("duplicate normalized content or ID")
            ids.add(row["id"])
            states.add(group)
            content.add(key)
            groups.setdefault(group, []).append(row)
        if states & states_seen or content & content_seen:
            raise ValueError("cross-partition overlap")
        if states & reserved:
            raise ValueError("reserved historical state reused")
        for group_rows in groups.values():
            first = group_rows[0]
            expected = make_rows(json.loads(first["state"]), split, first["assignment_index"])
            if {r["id"]: r for r in expected} != {r["id"]: r for r in group_rows}:
                raise ValueError("incomplete or modified question group")
        states_seen.update(states)
        content_seen.update(content)
        accounting[split] = dict(
            rows=len(rows), unique_ids=len(ids), unique_states=len(states), native_assets=0,
            modalities={"text_structured_state": len(rows)},
            sources={"condition_relevance_v1": len(rows)},
            fields=dict(sorted(Counter(r["field"] for r in rows).items())),
            views=dict(sorted(Counter(r["condition"] for r in rows).items())),
            candidate_counts=dict(sorted(Counter(len(r["options"]) for r in rows).items())),
            target_positions=dict(sorted(Counter(r["option_ids"].index(r["target_id"])
                                                 for r in rows).items())),
            query_count=len({r["question"] for r in rows}),
            attribute_values={field: dict(sorted(Counter(json.loads(g[0]["state"])[field]
                for g in groups.values()).items())) for field in FIELDS},
        )
    return dict(partitions=accounting, unique_states=len(states_seen),
                normalized_content_overlap=0, state_overlap=0, native_assets=0)


def load_development(root, purpose, manifest_sha256):
    if purpose not in DEVELOPMENT_FILES:
        raise ValueError("only a declared development partition may be loaded")
    root = Path(root)
    raw = (root / "result.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest_sha256:
        raise ValueError("manifest hash mismatch")
    manifest = json.loads(raw)
    if manifest["status"] != "PASS_FROZEN_CONDITION_RELEVANCE":
        raise ValueError("unverified corpus")
    name, split = DEVELOPMENT_FILES[purpose]
    raw = (root / name).read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest["artifacts"][name]:
        raise ValueError("partition hash mismatch")
    rows = [json.loads(line) for line in raw.splitlines()]
    if not rows or any(row["split"] != split for row in rows):
        raise ValueError("partition membership mismatch")
    return rows
