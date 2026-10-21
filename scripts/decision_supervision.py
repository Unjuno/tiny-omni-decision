"""Independent multi-capability supervision contract; no model dependencies."""

import copy
import hashlib
import json
import math
import random
import re
from pathlib import Path

from scripts.semantic_state_diagnostic import FIELDS, options, oracle_state

SPLITS = ("train", "validation", "transfer")
BINDING_FAMILIES = dict(train=range(4), validation=range(4, 6), transfer=range(6, 8))
NUMERIC_FIELDS = ("cost", "risk", "time")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def state_key(state):
    if isinstance(state, str):
        try:
            state = json.loads(state)
        except json.JSONDecodeError:
            return digest(["plain_text", re.sub(r"\s+", " ", state).strip()])
        if isinstance(state, str):
            return digest(["plain_text", re.sub(r"\s+", " ", state).strip()])
    if isinstance(state, list):
        state = [dict(item, cost=item["price"]) if isinstance(item, dict)
                 and "price" in item and "cost" not in item else item for item in state]
    if isinstance(state, list) and state and all(
        isinstance(item, dict) and all(k in item for k in NUMERIC_FIELDS) for item in state
    ):
        values = [tuple(item[k] for k in NUMERIC_FIELDS) for item in state]
        flat = [v for row in values for v in row]
        if any(type(v) is not int for v in flat):
            raise ValueError("integer numeric state required")
        low = min(flat)
        unit = math.gcd(*(v - low for v in flat)) or 1
        state = sorted(tuple((v - low) // unit for v in row) for row in values)
        return digest(["affine_numeric", state])
    return digest(state)


def binding_question(field, family):
    others = ", ".join(f for f in FIELDS if f != field)
    return (
        f"Extract {field} from the observation and choose the matching statement.",
        f"Choose a statement about {field}; statements about {others} are off topic.",
        f"Exclude {others} from consideration. Answer for the one property that remains.",
        f"Only the property outside this list is requested: {others}. Return its value.",
        f"I am asking for {field}, even though the observation also records {others}.",
        f"After removing {others}, which statement describes the remaining property?",
        f"The subject of my query is {field}. Which statement answers my query?",
        f"All properties except {others} are relevant here. Supply the relevant entry.",
    )[family]


def _check_states(states, split):
    if split not in SPLITS or not states or len({state_key(s) for s in states}) != len(states):
        raise ValueError("unique nonempty states and known partition required")


def _row(state, split, task, family, rule, candidates, truth, relevant, target_id):
    group = state_key(state)
    candidates = sorted(candidates, key=lambda c: digest([17, group, c[0]]))
    ids, descriptions = map(list, zip(*candidates, strict=True))
    labels = dict(
        fact_true=[bool(truth[c]) for c in ids] if truth is not None else None,
        request_relevant=[bool(relevant[c]) for c in ids] if relevant is not None else None,
        correct=[c == target_id for c in ids],
    )
    return dict(
        id=f"decision-supervision-v1:{split}:{group}:{task}:{family}",
        group=group, split=split, source="programmatic_decision_v1", task=task,
        family=family, state=canonical(state), rule=rule, options=descriptions,
        option_ids=ids, target_id=target_id, target=descriptions[ids.index(target_id)],
        supervision=labels,
    )


def binding_rows(states, split):
    _check_states(states, split)
    result = []
    candidates = [(o["id"], f"The {o['id'].split(':')[0]} is {o['id'].split(':')[1]}.")
                  for o in options()]
    for state in states:
        state = oracle_state(state)
        truth = {c: state[c.split(":")[0]] == c.split(":")[1] for c, _ in candidates}
        for family in BINDING_FAMILIES[split]:
            for field in FIELDS:
                relevant = {c: c.split(":")[0] == field for c, _ in candidates}
                row = _row(state, split, f"binding:{family}:{field}", family,
                           dict(field=field), candidates, truth, relevant,
                           f"{field}:{state[field]}")
                row["question"] = binding_question(field, family)
                result.append(row)
    return result


def numeric_question(rule, split):
    index = SPLITS.index(split)
    task = rule["operation"]
    if task == "extremum":
        field, direction = rule["field"], rule["direction"]
        word = "smallest" if direction == "minimum" else "largest"
        return (
            f"Pick the item whose {field} is {word}.",
            f"Compare the {field} values and return the item at the {direction}.",
            f"Which listed item has the {word} recorded {field}?",
        )[index]
    threshold = rule["threshold"]
    if task == "count":
        return (
            f"Count the items whose risk does not exceed {threshold}.",
            f"What is the number of items with risk less than or equal to {threshold}?",
            f"How many listed items meet the risk ceiling of {threshold}?",
        )[index]
    return (
        f"Choose the cheapest item with risk no greater than {threshold}; "
        "choose none if there is no qualifying item.",
        f"Limit the selection to risk at most {threshold}, then minimize cost. "
        "Return none when the eligible set is empty.",
        f"Which item has the lowest cost among those satisfying risk <= {threshold}? "
        "Use none if every item violates the limit.",
    )[index]


def numeric_solution(state, rule):
    if rule.get("operation") not in ("extremum", "count", "filter_min_cost"):
        raise ValueError("unknown numeric operation")
    if rule["operation"] == "extremum":
        if rule.get("field") not in NUMERIC_FIELDS or rule.get("direction") not in (
            "minimum", "maximum"
        ):
            raise ValueError("unknown comparison")
        values = [s[rule["field"]] for s in state]
        if len(set(values)) != 4:
            raise ValueError("numeric ties require an explicit policy")
        score = min(values) if rule["direction"] == "minimum" else max(values)
        return next(s["id"] for s in state if s[rule["field"]] == score)
    eligible = [s for s in state if s["risk"] <= rule["threshold"]]
    if rule["operation"] == "count":
        return str(len(eligible))
    return min(eligible, key=lambda s: s["cost"])["id"] if eligible else "none"


def numeric_rows(states, split, seed=17):
    _check_states(states, split)
    result = []
    for state in states:
        if len(state) != 4 or len({s["id"] for s in state}) != 4:
            raise ValueError("four distinct numeric items required")
        threshold = random.Random(digest([seed, state_key(state), "threshold"])).randint(0, 200)
        rules = [dict(operation="extremum", field=f, direction=d)
                 for f in NUMERIC_FIELDS for d in ("minimum", "maximum")]
        rules += [dict(operation=t, threshold=threshold) for t in ("count", "filter_min_cost")]
        for rule in rules:
            task = "numeric:" + ":".join(str(v) for k, v in rule.items() if k != "threshold")
            candidates = ([(str(i), str(i)) for i in range(5)]
                          if rule["operation"] == "count" else
                          [(s["id"], f"Item {s['id']}.") for s in state])
            if rule["operation"] == "filter_min_cost":
                candidates += [("none", "No item qualifies.")]
            row = _row(state, split, task, SPLITS.index(split), rule, candidates,
                       None, None, numeric_solution(state, rule))
            row["question"] = numeric_question(rule, split)
            result.append(row)
    return result


def validate_row(row):
    state = json.loads(row["state"])
    if row["group"] != state_key(state) or row["split"] not in SPLITS:
        raise ValueError("state identity or split mismatch")
    if row["task"].startswith("binding:"):
        family, field = row["family"], row["rule"]["field"]
        if family not in BINDING_FAMILIES[row["split"]] or field not in FIELDS:
            raise ValueError("binding syntax mismatch")
        expected = next(r for r in binding_rows([state], row["split"])
                        if r["family"] == family and r["rule"]["field"] == field)
    elif row["task"].startswith("numeric:"):
        expected = next((r for r in numeric_rows([state], row["split"])
                         if r["rule"] == row["rule"]), None)
    else:
        raise ValueError("unknown task")
    if expected != row:
        raise ValueError("semantic labels, question or candidate integrity mismatch")


def public_example(row):
    return {k: copy.deepcopy(row[k]) for k in ("state", "question", "options", "option_ids")}


def verify_partitions(parts, excluded=()):
    if not parts or set(parts) - set(SPLITS) or any(not rows for rows in parts.values()):
        raise ValueError("nonempty named partitions required")
    seen_states, seen_content, seen_ids = set(), set(), set()
    excluded = set(excluded)
    result = {}
    for split, rows in parts.items():
        states, content, ids = set(), set(), set()
        group_tasks = {}
        for row in rows:
            validate_row(row)
            if row["split"] != split or row["id"] in ids:
                raise ValueError("partition membership or duplicate ID")
            ids.add(row["id"])
            states.add(row["group"])
            normalized = re.sub(r"\s+", " ", canonical(public_example(row)).casefold())
            if normalized in content:
                raise ValueError("duplicate normalized content")
            content.add(normalized)
            group_tasks.setdefault(row["group"], []).append(row["task"])
        if states & seen_states:
            raise ValueError("cross-partition state overlap")
        if states & excluded:
            raise ValueError("previously observed state reused")
        if content & seen_content or ids & seen_ids:
            raise ValueError("cross-partition content or ID overlap")
        for tasks in group_tasks.values():
            expected = (16 if split == "train" else 8) if tasks[0].startswith("binding:") else 8
            if len(tasks) != expected or len(set(tasks)) != expected:
                raise ValueError("incomplete state question group")
        seen_states |= states
        seen_content |= content
        seen_ids |= ids
        result[split] = dict(rows=len(rows), states=len(states))
    return dict(partitions=result, state_overlap=0, normalized_content_overlap=0)


def select_validation(history, baseline, tasks):
    required = {"accuracy", "nll", "brier", "ece", "count"}
    if not isinstance(tasks, dict) or not tasks or set(baseline) != set(tasks) or not history:
        raise ValueError("complete expected task schema required")
    for t, m in baseline.items():
        if (set(m) != required or not all(math.isfinite(v) for v in m.values())
                or type(tasks[t]) is not int or tasks[t] < 1 or m["count"] != tasks[t]):
            raise ValueError("finite complete baseline required")
    if len({r["step"] for r in history}) != len(history):
        raise ValueError("unique optimizer steps required")
    allowed = []
    for row in history:
        metrics = row["metrics"]
        if row["purpose"] != "validation" or set(metrics) != set(tasks):
            raise ValueError("only complete validation task metrics can select checkpoints")
        if any(set(m) != required or m["count"] != baseline[t]["count"]
               for t, m in metrics.items()):
            raise ValueError("missing metric or changed evaluation counts")
        finite = all(math.isfinite(v) for m in metrics.values() for v in m.values())
        if type(row["step"]) is not int or row["step"] < 0:
            raise ValueError("nonnegative integer step required")
        if finite and any(not (0 <= m["accuracy"] <= 1 and m["nll"] >= 0
                               and 0 <= m["brier"] <= 2 and 0 <= m["ece"] <= 1)
                          or type(m["count"]) is not int for m in metrics.values()):
            raise ValueError("invalid metric domain")
        eligible = finite and all(
            m["accuracy"] >= baseline[t]["accuracy"] - .01 - 1e-12
            and all(m[k] <= baseline[t][k] + limit + 1e-12
                    for k, limit in (("nll", .02), ("brier", .01), ("ece", .02)))
            for t, m in metrics.items())
        if eligible:
            allowed.append((sum(m["nll"] for m in metrics.values()) / len(tasks), row["step"]))
    if not any(r["step"] == 0 and r["metrics"] == baseline for r in history):
        raise ValueError("unchanged step-zero baseline required")
    return min(allowed)[1]


def load_for_fit(directory, partition, manifest_sha256):
    if partition not in ("train", "validation", "retention_train", "retention_validation"):
        raise ValueError("partition unavailable to normal training/selection")
    directory = Path(directory)
    manifest_path = directory / "result.json"
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != manifest_sha256:
        raise ValueError("corpus manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["status"] != "PASS_FROZEN_DECISION_SUPERVISION":
        raise ValueError("incomplete corpus")
    path = directory / f"{partition}.jsonl"
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["artifacts"][path.name]:
        raise ValueError("partition hash mismatch")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    expected = "train" if partition.endswith("train") else "validation"
    if not rows or any(r["split"] != expected for r in rows):
        raise ValueError("partition membership mismatch")
    return rows
