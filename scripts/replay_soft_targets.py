"""Strict option-aligned training teacher probabilities for the replay experiment."""

import json
import math


def teacher_distribution(row, source):
    _, state, question = row["id"].split(":", 2)
    if state != source["state_id"]:
        raise ValueError("state mismatch")
    spec = json.loads(source["questions"])[question]
    teacher = json.loads(source["teacher"])[question]
    options = row["options"]
    if spec["type"] == "score":
        if options != spec["criteria"]:
            raise ValueError("score criterion order mismatch")
        p = [float(teacher["probabilities"][str(i)]) for i in range(len(options))]
    elif spec["type"] == "noul":
        keys = [o.split(":", 1)[0] for o in options]
        if set(keys) != {"true", "false"} or len(keys) != 2:
            raise ValueError("binary IDs mismatch")
        p = [float(teacher["noul"]) if k == "true" else 1 - float(teacher["noul"]) for k in keys]
    elif spec["type"] == "choice":
        keys = [o.split(":", 1)[0] for o in options]
        if len(set(keys)) != len(keys) or set(keys) != set(teacher["probabilities"]):
            raise ValueError("choice IDs mismatch")
        p = [float(teacher["probabilities"][k]) for k in keys]
    else:
        raise ValueError("unsupported teacher type")
    total = sum(p)
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in p) or abs(total - 1) > 0.001:
        raise ValueError("invalid probability simplex")
    return [v / total for v in p]
