"""Independent conditional-state contract and a training-only matching objective."""

import hashlib
import json

from scripts.binding_language import attributes
from scripts.condition_binding_corpus import all_states, identity, ordered, questions_for
from scripts.semantic_state_diagnostic import FIELDS

CONDITIONS = (
    "known",
    "focus",
    "complement",
    "focus_descriptions",
    "complement_descriptions",
    "reverse_focus_descriptions",
    "no_question",
)


def partition_states(excluded, counts=(512, 128, 256)):
    blocked = {identity(s) for s in excluded}
    available = ordered(
        [s for s in all_states() if identity(s) not in blocked], "contrast-states-v1"
    )
    if len(counts) != 3 or min(counts) < 1 or sum(counts) > len(available):
        raise ValueError("insufficient independent states")
    a, b, _ = counts
    return dict(
        train=available[:a],
        validation=available[a : a + b],
        transfer=available[a + b : sum(counts)],
    )


def _row(base, family, split, condition, new_description=False):
    row = dict(base)
    field = base["task"]
    others = ", ".join(f for f in attributes(base["group"], field) if f != field)
    all_fields = ", ".join(attributes(base["group"], "all"))
    templates = (
        f"Read the {field} entry. Give its recorded value.",
        f"Use only {field}; disregard {others}.",
        f"From {all_fields}, answer {field}.",
        f"The requested property is {field}. Ignore every other property.",
        f"Keep every property except {others}. Give the value of the sole survivor.",
        f"Discard the {others} entries. What value is left?",
        f"Ignore {others}; return the only property not ignored.",
        f"Of {all_fields}, do not choose {others}. Supply the remaining value.",
        f"The answer must describe {field}. Which candidate states it?",
        f"Three entries are ruled out: {others}. Give the value from the fourth entry.",
        f"Your output should concern {field}; values for {others} do not answer this request.",
        f"Only one property is allowed. All of {others} are forbidden. "
        "Which candidate gives the allowed value?",
    )
    row.update(
        id=f"contrast-v1:{split}:{base['group']}:{field}:{condition}",
        split=split,
        condition=condition,
        language_family=family,
        question=templates[family],
        rule=dict(excluded=[f for f in FIELDS if f != field])
        if family in (4, 5, 6, 7, 9, 11)
        else dict(requested=field),
    )
    row["option_ids"] = base["option_ids"].copy()
    row["options"] = []
    for oid in row["option_ids"]:
        f, v = oid.split(":", 1)
        if new_description:
            desc = f"For {f}, the value in the record is {v}."
        elif split == "train" and int(base["group"][:8], 16) % 2:
            desc = f"The recorded {f} equals {v}."
        else:
            desc = f"The {f} is {v}."
        row["options"].append(desc)
    if condition.startswith("reverse_"):
        row["option_ids"].reverse()
        row["options"].reverse()
    if condition == "no_question":
        row["question"] = ""
    row["target"] = row["options"][row["option_ids"].index(row["target_id"])]
    return row


def build_rows(states, split):
    if not states or len({identity(s) for s in states}) != len(states):
        raise ValueError("unique nonempty states required")
    if split not in ("train", "validation", "transfer"):
        raise ValueError("unknown partition")
    result = {n: [] for n in CONDITIONS} if split == "transfer" else []
    for index, state in enumerate(states):
        bases = questions_for(state, split)
        if split == "train":
            result.extend(_row(b, index % 8, split, "train") for b in bases)
        elif split == "validation":
            for family in (8, 9):
                result.extend(_row(b, family, split, f"validation-{family}") for b in bases)
        else:
            for condition in CONDITIONS:
                family = (
                    index % 8 if condition == "known" else 11 if "complement" in condition else 10
                )
                result[condition].extend(
                    _row(
                        b,
                        family,
                        split,
                        condition,
                        "descriptions" in condition or condition == "no_question",
                    )
                    for b in bases
                )
    return result


def validate_group(rows):
    if len(rows) != 4 or [r["task"] for r in rows] != list(FIELDS):
        raise ValueError("four aligned attribute queries required")
    if len({r["state"] for r in rows}) != 1 or len({r["group"] for r in rows}) != 1:
        raise ValueError("one shared state required")
    state = json.loads(rows[0]["state"])
    if identity(state) != rows[0]["group"]:
        raise ValueError("state identity mismatch")
    for r in rows:
        permitted = (
            {r["rule"]["requested"]}
            if "requested" in r["rule"]
            else set(FIELDS) - set(r["rule"]["excluded"])
        )
        if (
            permitted != {r["task"]}
            or r["target_id"] != f"{r['task']}:{state[r['task']]}"
            or r["option_ids"] != rows[0]["option_ids"]
            or r["options"] != rows[0]["options"]
            or len(r["options"]) != 40
            or len(set(r["option_ids"])) != 40
            or r["target"] != r["options"][r["option_ids"].index(r["target_id"])]
        ):
            raise ValueError("conditional group semantics or candidate alignment mismatch")


def grouped_schedule(original):
    if len(original) != 512 or sorted(i for b in original for s, i in b if s == 1) != list(
        range(2048)
    ):
        raise ValueError("complete fixed replay order required")
    groups = sorted(
        range(512), key=lambda i: hashlib.sha256(f"17:contrast-order:{i}".encode()).hexdigest()
    )
    return [
        [(0, 4 * g + i) for i in range(4)] + [v for v in b if v[0] == 1]
        for g, b in zip(groups, original, strict=True)
    ]


def contrast_loss(scores, widths, target):
    import torch

    if (
        widths != [40] * 4
        or scores.numel() != 160
        or target.numel() != 4
        or target.unique().numel() != 4
    ):
        raise ValueError("complete four-query forty-candidate group required")
    if not torch.isfinite(scores).all() or target.min() < 0 or target.max() >= 40:
        raise ValueError("finite logits and valid targets required")
    logp = scores.reshape(4, 40).log_softmax(-1)
    return torch.nn.functional.cross_entropy(
        logp[:, target].T, torch.arange(4, device=scores.device)
    )


def eligible(step, old_tasks_ok, legacy_accuracy):
    return step == 0 or bool(old_tasks_ok and legacy_accuracy >= 0.99)


def delta_digest(path):
    """Hash tensor content independently of torch ZIP container filename metadata."""
    import torch

    delta = torch.load(path, map_location="cpu", weights_only=True)
    digest = hashlib.sha256()
    for name, tensor in sorted(delta.items()):
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        digest.update(tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()
