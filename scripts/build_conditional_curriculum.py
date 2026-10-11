"""Generate an oracle-labeled development curriculum; never a product audit."""

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def build(seed, states, split):
    rng = random.Random(seed)
    rows = []
    for _ in range(states):
        labels = rng.sample(["K", "M", "P", "R", "T", "V", "X", "Z"], 4)
        columns = {key: rng.sample(range(1, 200), 4) for key in ("cost", "risk", "time")}
        state = [
            dict(id=label, **{k: v[i] for k, v in columns.items()})
            for i, label in enumerate(labels)
        ]
        # Identity includes a label/order-invariant numeric representation.
        group = digest(sorted(tuple(row[k] for k in ("cost", "risk", "time")) for row in state))
        threshold = rng.randint(20, 180)
        questions = []
        for field in ("cost", "risk", "time"):
            for direction in ("minimum", "maximum"):
                values = columns[field]
                target = labels[
                    values.index(min(values) if direction == "minimum" else max(values))
                ]
                questions.append(
                    (
                        f"Select the item with the {direction} {field} value.",
                        labels,
                        target,
                        "extremum" if field != "time" else "temporal_order",
                        {"field": field, "direction": direction},
                    )
                )
        count = sum(v <= threshold for v in columns["risk"])
        questions.append(
            (
                f"How many items have risk at most {threshold}?",
                list(map(str, range(5))),
                str(count),
                "count",
                {"threshold": threshold},
            )
        )
        eligible = [i for i, v in enumerate(columns["risk"]) if v <= threshold]
        selected = min(eligible, key=lambda i: columns["cost"][i]) if eligible else None
        target = labels[selected] if selected is not None else "none"
        questions.append(
            (
                f"Among items with risk at most {threshold}, select the item with the minimum cost. "
                "Select none if no item qualifies.",
                labels + ["none"],
                target,
                "conditional_selection",
                {"threshold": threshold},
            )
        )
        for index, (question, options, target, task, oracle) in enumerate(questions):
            options = list(options)
            rng.shuffle(options)
            rows.append(
                {
                    "id": f"{group}:{index}",
                    "group": group,
                    "split": split,
                    "state": json.dumps(state),
                    "question": question,
                    "options": options,
                    "target": target,
                    "task": task,
                    "oracle": oracle,
                }
            )
    return rows


def verify(row):
    state = json.loads(row["state"])
    oracle = row["oracle"]
    if row["task"] in ("extremum", "temporal_order"):
        ordered = sorted(state, key=lambda item: item[oracle["field"]])
        target = ordered[0 if oracle["direction"] == "minimum" else -1]["id"]
    elif row["task"] == "count":
        target = str(len([s for s in state if s["risk"] <= oracle["threshold"]]))
    else:
        candidates = sorted(
            (s for s in state if s["risk"] <= oracle["threshold"]), key=lambda s: s["cost"]
        )
        target = candidates[0]["id"] if candidates else "none"
    if target != row["target"] or target not in row["options"]:
        raise ValueError("oracle mismatch")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    splits = {"train": build(17, 1024, "train"), "validation": build(29, 256, "validation")}
    groups = [{r["group"] for r in rows} for rows in splits.values()]
    if groups[0] & groups[1]:
        raise ValueError("state identity overlap")
    for rows in splits.values():
        if len({r["id"] for r in rows}) != len(rows):
            raise ValueError("duplicate IDs")
        for row in rows:
            verify(row)
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "source": "local programmatic conditional curriculum",
        "purpose": "development repair only; not final/general/multimodal evaluation",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seeds": {"train": 17, "validation": 29},
        "splits": {},
    }
    for name, rows in splits.items():
        path = args.output / (name + ".jsonl")
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        report["splits"][name] = {
            "rows": len(rows),
            "states": len({r["group"] for r in rows}),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "tasks": dict(Counter(r["task"] for r in rows)),
            "target_positions": dict(Counter(r["options"].index(r["target"]) for r in rows)),
        }
    (args.output / "manifest.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
