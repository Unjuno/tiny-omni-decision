"""Prepare fixed post-vocabulary development inputs, never load a model."""

import argparse
import json
from pathlib import Path

from scripts.build_comparison_vocabulary import WORDS
from scripts.build_conditional_curriculum import build, verify
from scripts.build_lexical_range_curriculum import template
from scripts.build_numeric_range_curriculum import affine
from scripts.range_generalization import content, reject_overlap, sha, write

VARIANTS = tuple(f"{numeric}-{family}" for numeric in ("base", "expanded") for family in range(5))


def render(row, variant):
    if variant not in VARIANTS:
        raise ValueError("unknown variant")
    numeric, family = variant.split("-")
    family = int(family)
    row = affine(row, *((7, 19) if numeric == "expanded" else (1, 0)))
    o = row["oracle"]
    if row["task"] in ("extremum", "temporal_order"):
        direction = int(o["direction"] == "maximum")
        if family < 4:
            row["question"] = (
                f"Return the ID associated with the {WORDS[family][direction]} "
                f"{o['field']} among these items."
            )
        else:
            pairs = {
                "cost": ("cheapest", "priciest"),
                "risk": ("safest", "riskiest"),
                "time": ("earliest", "latest"),
            }
            row["question"] = (
                "Use cost as price, risk as danger, and time as timestamp. "
                f"Return the {pairs[o['field']][direction]} item's ID."
            )
    elif row["task"] == "count":
        row["question"] = f"Count the items whose risk is <= {o['threshold']}."
    else:
        row["question"] = (
            f"Under a risk ceiling of {o['threshold']} (inclusive), return the minimum-cost "
            "item ID, or none if no item is eligible."
        )
    row.update(base_id=row["id"], id=variant + ":" + row["id"], variant=variant)
    verify(row)
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    rows = [render(r, v) for v in VARIANTS for r in build(167, 128, "development")]
    names = (
        "conditional-curriculum-01/train.jsonl",
        "conditional-curriculum-01/validation.jsonl",
        "numeric-range-curriculum-01/train.jsonl",
        "numeric-range-curriculum-01/transfer.jsonl",
        "condition-transfer-controls-01/inputs.jsonl",
        "range-generalization-inputs-01/inputs.jsonl",
        "lexical-range-curriculum-01/train.jsonl",
        "lexical-transfer-inputs-01/inputs.jsonl",
        "comparative-word-inputs-01/inputs.jsonl",
        "comparison-vocabulary-curriculum-01/train.jsonl",
    )
    old = [
        json.loads(line)
        for n in names
        for line in (args.root / n).read_text(encoding="utf-8").splitlines()
    ]
    reject_overlap(rows, old)
    if {template(r["question"]) for r in rows} & {template(r["question"]) for r in old}:
        raise ValueError("template overlap")
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / "inputs.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    write(
        args.output / "manifest.json",
        dict(
            seed=167,
            states=128,
            condition_rows=len(rows),
            unique_base_question_ids=len({r["base_id"] for r in rows}),
            unique_content=len({content(r) for r in rows}),
            input_sha256=sha(path),
            source_sha256=sha(Path(__file__)),
            previous_sha256={n: sha(args.root / n) for n in names},
            variants=VARIANTS,
            overlap=0,
            inference_started=False,
        ),
    )


if __name__ == "__main__":
    main()
