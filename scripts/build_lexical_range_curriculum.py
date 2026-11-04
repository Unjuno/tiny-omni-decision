"""Prepare train-only wording coverage, retaining numeric data and identities."""

import argparse
import copy
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

from scripts.build_conditional_curriculum import verify


def template(question):
    return re.sub(r"\d+", "<number>", " ".join(question.lower().split()))


def reword(row, variant):
    if variant not in range(4):
        raise ValueError("unknown wording variant")
    row = copy.deepcopy(row)
    oracle = row["oracle"]
    if variant:
        if row["task"] in ("extremum", "temporal_order"):
            field, direction = oracle["field"], oracle["direction"]
            text = [
                f"Identify the item whose {field} is the {direction}.",
                f"Compare {field} across the items and return the {direction} item's ID.",
                f"For {field}, which item attains the {direction} value?",
            ]
        elif row["task"] == "count":
            n = oracle["threshold"]
            text = [
                f"Give the count of items with risk <= {n}.",
                f"What is the total number of items whose risk does not exceed {n}?",
                f"Risk limit: {n}, inclusive. How many items satisfy this limit?",
            ]
        else:
            n = oracle["threshold"]
            text = [
                f"Keep only items with risk <= {n}; return the minimum-cost item's ID. "
                "If the eligible set is empty, return none.",
                f"Risk must not exceed {n}. Pick the minimum-cost eligible item, "
                "or none when there are no eligible items.",
                f"Apply an inclusive risk ceiling of {n}, then minimize cost among "
                "eligible items. Return the item ID or none if the set is empty.",
            ]
        row["question"] = text[variant - 1]
    row["wording_variant"] = variant
    verify(row)
    return row


def build_rows(rows):
    groups = sorted({r["group"] for r in rows})
    random.Random(101).shuffle(groups)
    mapping = {g: i % 4 for i, g in enumerate(groups)}
    return [reword(r, mapping[r["group"]]) for r in rows]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    source = args.root / "numeric-range-curriculum-01/train.jsonl"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != "95f4edc33711a6ac334fff77cb08300a45bbe2bc8acefad1e8591b069c076101":
        raise ValueError("source hash mismatch")
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    output = build_rows(rows)
    # Original template retention is intentional; prohibit reuse of probe paraphrases.
    probe_templates = set()
    for name in ("condition-transfer-controls-01", "range-generalization-inputs-01"):
        for line in (args.root / name / "inputs.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["variant"] in ("wording", "combined"):
                probe_templates.add(template(row["question"]))
    if {template(r["question"]) for r in output if r["wording_variant"]} & probe_templates:
        raise ValueError("observed probe paraphrase reused")
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / "train.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in output), encoding="utf-8")
    manifest = dict(
        source_sha256=digest,
        input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        seed=101,
        rows=len(output),
        groups=len({r["group"] for r in output}),
        variants=dict(Counter(r["wording_variant"] for r in output)),
        observed_probe_template_overlap=0,
        training_started=False,
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
