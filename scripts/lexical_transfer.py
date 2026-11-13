"""Fixed selected-weight wording transfer, without training or selection."""

import argparse
import json
from pathlib import Path

from scripts.build_conditional_curriculum import build, verify
from scripts.build_lexical_range_curriculum import template
from scripts.range_generalization import VARIANTS, evaluate, reject_overlap, sha, transform, write

CHECKPOINTS = {
    "parent": (
        "conditional-repair-range-warmstart-01",
        "f287bb0138fd0f264331cd2fc983567bc68323a85a25cda467387fffae7bc93b",
    ),
    "candidate": (
        "conditional-repair-lexical-range-01",
        "fccee21e0e5ffbaffdee899a65fe6538f52f3b788752a56b6b5b6388cad15abb",
    ),
}


def transform_new(row, variant):
    row = transform(row, variant)
    if variant in ("wording", "combined"):
        o = row["oracle"]
        if row["task"] in ("extremum", "temporal_order"):
            row["question"] = f"Find the {o['direction']} {o['field']}; give the item's identifier."
        elif row["task"] == "count":
            row["question"] = f"How many records satisfy risk <= {o['threshold']}?"
        else:
            row["question"] = (
                f"Considering only records with risk <= {o['threshold']}, return the identifier "
                "with minimum cost. If no record meets the condition, return none."
            )
    verify(row)
    return row


def prepare(root, output):
    rows = [transform_new(r, v) for v in VARIANTS for r in build(127, 256, "development")]
    names = (
        "conditional-curriculum-01/train.jsonl",
        "conditional-curriculum-01/validation.jsonl",
        "numeric-range-curriculum-01/train.jsonl",
        "numeric-range-curriculum-01/transfer.jsonl",
        "condition-transfer-controls-01/inputs.jsonl",
        "range-generalization-inputs-01/inputs.jsonl",
        "lexical-range-curriculum-01/train.jsonl",
    )
    previous = [
        json.loads(line)
        for name in names
        for line in (root / name).read_text(encoding="utf-8").splitlines()
    ]
    reject_overlap(rows, previous)
    old_templates = {template(r["question"]) for r in previous}
    novel = {template(r["question"]) for r in rows if r["variant"] in ("wording", "combined")}
    if novel & old_templates:
        raise ValueError("paraphrase template overlap")
    output.mkdir(parents=True, exist_ok=False)
    path = output / "inputs.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    write(
        output / "manifest.json",
        dict(
            seed=127,
            rows=len(rows),
            states=256,
            input_sha256=sha(path),
            previous_sha256={name: sha(root / name) for name in names},
            source_sha256=sha(Path(__file__)),
            group_content_overlap=0,
            novel_template_overlap=0,
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "evaluate"))
    for name in ("root", "model", "data", "output"):
        parser.add_argument("--" + name, type=Path, required=name in ("root", "output"))
    parser.add_argument("--input-sha256")
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.root, args.output)
    else:
        if not all((args.model, args.data, args.input_sha256)):
            parser.error("model, data, input-sha256 required")
        evaluate(args.root, args.model, args.data, args.input_sha256, args.output, CHECKPOINTS)


if __name__ == "__main__":
    main()
