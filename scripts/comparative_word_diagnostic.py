"""Factorial comparison-word diagnostic; not a replacement benchmark."""

import argparse
import json
from pathlib import Path

from scripts.build_conditional_curriculum import build, verify
from scripts.build_numeric_range_curriculum import affine
from scripts.lexical_transfer import CHECKPOINTS
from scripts.range_generalization import evaluate, reject_overlap, sha, write

VARIANTS = tuple(
    f"{r}-{v}-{s}"
    for r in ("base", "expanded")
    for v in ("minimum", "smallest")
    for s in ("select", "which")
)


def render(row, variant):
    if variant not in VARIANTS or row["task"] != "extremum":
        raise ValueError("unsupported diagnostic row or variant")
    numeric, vocabulary, structure = variant.split("-")
    row = affine(row, *((7, 19) if numeric == "expanded" else (1, 0)))
    direction = row["oracle"]["direction"]
    word = (
        direction
        if vocabulary == "minimum"
        else ("smallest" if direction == "minimum" else "largest")
    )
    field = row["oracle"]["field"]
    row["question"] = (
        f"Select the item with the {word} {field} value."
        if structure == "select"
        else f"Which item has the {word} {field}?"
    )
    row.update(base_id=row["id"], id=variant + ":" + row["id"], variant=variant)
    verify(row)
    return row


def prepare(root, output):
    base = [r for r in build(137, 128, "development") if r["task"] == "extremum"]
    rows = [render(r, v) for v in VARIANTS for r in base]
    names = (
        "conditional-curriculum-01/train.jsonl",
        "conditional-curriculum-01/validation.jsonl",
        "numeric-range-curriculum-01/train.jsonl",
        "numeric-range-curriculum-01/transfer.jsonl",
        "condition-transfer-controls-01/inputs.jsonl",
        "range-generalization-inputs-01/inputs.jsonl",
        "lexical-range-curriculum-01/train.jsonl",
        "lexical-transfer-inputs-01/inputs.jsonl",
    )
    previous = [
        json.loads(line)
        for name in names
        for line in (root / name).read_text(encoding="utf-8").splitlines()
    ]
    reject_overlap(rows, previous)
    output.mkdir(parents=True, exist_ok=False)
    path = output / "inputs.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    write(
        output / "manifest.json",
        dict(
            seed=137,
            states=128,
            rows=len(rows),
            input_sha256=sha(path),
            previous_sha256={n: sha(root / n) for n in names},
            source_sha256=sha(Path(__file__)),
            scope="Known-word factorial development diagnosis",
        ),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("prepare", "evaluate"))
    for name in ("root", "model", "data", "output"):
        p.add_argument("--" + name, type=Path, required=name in ("root", "output"))
    p.add_argument("--input-sha256")
    args = p.parse_args()
    if args.mode == "prepare":
        prepare(args.root, args.output)
    else:
        if not all((args.model, args.data, args.input_sha256)):
            p.error("model, data, input-sha256 required")
        evaluate(
            args.root,
            args.model,
            args.data,
            args.input_sha256,
            args.output,
            CHECKPOINTS,
            VARIANTS,
            4096,
        )


if __name__ == "__main__":
    main()
