"""New state records for a predeclared decoder comparison; no model inference."""

import argparse
import json
from pathlib import Path

from scripts.build_conditional_curriculum import build, verify
from scripts.build_vocabulary_transfer import render
from scripts.range_generalization import content, reject_overlap, sha, write

VARIANTS = ("base", "option_reverse", "state_reverse", "expanded", "domain", "expanded_domain")


def transform(row, variant):
    if variant not in VARIANTS:
        raise ValueError("unknown variant")
    numeric = "expanded" if variant in ("expanded", "expanded_domain") else "base"
    family = 4 if variant in ("domain", "expanded_domain") else 0
    value = render(row, f"{numeric}-{family}")
    if variant == "option_reverse":
        value["options"] = value["options"][::-1]
    if variant == "state_reverse":
        value["state"] = json.dumps(json.loads(value["state"])[::-1])
    value.update(id=variant + ":" + row["id"], base_id=row["id"], variant=variant)
    verify(value)
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
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
        "vocabulary-transfer-inputs-01/inputs.jsonl",
    )
    rows = [transform(r, v) for v in VARIANTS for r in build(179, 128, "development")]
    previous = [
        json.loads(s) for n in names for s in (a.root / n).read_text(encoding="utf-8").splitlines()
    ]
    reject_overlap(rows, previous)
    assert len(rows) == 6144 and len({r["group"] for r in rows}) == 128
    a.output.mkdir(parents=True, exist_ok=False)
    path = a.output / "inputs.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    write(
        a.output / "manifest.json",
        dict(
            seed=179,
            states=128,
            condition_rows=len(rows),
            base_questions=len({r["base_id"] for r in rows}),
            unique_content=len({content(r) for r in rows}),
            input_sha256=sha(path),
            builder_sha256=sha(Path(__file__)),
            variants=VARIANTS,
            previous_sha256={n: sha(a.root / n) for n in names},
            overlap=0,
            inference_started=False,
            known_templates=True,
        ),
    )


if __name__ == "__main__":
    main()
