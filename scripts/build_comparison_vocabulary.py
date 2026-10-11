"""Train-only comparison vocabulary coverage; preserve all numeric examples."""

import argparse
import copy
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

from scripts.build_conditional_curriculum import verify

WORDS = (
    ("minimum", "maximum"),
    ("smallest", "largest"),
    ("lowest", "highest"),
    ("least", "greatest"),
)


def expand(rows):
    strata = {}
    for row in rows:
        key = (row["wording_variant"], row["numeric_transform"]["scale"])
        strata.setdefault(key, set()).add(row["group"])
    rng = random.Random(151)
    assignments = {}
    for key in sorted(strata):
        groups = sorted(strata[key])
        rng.shuffle(groups)
        for i, group in enumerate(groups):
            if group in assignments:
                raise ValueError("state crosses strata")
            assignments[group] = i % 4
    result = []
    for source in rows:
        row = copy.deepcopy(source)
        variant = assignments[row["group"]]
        if row["task"] in ("extremum", "temporal_order"):
            direction = row["oracle"]["direction"]
            word = WORDS[variant][int(direction == "maximum")]
            question, n = re.subn(r"\b" + direction + r"\b", word, row["question"])
            if n != 1:
                raise ValueError("expected one comparison operator")
            row["question"] = question
            row["comparison_variant"] = variant
        else:
            row["comparison_variant"] = None
        verify(row)
        result.append(row)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    digest = hashlib.sha256(args.source.read_bytes()).hexdigest()
    if digest != "91c4b3f9ff81a3d01c5f4a9db0c83d548b33daab44a9ee879b236a5ded136645":
        raise ValueError("source hash mismatch")
    before = [json.loads(line) for line in args.source.read_text(encoding="utf-8").splitlines()]
    rows = expand(before)
    for a, b in zip(before, rows, strict=True):
        if {k: v for k, v in a.items() if k != "question"} != {
            k: v for k, v in b.items() if k not in ("question", "comparison_variant")
        }:
            raise ValueError("non-question data changed")
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / "train.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    manifest = dict(
        seed=151,
        source_sha256=digest,
        input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        rows=len(rows),
        states=len({r["group"] for r in rows}),
        changed_questions=sum(
            a["question"] != b["question"] for a, b in zip(before, rows, strict=True)
        ),
        comparison_counts=dict(Counter(str(r["comparison_variant"]) for r in rows)),
        joint_counts=dict(
            Counter(
                str(
                    (r["wording_variant"], r["numeric_transform"]["scale"], r["comparison_variant"])
                )
                for r in rows
            )
        ),
        training_started=False,
        scope="Known-word development intervention, not blind evaluation",
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
