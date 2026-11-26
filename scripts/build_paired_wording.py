"""Train-only wording views. Repeated views are not new underlying examples."""

import argparse
import copy
import json
import re
from collections import Counter
from pathlib import Path

from scripts.build_comparison_vocabulary import WORDS
from scripts.build_conditional_curriculum import verify
from scripts.build_lexical_range_curriculum import reword
from scripts.range_generalization import sha, write


def views(source):
    verify(source)
    result = []
    for view in range(4):
        row = copy.deepcopy(source)
        if view:
            # reword variants 1..3 replace the entire question, so existing synonyms
            # cannot accidentally survive substitution of the canonical operator.
            row = reword(row, view)
            if row["task"] in ("extremum", "temporal_order"):
                direction = row["oracle"]["direction"]
                family = (source["comparison_variant"] + view) % 4
                row["question"], n = re.subn(
                    r"\b" + direction + r"\b",
                    WORDS[family][int(direction == "maximum")], row["question"],
                )
                if n != 1:
                    raise ValueError("comparison replacement failed")
                row["comparison_variant"] = family
        for key in ("state", "options", "target", "oracle", "group", "task"):
            if row[key] != source[key]:
                raise ValueError("non-wording data changed")
        row.update(id=f"{source['id']}:wording-view-{view}", parent_id=source["id"], view=view)
        verify(row)
        result.append(row)
    if len({r["question"] for r in result}) != 4:
        # Count/conditional templates may already equal one of the alternatives.
        # Fail rather than silently counting identical text as diversity.
        raise ValueError("duplicate wording view")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = Path("C:/CodexArtifacts/pretrained-reuse-r2/"
                  "comparison-vocabulary-curriculum-01/train.jsonl")
    if sha(source) != "41b35f3c2b1161d21ebf123d8ba39b29aa17dc9dc82c5f03ff7f291b9d368c76":
        raise ValueError("frozen source changed")
    parents = [json.loads(x) for x in source.read_text(encoding="utf-8").splitlines()]
    # Restrict augmentation to comparison questions; retain other tasks once.
    rows = []
    for parent in parents:
        if parent["task"] in ("extremum", "temporal_order"):
            rows.extend(views(parent))
        else:
            rows.append(dict(parent, parent_id=parent["id"], view=0))
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output / "train-views.jsonl"
    output.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    write(args.output / "manifest.json", dict(
        source_sha256=sha(source), output_sha256=sha(output), builder_sha256=sha(Path(__file__)),
        rows=len(rows), unique_parent_questions=len(parents),
        unique_states=len({r["group"] for r in rows}),
        views_by_task=dict(Counter(r["task"] for r in rows)),
        training_started=False, scope="Train-only views; no new unique states",
        sampling_warning="Sample parents first; uniform view sampling changes task weights.",
    ))


if __name__ == "__main__":
    main()
