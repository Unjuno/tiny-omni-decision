"""Fixed one-factor transfer controls, development only; no model selection."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

from scripts.build_conditional_curriculum import build, verify

VARIANTS = ("control", "labels", "identifier_key", "wording", "magnitude", "field_name", "combined")


def transform(row, variant):
    if variant not in VARIANTS:
        raise ValueError("unknown variant")
    row = copy.deepcopy(row)
    state = json.loads(row["state"])
    enabled = set(VARIANTS[1:-1]) if variant == "combined" else {variant}
    if "labels" in enabled:
        mapping = {item["id"]: chr(65 + i) for i, item in enumerate(state)}
        for item in state:
            item["id"] = mapping[item["id"]]
        if row["task"] != "count":
            row["options"] = [mapping.get(v, v) for v in row["options"]]
            row["target"] = mapping.get(row["target"], row["target"])
    if "magnitude" in enabled:
        for item in state:
            for key in ("cost", "risk", "time"):
                item[key] = 5 * item[key] + 10
        if "threshold" in row["oracle"]:
            old = row["oracle"]["threshold"]
            row["oracle"]["threshold"] = 5 * old + 10
            row["question"] = row["question"].replace(str(old), str(5 * old + 10))
    if "wording" in enabled:
        oracle = row["oracle"]
        if row["task"] in ("extremum", "temporal_order"):
            direction = "lowest" if oracle["direction"] == "minimum" else "highest"
            row["question"] = f"Select the candidate with the {direction} {oracle['field']}."
        elif row["task"] == "count":
            row["question"] = (
                f"Count candidates whose risk is no greater than {oracle['threshold']}."
            )
        else:
            row["question"] = (
                "Choose the lowest-cost candidate whose risk is no greater than "
                f"{oracle['threshold']}. "
                "Choose none if there is no eligible candidate."
            )
    row["state"] = json.dumps(state)
    verify(row)
    if "field_name" in enabled:
        for item in state:
            item["price"] = item.pop("cost")
        row["question"] = row["question"].replace("cost", "price")
    if "identifier_key" in enabled:
        for item in state:
            item["candidate"] = item.pop("id")
    # Preserve original key order for factors that rename a key.
    keys = [
        "candidate" if "identifier_key" in enabled else "id",
        "price" if "field_name" in enabled else "cost",
        "risk",
        "time",
    ]
    row["state"] = json.dumps([{k: item[k] for k in keys} for item in state])
    row["base_id"] = row["id"]
    row["id"] = variant + ":" + row["id"]
    row["variant"] = variant
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previous-corpus", type=Path, required=True)
    args = parser.parse_args()
    rows = build(43, 128, "development_transfer")
    old_groups = set()
    for split in ("train", "validation"):
        old_groups.update(
            json.loads(line)["group"]
            for line in (args.previous_corpus / (split + ".jsonl")).read_text().splitlines()
        )
    if {r["group"] for r in rows} & old_groups:
        raise ValueError("underlying numeric state overlap")
    transformed = [transform(r, variant) for variant in VARIANTS for r in rows]
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / "inputs.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in transformed), encoding="utf-8")
    metadata = {
        "seed": 43,
        "states": 128,
        "questions_per_variant": 1024,
        "variants": VARIANTS,
        "rows": len(transformed),
        "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "purpose": "fixed-weight one-factor development diagnosis, not product audit",
        "previous_corpus_group_overlap": 0,
    }
    (args.output / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
