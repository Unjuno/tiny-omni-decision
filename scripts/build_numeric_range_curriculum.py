"""Change only positive numeric range coverage; preserve questions and splits."""

import argparse
import copy
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from scripts.build_conditional_curriculum import build, verify


def affine(row, scale, offset):
    if scale <= 0:
        raise ValueError("order-preserving scale must be positive")
    row = copy.deepcopy(row)
    state = json.loads(row["state"])
    for item in state:
        for field in ("cost", "risk", "time"):
            item[field] = scale * item[field] + offset
    if "threshold" in row["oracle"]:
        old = row["oracle"]["threshold"]
        threshold = scale * old + offset
        row["oracle"]["threshold"] = threshold
        row["question"] = row["question"].replace(str(old), str(threshold))
    row["state"] = json.dumps(state)
    row["numeric_transform"] = {"scale": scale, "offset": offset}
    verify(row)
    return row


def apply_by_group(rows, transforms, seed):
    rng = random.Random(seed)
    groups = list(dict.fromkeys(r["group"] for r in rows))
    rng.shuffle(groups)
    mapping = {group: transforms[i % len(transforms)] for i, group in enumerate(groups)}
    return [affine(r, *mapping[r["group"]]) for r in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source / "train.jsonl"
    if hashlib.sha256(source.read_bytes()).hexdigest() != (
        "4d6110f4a401cfd6e50b9d0720a3a5fcc3aea146118ff7d1b0a05351764afbbe"
    ):
        raise ValueError("original training corpus hash mismatch")
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    train = apply_by_group(rows, [(1, 0), (2, 7), (4, 31), (8, 0)], 71)
    transfer = apply_by_group(
        build(59, 256, "development_transfer"), [(3, 11), (5, 23), (12, 11), (16, 23)], 73
    )
    old_groups = {r["group"] for r in rows}
    old_groups.update(
        json.loads(line)["group"]
        for line in (args.source / "validation.jsonl").read_text().splitlines()
    )
    old_groups.update(r["group"] for r in build(43, 128, "development_transfer"))
    if old_groups & {r["group"] for r in transfer}:
        raise ValueError("latent state overlap")
    if {r["state"] for r in train} & {r["state"] for r in transfer}:
        raise ValueError("rendered state overlap")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "purpose": "development range coverage; not final audit",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "train_transform_seed": 71,
        "transfer_state_seed": 59,
        "transfer_transform_seed": 73,
        "splits": {},
    }
    for name, split in (("train", train), ("transfer", transfer)):
        path = args.output / (name + ".jsonl")
        path.write_text("".join(json.dumps(r) + "\n" for r in split), encoding="utf-8")
        manifest["splits"][name] = {
            "rows": len(split),
            "groups": len({r["group"] for r in split}),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "transform_counts": dict(Counter(str(r["numeric_transform"]) for r in split)),
        }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
