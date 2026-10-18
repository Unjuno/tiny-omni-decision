"""Freeze subject groups from archive metadata, without reading observations."""

import json
import random
import re
from pathlib import Path

from scripts.range_generalization import sha, write


def split_inventory(inventory):
    records = []
    for item in inventory:
        match = re.fullmatch(
            r"dataset/videoandimus/(S\d+)/(S\d+)_(A\d+)_(T\d+)\.raw", item["name"]
        )
        if match:
            folder, subject, activity, trial = match.groups()
            if folder != subject:
                raise ValueError("subject identity mismatch")
            records.append({"subject": subject, "activity": activity, "trial": trial,
                            "member": item["name"]})
    subjects = sorted({r["subject"] for r in records})
    if len(subjects) != 16 or len(records) != 208:
        raise ValueError("unexpected pinned inventory coverage")
    if len({r["member"] for r in records}) != len(records):
        raise ValueError("duplicate record")
    for subject in subjects:
        activities = [r["activity"] for r in records if r["subject"] == subject]
        if sorted(activities) != [f"A{i:02}" for i in range(1, 14)]:
            raise ValueError("unexpected activity coverage")
    random.Random(17).shuffle(subjects)
    groups = {"train": subjects[:10], "validation": subjects[10:13],
              "sealed_audit": subjects[13:]}
    return {"seed": 17, "groups": groups, "records": {
        split: sorted((r for r in records if r["subject"] in members),
                      key=lambda r: r["member"])
        for split, members in groups.items()
    }}


def main():
    source = Path("C:/CodexArtifacts/pretrained-reuse-r2/vidimu-source-audit-02")
    inventory = source / "dataset.zip.inventory.json"
    result = json.loads((source / "result.json").read_text(encoding="utf-8"))
    expected = "605b695dcec5c2e7c4fd9a81dda22b60049697063d9eda150a637166813b79c2"
    if result["archive_sha256"] != expected or sha(source / "dataset.zip") != expected:
        raise ValueError("archive pin mismatch")
    split = split_inventory(json.loads(inventory.read_text(encoding="utf-8")))
    split.update({"archive_sha256": expected, "inventory_sha256": sha(inventory),
                  "payloads_read": False, "synchronization_verified": False})
    out = Path("C:/CodexArtifacts/pretrained-reuse-r2/vidimu-frozen-split-01")
    out.mkdir(exist_ok=False)
    write(out / "split.json", split)
    write(out / "hash.json", {"sha256": sha(out / "split.json")})
    print(json.dumps({"groups": split["groups"], "sha256": sha(out / "split.json")}))


if __name__ == "__main__":
    main()
