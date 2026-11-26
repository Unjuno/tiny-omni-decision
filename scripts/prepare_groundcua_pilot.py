"""Freeze app-disjoint pilot before loading annotations; never load audit apps."""

import hashlib
import io
import json
import random
from collections import Counter
from pathlib import Path

from scripts.audit_groundcua import REVISION, fetch
from scripts.range_generalization import sha, write


def split_apps(apps):
    inspected = ["7-Zip", "Anki", "Audacity"]
    rest = sorted(set(apps) - set(inspected))
    if len(rest) != 84 or not set(inspected).issubset(apps):
        raise ValueError("unexpected platform inventory")
    random.Random(17).shuffle(rest)
    return {"train": inspected + rest[:57], "validation": rest[57:67],
            "sealed_audit": rest[67:]}


def main():
    from PIL import Image

    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    source = root / "groundcua-source-audit-01/inventory.json"
    inventory = json.loads(source.read_text())
    if inventory["sha"] != REVISION:
        raise ValueError("source revision mismatch")
    paths = sorted(s["rfilename"] for s in inventory["siblings"]
                   if s["rfilename"].startswith("data/"))
    apps = sorted({p.split("/")[1] for p in paths})
    splits = split_apps(apps)
    selected = []
    for split, count in (("train", 8), ("validation", 4)):
        for app in splits[split][:count]:
            candidates = [p for p in paths if p.startswith(f"data/{app}/")]
            if len(candidates) < 4:
                raise ValueError("insufficient screenshots")
            for path in candidates[:4]:
                selected.append({"split": split, "app": app, "annotation": path})
    output = root / "groundcua-pilot-inputs-01"
    output.mkdir(exist_ok=False)
    write(output / "frozen-selection.json", {
        "revision": REVISION, "inventory_sha256": sha(source), "seed": 17,
        "app_splits": splits, "selected": selected,
        "policy": "first4 lexical files, first8 train/4 validation apps; no replacements",
    })
    records = []
    seen_pixels = {"train": set(), "validation": set()}
    for i, selected_row in enumerate(selected):
        raw = fetch(selected_row["annotation"])
        annotations = json.loads(raw)
        image_paths = {r["image_path"] for r in annotations}
        if len(image_paths) != 1:
            raise ValueError("inconsistent pairing")
        image_path = "images/" + next(iter(image_paths))
        if image_path != selected_row["annotation"].replace("data/", "images/", 1)[:-5] + ".png":
            raise ValueError("unexpected image reference")
        pixels = fetch(image_path)
        (output / f"{i:03d}.png").write_bytes(pixels)
        (output / f"{i:03d}.json").write_bytes(raw)
        with Image.open(io.BytesIO(pixels)) as image:
            image = image.convert("RGB")
            digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
            size = image.size
        seen_pixels[selected_row["split"]].add(digest)
        if seen_pixels["train"] & seen_pixels["validation"]:
            raise ValueError("cross-split exact pixel duplicate")
        categories = Counter(r.get("category", "") for r in annotations)
        invalid = []
        for r in annotations:
            a, b, c, d = r["bbox"]
            if not (0 <= a < c <= size[0] and 0 <= b < d <= size[1]):
                invalid.append(r["id"])
        row = {**selected_row, "local_index": i, "image_path": image_path,
               "image_sha256": hashlib.sha256(pixels).hexdigest(), "pixel_sha256": digest,
               "annotation_sha256": hashlib.sha256(raw).hexdigest(), "size": size,
               "category_counts": dict(categories), "invalid_bbox_ids": invalid}
        records.append(row)
        write(output / "progress.json", records)
        print(json.dumps({"completed": len(records), "total": len(selected),
                          "app": row["app"], "labels": dict(categories)}), flush=True)
    write(output / "result.json", {"records": records, "exact_cross_split_pixel_overlap": 0,
                                   "selection_sha256": sha(output / "frozen-selection.json")})


if __name__ == "__main__":
    main()
