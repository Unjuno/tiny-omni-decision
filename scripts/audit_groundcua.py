"""Pinned three-platform source audit; these examples are development-only."""

import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from urllib.parse import quote

import requests
from PIL import Image

from scripts.range_generalization import write

REVISION = "5d6845b0116029d46ec762e734701c5b8ce207c3"
REPO = "ServiceNow/GroundCUA"


def fetch(path):
    url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{quote(path)}"
    response = requests.get(url, timeout=45)
    response.raise_for_status()
    if len(response.content) > 20_000_000:
        raise ValueError("audit file exceeds budget")
    return response.content


def main():
    output = Path("C:/CodexArtifacts/pretrained-reuse-r2/groundcua-source-audit-01")
    output.mkdir(exist_ok=False)
    response = requests.get(
        f"https://huggingface.co/api/datasets/{REPO}/revision/{REVISION}", timeout=45
    )
    response.raise_for_status()
    inventory = response.json()
    if inventory["sha"] != REVISION or inventory["cardData"]["license"] != "mit":
        raise ValueError("revision or license changed")
    write(output / "inventory.json", inventory)
    readme = fetch("README.md")
    (output / "README.md").write_bytes(readme)
    filenames = sorted(s["rfilename"] for s in inventory["siblings"])
    selected = [next(p for p in filenames if p.startswith(f"data/{app}/"))
                for app in ("7-Zip", "Anki", "Audacity")]
    # Freeze sample identities before loading any new annotation content.
    write(output / "selected.json", {"paths": selected, "role": "development-only"})
    rows = []
    for index, path in enumerate(selected):
        raw = fetch(path)
        (output / f"annotation-{index}.json").write_bytes(raw)
        annotations = json.loads(raw)
        image_paths = {r["image_path"] for r in annotations}
        if len(image_paths) != 1:
            raise ValueError("inconsistent image pairing")
        image_path = "images/" + next(iter(image_paths))
        if image_path not in filenames:
            raise ValueError("image missing from pinned inventory")
        pixels = fetch(image_path)
        digest = hashlib.sha256(pixels).hexdigest()
        (output / f"screen-{index}.png").write_bytes(pixels)
        with Image.open(io.BytesIO(pixels)) as image:
            image.load()
            width, height = image.size
        invalid = []
        for r in annotations:
            x1, y1, x2, y2 = r["bbox"]
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                invalid.append(r["id"])
        rows.append({
            "annotation_path": path, "annotation_sha256": hashlib.sha256(raw).hexdigest(),
            "image_path": image_path, "image_sha256": digest,
            "filename_matches_byte_sha256": Path(image_path).stem == digest,
            "size": [width, height], "annotations": len(annotations),
            "category_counts": dict(Counter(r.get("category", "MISSING") for r in annotations)),
            "invalid_bbox_ids": invalid,
        })
    result = {"repo": REPO, "revision": REVISION, "declared_license": "MIT",
              "readme_sha256": hashlib.sha256(readme).hexdigest(), "samples": rows,
              "training_or_evaluation_performed": False,
              "exclusion": "All three inspected screenshots must be development-only"}
    write(output / "result.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
