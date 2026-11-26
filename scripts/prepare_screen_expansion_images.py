"""Acquire pinned training-only screen images and validate region admission."""

import hashlib
import io
import json
import math
from pathlib import Path

from PIL import Image

from scripts.audit_groundcua import fetch
from scripts.range_generalization import sha, write
from scripts.train_screen_pilot import CATEGORIES

root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
source = root / "groundcua-train-expansion-annotations-01"
report = json.loads((source / "result.json").read_text())
assert sha(source / "selection.json") == report["selection_sha256"]
old = json.loads((root / "groundcua-pilot-inputs-01/result.json").read_text())["records"]
old_regions = json.loads((root / "groundcua-classifier-01/records.json").read_text())
validation_pixels = {r["pixel_sha256"] for r in old if r["split"] == "validation"}
validation_crops = {r["crop_sha256"] for r in old_regions if r["split"] == "validation"}
out = root / "groundcua-train-expansion-images-01"
out.mkdir(exist_ok=False)
records = []
admitted = []
held = []
for i, r in enumerate(report["records"]):
    path = source / f"{i:03d}.json"
    assert sha(path) == r["sha256"]
    annotations = json.loads(path.read_text())
    images = {a["image_path"] for a in annotations}
    assert len(images) == 1
    image_path = "images/" + next(iter(images))
    assert image_path == r["path"].replace("data/", "images/", 1)[:-5] + ".png"
    raw = fetch(image_path)
    (out / f"{i:03d}.png").write_bytes(raw)
    with Image.open(io.BytesIO(raw)) as im:
        im = im.convert("RGB")
        pixel = hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest()
        assert pixel not in validation_pixels
        for a in annotations:
            category = a.get("category", "")
            box = a["bbox"]
            x1, y1, x2, y2 = box
            reason = None
            if not category:
                reason = "blank_category"
            elif category not in CATEGORIES:
                reason = "unknown_category"
            elif not all(math.isfinite(v) for v in box) or not (
                0 <= x1 < x2 <= im.width and 0 <= y1 < y2 <= im.height
            ):
                reason = "invalid_bbox"
            if reason:
                held.append({"screen": i, "id": a["id"], "reason": reason})
                continue
            crop = im.crop((math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2)))
            digest = hashlib.sha256(str(crop.size).encode() + crop.tobytes()).hexdigest()
            assert digest not in validation_crops
            admitted.append(
                {
                    "screen": i,
                    "app": r["app"],
                    "id": a["id"],
                    "category": category,
                    "bounds": [math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2)],
                    "crop_sha256": digest,
                }
            )
        records.append(
            {
                "screen": i,
                "app": r["app"],
                "image_path": image_path,
                "image_sha256": sha(out / f"{i:03d}.png"),
                "pixel_sha256": pixel,
                "size": list(im.size),
                "annotation_sha256": sha(path),
            }
        )
    write(out / "progress.json", records)
    print(i + 1, flush=True)
write(out / "records.json", records)
write(out / "admitted.json", admitted)
write(out / "held.json", held)
write(
    out / "result.json",
    {
        "screens": len(records),
        "admitted": len(admitted),
        "held": len(held),
        "cross_split_exact_pixel_overlap": 0,
        "cross_split_exact_crop_overlap": 0,
        "unique_admitted_crops": len({r["crop_sha256"] for r in admitted}),
        "source_result_sha256": sha(source / "result.json"),
        "script_sha256": sha(Path(__file__)),
        "artifacts": {p.name: sha(p) for p in out.iterdir() if p.is_file()},
        "training": False,
    },
)
