"""Fixed zero-shot screen category diagnostic with audited TinyCLIP."""

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from scripts.range_generalization import sha, write
from scripts.train_screen_pilot import CATEGORIES
from scripts.train_sensor_subjects import metrics

root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
source = root / "groundcua-pilot-inputs-01"
rows = [
    r
    for r in json.loads((root / "groundcua-classifier-01/records.json").read_text())
    if r["split"] == "validation"
]
assert len(rows) == 723
screens = {r["local_index"]: r for r in json.loads((source / "result.json").read_text())["records"]}
model_path = Path("C:/CodexArtifacts/pretrained-reuse-r1/tinyclip-39m19m-audit-01/model")
assert (
    sha(model_path / "model.safetensors")
    == "112bdfb07ebc8994e9afba7a2a314e278493a675033595cf48e54cf8a17ae587"
)
out = root / "tinyclip-screen-zero-shot-01"
out.mkdir(exist_ok=False)
prompts = [f"A screenshot of a user interface element: {name}." for name in CATEGORIES]
write(
    out / "protocol.json",
    {
        "prompts": prompts,
        "rows_sha256": sha(root / "groundcua-classifier-01/records.json"),
        "model_sha256": sha(model_path / "model.safetensors"),
        "script_sha256": sha(Path(__file__)),
        "selection": False,
        "calibration": False,
    },
)
crops = []
for screen in sorted({r["screen"] for r in rows}):
    assert sha(source / f"{screen:03d}.png") == screens[screen]["image_sha256"]
    assert sha(source / f"{screen:03d}.json") == screens[screen]["annotation_sha256"]
annotations = {
    screen: {a["id"]: a for a in json.loads((source / f"{screen:03d}.json").read_text())}
    for screen in {r["screen"] for r in rows}
}
for r in rows:
    a = annotations[r["screen"]][r["id"]]
    x1, y1, x2, y2 = a["bbox"]
    with Image.open(source / f"{r['screen']:03d}.png") as image:
        crop = image.convert("RGB").crop(
            (math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2))
        )
        assert (
            hashlib.sha256(str(crop.size).encode() + crop.tobytes()).hexdigest() == r["crop_sha256"]
        )
        crops.append(crop)
torch.set_num_threads(4)
start = time.perf_counter()
model = (
    CLIPModel.from_pretrained(model_path, local_files_only=True).cuda().eval().requires_grad_(False)
)
processor = CLIPProcessor.from_pretrained(model_path, local_files_only=True)
scores = []
with torch.inference_mode():
    tokens = processor(text=prompts, padding=True, return_tensors="pt").to("cuda")
    text = torch.nn.functional.normalize(
        model.text_projection(model.text_model(**tokens).pooler_output), dim=-1
    )
    for i in range(0, len(crops), 8):
        batch = processor(images=crops[i : i + 8], return_tensors="pt").to("cuda")
        visual = torch.nn.functional.normalize(
            model.visual_projection(model.vision_model(**batch).pooler_output), dim=-1
        )
        scores.append((model.logit_scale.exp() * visual @ text.T).cpu())
logits = torch.cat(scores)
assert torch.isfinite(logits).all()
p = logits.double().softmax(-1).numpy()
y = np.array([r["target"] for r in rows])
np.save(out / "probabilities.npy", p)
write(out / "rows.json", rows)
write(
    out / "result.json",
    {
        "metrics": metrics(p, y),
        "rotated_crop_control": metrics(np.roll(p, 1, axis=0), y),
        "seconds": time.perf_counter() - start,
        "peak_vram": torch.cuda.max_memory_allocated(),
        "parameters": sum(p.numel() for p in model.parameters()),
        "artifacts": {p.name: sha(p) for p in out.iterdir() if p.is_file()},
    },
)
print(metrics(p, y)["accuracy"])
