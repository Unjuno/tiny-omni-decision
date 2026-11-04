"""Extract admitted training screen crops with the unchanged frozen visual path."""

import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from scripts.range_generalization import sha, write
from scripts.train_dinov2_image_probe import WEIGHTS
from tiny_omni_decision.dinov2_features import cls_patch_mean

root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
source = root / "groundcua-train-expansion-images-01"
report = json.loads((source / "result.json").read_text())
for name, digest in report["artifacts"].items():
    assert sha(source / name) == digest
rows = json.loads((source / "admitted.json").read_text())
assert len(rows) == 906
model_path = Path("C:/CodexArtifacts/pretrained-reuse-r1/dinov2-small-audit-01/model")
assert sha(model_path / "model.safetensors") == WEIGHTS
out = root / "groundcua-train-expansion-features-01"
out.mkdir(exist_ok=False)
torch.set_num_threads(2)
start = time.perf_counter()
model = (
    AutoModel.from_pretrained(model_path, local_files_only=True, dtype=torch.float32)
    .cuda()
    .eval()
    .requires_grad_(False)
)
processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True)
values = []
with torch.inference_mode():
    for offset in range(0, len(rows), 16):
        crops = []
        for r in rows[offset : offset + 16]:
            with Image.open(source / f"{r['screen']:03d}.png") as image:
                crops.append(
                    image.convert("RGB")
                    .crop(r["bounds"])
                    .resize((224, 224), Image.Resampling.BICUBIC)
                )
        batch = processor(
            images=crops, do_resize=False, do_center_crop=False, return_tensors="pt"
        ).to("cuda")
        values.append(cls_patch_mean(model(**batch).last_hidden_state.cpu().numpy()))
features = np.concatenate(values)
assert features.shape == (906, 768) and np.isfinite(features).all()
np.save(out / "features.npy", features, allow_pickle=False)
assert np.array_equal(np.load(out / "features.npy"), features)
assert sha(model_path / "model.safetensors") == WEIGHTS
write(out / "records.json", rows)
write(
    out / "result.json",
    {
        "shape": list(features.shape),
        "seconds": time.perf_counter() - start,
        "peak_vram": torch.cuda.max_memory_allocated(),
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": 0,
        "model_sha256": WEIGHTS,
        "source_result_sha256": sha(source / "result.json"),
        "script_sha256": sha(Path(__file__)),
        "features_sha256": sha(out / "features.npy"),
        "records_sha256": sha(out / "records.json"),
        "reload_exact": True,
        "training": False,
    },
)
print((out / "result.json").read_text())
