"""Extract fixed full-screen context without text, labels or encoder training."""

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
source = root / "groundcua-pilot-inputs-01"
out = root / "groundcua-fullscreen-features-01"
out.mkdir(exist_ok=False)
records = json.loads((source / "result.json").read_text())["records"]
assert len(records) == 48 and {r["split"] for r in records} == {"train", "validation"}
model_path = Path("C:/CodexArtifacts/pretrained-reuse-r1/dinov2-small-audit-01/model")
assert sha(model_path / "model.safetensors") == WEIGHTS
images = []
for r in records:
    path = source / f"{r['local_index']:03d}.png"
    assert sha(path) == r["image_sha256"]
    with Image.open(path) as im:
        images.append(im.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC))
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
    for i in range(0, len(images), 8):
        token = processor(
            images=images[i : i + 8], do_resize=False, do_center_crop=False, return_tensors="pt"
        ).to("cuda")
        values.append(cls_patch_mean(model(**token).last_hidden_state.cpu().numpy()))
    check = processor(
        images=images[:8], do_resize=False, do_center_crop=False, return_tensors="pt"
    ).to("cuda")
    repeated = cls_patch_mean(model(**check).last_hidden_state.cpu().numpy())
values = np.concatenate(values)
assert values.shape == (48, 768) and np.isfinite(values).all()
assert np.array_equal(repeated, values[:8])
np.save(out / "features.npy", values, allow_pickle=False)
assert np.array_equal(np.load(out / "features.npy"), values)
assert sha(model_path / "model.safetensors") == WEIGHTS
write(
    out / "records.json",
    [
        {"screen": r["local_index"], "split": r["split"], "image_sha256": r["image_sha256"]}
        for r in records
    ],
)
write(
    out / "result.json",
    {
        "shape": list(values.shape),
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": 0,
        "seconds": time.perf_counter() - start,
        "peak_vram": torch.cuda.max_memory_allocated(),
        "weight_sha256": WEIGHTS,
        "source_manifest_sha256": sha(source / "result.json"),
        "script_sha256": sha(Path(__file__)),
        "features_sha256": sha(out / "features.npy"),
        "records_sha256": sha(out / "records.json"),
        "repeated_forward_exact": True,
        "reload_exact": True,
        "preprocessing": (
            "full screen bicubic224x224, normalize only; no crop; no annotation inputs"
        ),
        "quality_measured": False,
    },
)
print((out / "result.json").read_text())
