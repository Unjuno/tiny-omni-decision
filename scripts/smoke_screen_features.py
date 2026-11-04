"""Frozen visual feature smoke on inspected development screens; no text inputs."""

import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from scripts.range_generalization import sha, write
from scripts.train_dinov2_image_probe import WEIGHTS
from tiny_omni_decision.dinov2_features import cls_patch_mean


def main():
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    source = root / "groundcua-source-audit-01"
    output = root / "groundcua-frozen-screen-smoke-01"
    model_path = Path("C:/CodexArtifacts/pretrained-reuse-r1/dinov2-small-audit-01/model")
    if sha(model_path / "model.safetensors") != WEIGHTS:
        raise ValueError("model hash mismatch")
    output.mkdir(exist_ok=False)
    source_report = json.loads((source / "result.json").read_text())
    images, records = [], []
    for index, sample in enumerate(source_report["samples"]):
        path = source / f"screen-{index}.png"
        annotations_path = source / f"annotation-{index}.json"
        if (sha(path) != sample["image_sha256"] or
                sha(annotations_path) != sample["annotation_sha256"]):
            raise ValueError("source hash mismatch")
        annotations = json.loads(annotations_path.read_text())
        with Image.open(path) as original:
            original = original.convert("RGB")
            # Full screens plus first8 annotated regions, including blank labels.
            images.append(original.resize((224, 224), Image.Resampling.BICUBIC))
            records.append({"screen": index, "region": None, "label": None})
            for region in annotations[:8]:
                x1, y1, x2, y2 = region["bbox"]
                bounds = (math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2))
                if not (0 <= bounds[0] < bounds[2] <= original.width and
                        0 <= bounds[1] < bounds[3] <= original.height):
                    raise ValueError("invalid crop")
                images.append(original.crop(bounds).resize((224, 224), Image.Resampling.BICUBIC))
                records.append({"screen": index, "region": region["id"],
                                "bounds": bounds, "label": region.get("category")})
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = AutoModel.from_pretrained(model_path, local_files_only=True,
                                     dtype=torch.float32).cuda().eval().requires_grad_(False)
    processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True)
    features = []
    with torch.inference_mode():
        for offset in range(0, len(images), 8):
            batch = processor(images=images[offset:offset + 8], do_resize=False,
                              do_center_crop=False, return_tensors="pt").to("cuda")
            features.append(cls_patch_mean(model(**batch).last_hidden_state.cpu().numpy()))
    values = np.concatenate(features)
    if not np.isfinite(values).all() or values.shape != (27, 768):
        raise ValueError("invalid feature output")
    np.save(output / "features.npy", values, allow_pickle=False)
    write(output / "records.json", records)
    result = {
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "script_sha256": sha(Path(__file__)), "source_report_sha256": sha(source / "result.json"),
        "model_files": {p.name: sha(p) for p in model_path.iterdir() if p.is_file()},
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "shape": list(values.shape), "seconds_including_model_load": time.perf_counter() - started,
        "peak_allocated_vram": torch.cuda.max_memory_allocated(), "torch": torch.__version__,
        "device": torch.cuda.get_device_name(), "dtype": "float32", "text_encoder": False,
        "preprocess": "outward integer crop; bicubic224x224; processor normalize only",
        "quality_measured": False, "feature_sha256": sha(output / "features.npy"),
    }
    write(output / "result.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
