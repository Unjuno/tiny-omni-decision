"""Extract pinned physical-mirror training cells without touching validation."""

import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import AutoImageProcessor, AutoModel

from scripts.train_dinov2_spatial_control import HASHES, WEIGHTS, sha, write
from tiny_omni_decision.dinov2_features import spatial_cells


def main():
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    directory = root / "clevr4-full-coverage-seed17"
    out = root / "dinov2-train-mirror-cache-01"
    model_path = Path("C:/CodexArtifacts/pretrained-reuse-r1/dinov2-small-audit-01/model")
    if out.exists():
        raise FileExistsError(out)
    manifests = {k: directory / ("train-" + k + ".jsonl") for k in ["labels", "image-manifest"]}
    for p in manifests.values():
        assert sha(p) == HASHES[p.name]
    assert sha(model_path / "model.safetensors") == WEIGHTS
    rows = [json.loads(x) for x in manifests["image-manifest"].read_text().splitlines()]
    labels = [json.loads(x) for x in manifests["labels"].read_text().splitlines()]
    assert len(rows) == 8424 and [r["id"] for r in rows] == [r["id"] for r in labels]
    required = 8424 * 17 * 384 * 4
    free = shutil.disk_usage(root).free
    if free < required + 1024**3:
        raise OSError("insufficient disk for cache plus1GiB reserve")
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    model = (
        AutoModel.from_pretrained(model_path, local_files_only=True, dtype=torch.float32)
        .to("cuda")
        .eval()
    )
    model.requires_grad_(False)
    processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True)
    out.mkdir()
    config = dict(
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        script_sha256=sha(Path(__file__)),
        model_sha256=WEIGHTS,
        model_files={p.name: sha(p) for p in model_path.iterdir() if p.is_file()},
        input_hashes={p.name: sha(p) for p in manifests.values()},
        ids=[r["id"] for r in rows],
        split="train",
        batch_size=32,
        preprocessing="RGB physical mirror then224x224 bicubic; no crop",
        dtype="float32",
        torch=torch.__version__,
        free_disk_before=free,
    )
    write(out / "config.json", config)
    features = np.lib.format.open_memmap(
        out / "train-mirror-cells.npy", mode="w+", dtype=np.float32, shape=(8424, 17, 384)
    )
    with torch.inference_mode():
        for offset in range(0, len(rows), 32):
            images = []
            for row in rows[offset : offset + 32]:
                path = directory / "train/images" / (row["id"] + ".png")
                assert sha(path) == row["sha256"]
                with Image.open(path) as im:
                    images.append(
                        ImageOps.mirror(im.convert("RGB")).resize(
                            (224, 224), Image.Resampling.BICUBIC
                        )
                    )
            inputs = processor(
                images=images, do_resize=False, do_center_crop=False, return_tensors="pt"
            ).to("cuda")
            features[offset : offset + len(images)] = spatial_cells(
                model(**inputs).last_hidden_state.cpu().numpy()
            )
            if offset % 1024 == 0:
                print(offset + len(images), flush=True)
    features.flush()
    prior = np.load(
        root / "dinov2-physical-mirror-01/physical_mirror-cells.npy", allow_pickle=False
    )
    parity = float(np.abs(features[:64] - prior).max())
    assert parity == 0, parity
    assert np.isfinite(features).all()
    assert sha(model_path / "model.safetensors") == WEIGHTS
    result = dict(
        status="PASS_EXTRACTION_ONLY",
        images=8424,
        unique_original_assets=8424,
        augmented_views=8424,
        new_independent_assets=0,
        first64_exact=True,
        feature_sha256=sha(out / "train-mirror-cells.npy"),
        bytes=required,
        seconds=time.perf_counter() - start,
        peak_vram_bytes=torch.cuda.max_memory_allocated(),
        gpu=torch.cuda.get_device_name(),
        cost_usd=0,
        training=False,
        free_disk_after=shutil.disk_usage(root).free,
    )
    write(out / "result.json", result)
    write(Path("docs/pretrained_reuse/R2_MIRROR_CACHE_RESULTS_20261012.json"), result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
