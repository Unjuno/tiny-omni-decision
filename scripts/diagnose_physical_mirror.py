"""Training-only physical mirror diagnostic with frozen models."""

import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import AutoImageProcessor, AutoModel

from scripts.report_spatial_curve import read
from scripts.train_dinov2_spatial_control import (
    CLEVR4_TAXONOMIES,
    HASHES,
    PARENT_HASHES,
    TEXT,
    WEIGHTS,
    BilinearCandidateScorer,
    _option_embeddings,
    _prediction_rows,
    _quality_metrics,
    _taxonomy_metrics,
    readout_features,
    sha,
    write,
)
from tiny_omni_decision.dinov2_features import spatial_cells


def main():
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    source = root / "dinov2-spatial-bilinear-01"
    cache = root / "dinov2-spatial-control-02/train-cells.npy"
    labels = root / "clevr4-full-coverage-seed17/train-labels.jsonl"
    text = root / "path-a-audio-multiquery-runtime-bundle-v2/models/text-minilm-en"
    checkpoint = source / "spatial/best.pt"
    out = root / "dinov2-physical-mirror-01"
    if out.exists():
        raise FileExistsError(out)
    assert sha(cache) == PARENT_HASHES["train-cells.npy"]
    assert sha(labels) == HASHES[labels.name]
    assert sha(text / "model.safetensors") == TEXT
    assert sha(checkpoint) == read(source / "result.json")["conditions"]["spatial"]["best_sha256"]
    device = torch.device("cuda")
    torch.set_num_threads(2)
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    cells = np.load(cache, allow_pickle=False)[:64]
    rows = [json.loads(x) for x in labels.read_text().splitlines()][:64]
    manifest = labels.with_name("train-image-manifest.jsonl")
    assert sha(manifest) == HASHES[manifest.name]
    media = [json.loads(x) for x in manifest.read_text().splitlines()][:64]
    assert [r["id"] for r in media] == [r["id"] for r in rows]
    model_path = Path("C:/CodexArtifacts/pretrained-reuse-r1/dinov2-small-audit-01/model")
    assert sha(model_path / "model.safetensors") == WEIGHTS
    visual_encoder = (
        AutoModel.from_pretrained(model_path, local_files_only=True, dtype=torch.float32)
        .to(device)
        .eval()
    )
    visual_encoder.requires_grad_(False)
    processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True)
    extracted = {}
    with torch.inference_mode():
        for condition in ["identity", "physical_mirror"]:
            parts = []
            for offset in range(0, 64, 32):
                images = []
                for r in media[offset : offset + 32]:
                    path = labels.parent / "train/images" / (r["id"] + ".png")
                    assert sha(path) == r["sha256"]
                    with Image.open(path) as im:
                        image = im.convert("RGB")
                        if condition == "physical_mirror":
                            image = ImageOps.mirror(image)
                        images.append(image.resize((224, 224), Image.Resampling.BICUBIC))
                inputs = processor(
                    images=images, do_resize=False, do_center_crop=False, return_tensors="pt"
                ).to(device)
                parts.append(
                    spatial_cells(visual_encoder(**inputs).last_hidden_state.cpu().numpy())
                )
            extracted[condition] = np.concatenate(parts)
    parity = float(np.abs(extracted["identity"] - cells).max())
    assert parity <= 1e-5, parity
    del visual_encoder
    extracted["cell_reversal"] = np.concatenate(
        [cells[:, :1], cells[:, 1:][:, np.arange(16).reshape(4, 4)[:, ::-1].ravel()]], axis=1
    )
    targets = [opts.index(str(r[t])) for r in rows for t, opts in CLEVR4_TAXONOMIES.items()]
    types = list(CLEVR4_TAXONOMIES) * len(rows)
    embeddings, encoder = _option_embeddings(device, text)
    options = torch.stack(
        [
            torch.stack([embeddings[f"{t}|{o}"] for o in opts])
            for t, opts in CLEVR4_TAXONOMIES.items()
        ]
    )
    options = options.to(device)
    del encoder, embeddings
    model = BilinearCandidateScorer(6528, 384, 128).to(device).eval()
    model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cpu"))
    model.requires_grad_(False)
    out.mkdir()
    results, predictions = {}, {}
    with torch.inference_mode():
        for name, changed in extracted.items():
            np.save(out / (name + "-cells.npy"), changed)
            features = torch.from_numpy(readout_features(changed, "spatial")).to(device)
            logits = []
            for offset in range(0, len(targets), 128):
                ids = torch.arange(offset, min(offset + 128, len(targets)), device=device)
                visual = features[ids // 4, None].expand(-1, 10, -1)
                logits.extend(model(torch.cat([visual, options[ids % 4]], -1)).cpu().unbind())
            pred = _prediction_rows(rows, logits, targets)
            predictions[name] = pred
            write(out / (name + "-predictions.json"), pred)
            results[name] = dict(
                metrics=_quality_metrics(logits, targets),
                tasks=_taxonomy_metrics(logits, targets, types),
            )
    result = dict(
        results=results,
        original_cache_max_abs=parity,
        split="train",
        unique_images=64,
        questions=256,
        peak_vram_bytes=torch.cuda.max_memory_allocated(),
        seconds=time.perf_counter() - start,
        device="cuda",
        training=False,
        hashes={str(p): sha(p) for p in [cache, labels, checkpoint, Path(__file__)]},
    )
    write(out / "result.json", result)
    write(Path("docs/pretrained_reuse/R2_PHYSICAL_MIRROR_RESULTS_20261012.json"), result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
