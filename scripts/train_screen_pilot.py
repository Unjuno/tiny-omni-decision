"""Bounded frozen visual region classifier; no text encoder or audit data."""

import hashlib
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
from scripts.train_sensor_subjects import metrics
from tiny_omni_decision.dinov2_features import cls_patch_mean

CATEGORIES = (
    "Button",
    "Menu",
    "Input Elements",
    "Navigation",
    "Sidebar",
    "Visual Elements",
    "Information Display",
    "Others",
)
ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")


def main():
    source = ROOT / "groundcua-pilot-inputs-01"
    output = ROOT / "groundcua-classifier-01"
    output.mkdir(exist_ok=False)
    manifest = json.loads((source / "result.json").read_text())
    model_path = Path("C:/CodexArtifacts/pretrained-reuse-r1/dinov2-small-audit-01/model")
    if sha(model_path / "model.safetensors") != WEIGHTS:
        raise ValueError("model hash mismatch")
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    encoder = (
        AutoModel.from_pretrained(model_path, local_files_only=True, dtype=torch.float32)
        .cuda()
        .eval()
        .requires_grad_(False)
    )
    processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True)
    features, records, skipped = [], [], []
    hashes = {"train": set(), "validation": set()}
    for screen in manifest["records"]:
        i = screen["local_index"]
        image_path, label_path = source / f"{i:03d}.png", source / f"{i:03d}.json"
        if (
            sha(image_path) != screen["image_sha256"]
            or sha(label_path) != screen["annotation_sha256"]
        ):
            raise ValueError("input hash mismatch")
        annotations = json.loads(label_path.read_text())
        crops = []
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            for row in annotations:
                category = row.get("category", "")
                if category and category not in CATEGORIES:
                    raise ValueError("unknown category")
                if not category or row["id"] in screen["invalid_bbox_ids"]:
                    skipped.append(
                        {
                            "id": row["id"],
                            "split": screen["split"],
                            "reason": "blank" if not category else "invalid_bbox",
                        }
                    )
                    continue
                x1, y1, x2, y2 = row["bbox"]
                crop = image.crop((math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2)))
                digest = hashlib.sha256(str(crop.size).encode() + crop.tobytes()).hexdigest()
                hashes[screen["split"]].add(digest)
                crops.append(crop.resize((224, 224), Image.Resampling.BICUBIC))
                records.append(
                    {
                        "id": row["id"],
                        "screen": i,
                        "app": screen["app"],
                        "split": screen["split"],
                        "target": CATEGORIES.index(category),
                        "crop_sha256": digest,
                    }
                )
        with torch.inference_mode():
            for offset in range(0, len(crops), 16):
                batch = processor(
                    images=crops[offset : offset + 16],
                    do_resize=False,
                    do_center_crop=False,
                    return_tensors="pt",
                ).to("cuda")
                features.append(cls_patch_mean(encoder(**batch).last_hidden_state.cpu().numpy()))
    values = np.concatenate(features)
    np.save(output / "features.npy", values, allow_pickle=False)
    write(output / "records.json", records)
    overlap = len(hashes["train"] & hashes["validation"])
    write(
        output / "preflight.json",
        {
            "skipped": skipped,
            "cross_split_crop_overlap": overlap,
            "unique_crops": {k: len(v) for k, v in hashes.items()},
        },
    )
    if overlap:
        raise ValueError("cross-split exact crop overlap; no fitting permitted")
    extraction_seconds = time.perf_counter() - started
    del encoder
    torch.cuda.empty_cache()
    train = np.array([r["split"] == "train" for r in records])
    targets = np.array([r["target"] for r in records])
    mean, std = values[train].mean(0), np.maximum(values[train].std(0), 1e-6)
    np.savez(output / "normalization.npz", mean=mean, std=std)
    x = torch.tensor((values - mean) / std)
    y = torch.tensor(targets)
    tx, ty, vx, vy = x[train], y[train], x[~train], y[~train]
    # Independent disposable training-only smoke before main initialization.
    torch.manual_seed(17)
    smoke = torch.nn.Linear(768, 8)
    loss = torch.nn.functional.cross_entropy(smoke(tx[:8]), ty[:8])
    loss.backward()
    if not all(torch.isfinite(p.grad).all() for p in smoke.parameters()):
        raise ValueError("smoke gradient failure")
    del smoke
    torch.manual_seed(17)
    model = torch.nn.Linear(768, 8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    generator = torch.Generator().manual_seed(17)
    history, best, best_epoch = [], float("inf"), 0
    write(
        output / "config.json",
        {
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "manifest_sha256": sha(source / "result.json"),
            "model_sha256": WEIGHTS,
            "seed": 17,
            "epochs": 30,
            "lr": 0.001,
            "weight_decay": 0.01,
            "batch": 64,
            "categories": CATEGORIES,
            "train": len(tx),
            "validation": len(vx),
            "torch": torch.__version__,
            "feature_sha256": sha(output / "features.npy"),
        },
    )
    for epoch in range(31):
        total = 0.0
        if epoch:
            for indices in torch.randperm(len(tx), generator=generator).split(64):
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.cross_entropy(model(tx[indices]), ty[indices])
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite loss")
                loss.backward()
                optimizer.step()
                total += loss.item() * len(indices)
        with torch.inference_mode():
            p = model(vx).double().softmax(-1).numpy()
        score = metrics(p, vy.numpy())
        history.append(
            {"epoch": epoch, "train_ce": total / len(tx) if epoch else None, "validation": score}
        )
        np.save(output / f"epoch-{epoch}.npy", p, allow_pickle=False)
        if score["nll"] < best:
            best, best_epoch = score["nll"], epoch
            torch.save(model.state_dict(), output / "best.pt")
    torch.save(model.state_dict(), output / "final.pt")
    model.load_state_dict(torch.load(output / "best.pt", weights_only=True))
    with torch.inference_mode():
        p = model(vx).double().softmax(-1).numpy()
    assert np.array_equal(p, np.load(output / f"epoch-{best_epoch}.npy"))
    prior = (np.bincount(ty.numpy(), minlength=8) + 1) / (len(ty) + 8)
    apps = np.array([r["app"] for r in records])[~train]
    result = {
        "best_epoch": best_epoch,
        "selected": metrics(p, vy.numpy()),
        "prior": metrics(np.tile(prior, (len(vy), 1)), vy.numpy()),
        "apps": {a: metrics(p[apps == a], vy.numpy()[apps == a]) for a in sorted(set(apps))},
        "exact_reload": True,
        "head_parameters": sum(p.numel() for p in model.parameters()),
        "encoder_parameters": 22056576,
        "extraction_seconds": extraction_seconds,
        "total_seconds": time.perf_counter() - started,
        "peak_allocated_vram": torch.cuda.max_memory_allocated(),
        "best_sha256": sha(output / "best.pt"),
        "final_sha256": sha(output / "final.pt"),
    }
    write(output / "history.json", history)
    write(output / "result.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
