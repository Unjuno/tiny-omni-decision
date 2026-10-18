"""Extract frozen DINOv2 features for indexed official-train Clevr-4 images."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import time
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from tiny_omni_decision.dinov2_features import cls_patch_mean, require_image_splits

WEIGHTS_SHA256 = "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def canonical_jsonl_sha256(rows: list[dict]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--media-hashed-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    weight_path = args.model / "model.safetensors"
    if sha256_file(weight_path) != WEIGHTS_SHA256:
        raise ValueError("pinned DINOv2 model weight SHA-256 mismatch")

    report = json.loads((args.index / "index-report.json").read_text(encoding="utf-8"))
    hashed_report_path = args.media_hashed_dir / "media-hash-report.json"
    if not hashed_report_path.is_file():
        raise ValueError("media-hash-report.json is required before feature extraction")
    hashed_report = json.loads(hashed_report_path.read_text(encoding="utf-8"))
    if hashed_report.get("status") != "MEDIA_HASHED":
        raise ValueError("media hashing has not completed")
    if hashed_report.get("official_val_media_opened") != 0:
        raise ValueError("official validation assets must remain untouched")

    manifests = {
        split: load_jsonl(args.media_hashed_dir / f"{split}-media-manifest.jsonl")
        for split in ("train", "validation")
    }
    labels = {
        split: load_jsonl(args.index / f"{split}-labels.jsonl") for split in ("train", "validation")
    }
    for split in manifests:
        if [row["id"] for row in manifests[split]] != [row["id"] for row in labels[split]]:
            raise ValueError(f"{split} labels and media order differ")
        label_name = f"{split}-labels.jsonl"
        if canonical_jsonl_sha256(labels[split]) != report["manifest_sha256"][label_name]:
            raise ValueError(f"{split} label manifest differs from the frozen index")
        expected_hash = hashed_report["media_manifest_sha256"][f"{split}-media-manifest.jsonl"]
        if sha256_file(args.media_hashed_dir / f"{split}-media-manifest.jsonl") != expected_hash:
            raise ValueError(f"{split} media manifest differs from the frozen index")
    require_image_splits(
        [{"id": row["id"], "sha256": row["media_sha256"]} for row in manifests["train"]],
        [{"id": row["id"], "sha256": row["media_sha256"]} for row in manifests["validation"]],
    )

    args.output.mkdir(parents=True)
    config = {
        "started_utc": datetime.now(UTC).isoformat(),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "archive_sha512": report["archive_sha512"],
        "index_report_sha256": sha256_file(args.index / "index-report.json"),
        "hashed_manifest_report_sha256": sha256_file(hashed_report_path),
        "model_weight_sha256": WEIGHTS_SHA256,
        "model_revision": "facebook/dinov2-small@ed25f3a31f01632728cabb09d1542f84ab7b0056",
        "model_path": str(args.model),
        "batch_size": args.batch_size,
        "hotgpu_reservation": {
            "reserved_mib": int(os.environ["TINY_OMNI_GPU_GATE_RESERVED_MIB"]),
            "free_mib_at_child_start": int(os.environ["TINY_OMNI_GPU_GATE_FREE_MIB"]),
            "total_mib": int(os.environ["TINY_OMNI_GPU_GATE_TOTAL_MIB"]),
        },
        "image_resize": [224, 224],
        "representation": "DINOv2 CLS token concatenated with mean of 256 patch tokens",
        "seed": 17,
        "official_val_media_opened": 0,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "numpy", "Pillow")
        },
    }
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    torch.set_num_threads(2)
    device = torch.device("cuda")
    model, info = AutoModel.from_pretrained(
        args.model, local_files_only=True, dtype=torch.float32, output_loading_info=True
    )
    if info.get("missing_keys") or info.get("unexpected_keys"):
        raise ValueError("DINOv2 checkpoint did not load exactly")
    model.eval().requires_grad_(False).to(device)
    processor = AutoImageProcessor.from_pretrained(args.model, local_files_only=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    file_hashes = {}
    with ZipFile(args.archive) as zipped:
        archive_names = set(zipped.namelist())
        for split in ("train", "validation"):
            rows = manifests[split]
            if not rows:
                raise ValueError(f"empty {split} split")
            output_path = args.output / f"{split}-features.npy"
            array = np.lib.format.open_memmap(
                output_path,
                mode="w+",
                dtype=np.float32,
                shape=(len(rows), 768),
            )
            ids = [row["id"] for row in rows]
            for offset in range(0, len(rows), args.batch_size):
                batch = rows[offset : offset + args.batch_size]
                images = []
                for row in batch:
                    member = row["archive_member"]
                    if member not in archive_names:
                        raise ValueError(f"missing indexed archive member: {member}")
                    with zipped.open(member) as stream:
                        raw = stream.read()
                    actual_hash = hashlib.sha256(raw).hexdigest()
                    if actual_hash != row["media_sha256"]:
                        raise ValueError(f"media hash mismatch for {row['id']}")
                    file_hashes[row["id"]] = actual_hash
                    with Image.open(BytesIO(raw)) as image:
                        images.append(
                            image.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
                        )
                encoded = processor(
                    images=images,
                    do_resize=False,
                    do_center_crop=False,
                    return_tensors="pt",
                ).to(device)
                with torch.inference_mode():
                    tokens = model(**encoded).last_hidden_state
                features = cls_patch_mean(tokens.float().cpu().numpy())
                if not np.isfinite(features).all():
                    raise ValueError("non-finite DINOv2 feature")
                array[offset : offset + len(batch)] = features
                if offset % 1024 == 0 or offset + len(batch) == len(rows):
                    print(
                        json.dumps(
                            {
                                "split": split,
                                "completed": offset + len(batch),
                                "total": len(rows),
                                "elapsed_seconds": round(time.perf_counter() - started, 2),
                            }
                        ),
                        flush=True,
                    )
            array.flush()
            (args.output / f"{split}-ids.json").write_text(
                json.dumps(ids, separators=(",", ":")) + "\n", encoding="utf-8"
            )
            del array

    hash_path = args.output / "feature-source-media-sha256.json"
    hash_path.write_text(json.dumps(file_hashes, sort_keys=True) + "\n", encoding="utf-8")
    result = {
        **config,
        "finished_utc": datetime.now(UTC).isoformat(),
        "train_images": len(manifests["train"]),
        "validation_images": len(manifests["validation"]),
        "unique_media_hashes": len(set(file_hashes.values())),
        "extracted_file_count": 0,
        "official_val_media_opened": 0,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated(),
        "feature_sha256": {
            name: sha256_file(args.output / name)
            for name in (
                "train-features.npy",
                "validation-features.npy",
                "train-ids.json",
                "validation-ids.json",
                "feature-source-media-sha256.json",
            )
        },
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
