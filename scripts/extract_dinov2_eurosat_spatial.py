"""Extract pinned frozen DINOv2 features for EuroSAT spatial train/validation only."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tiny_omni_decision.dinov2_features import cls_patch_mean
from tiny_omni_decision.eurosat import eurosat_class_from_filename, validate_spatial_splits

ARCHIVE_SHA256 = "b4f5b234ecb7d7ff9c6cddb046543b4717c53fd6e9815be6c0e80cc614f51b90"
DINO_SHA256 = "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1"
SPLIT_HASHES = {
    "train": "2db7d455afb8dcbca898ea19a00f1f90c091734efdbba89e22aaf24056da243f",
    "validation": "6c758477604b7057a0fd990d7f6327b63b99a6725aac11a6a9d0174a7fdd8f0b",
    "test": "de22dec83d350cac3b3e4ca8e285cb6733c81ab94bf5bcf9213a567993402452",
}
TORCHGEO_SPLIT_REVISION = "1c11c73a87b40b0485d103231a97829991b8e22f"
CLASS_ORDER = (
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if sha256_file(args.archive) != ARCHIVE_SHA256:
        raise ValueError("EuroSAT RGB archive SHA-256 mismatch")
    split_paths = {
        "train": args.split_dir / "eurosat-spatial-train.txt",
        "validation": args.split_dir / "eurosat-spatial-val.txt",
        "test": args.split_dir / "eurosat-spatial-test.txt",
    }
    split_texts = {key: path.read_text(encoding="utf-8") for key, path in split_paths.items()}
    for split, path in split_paths.items():
        if sha256_file(path) != SPLIT_HASHES[split]:
            raise ValueError(f"pinned TorchGeo {split} split hash mismatch")
    weight_path = args.model / "model.safetensors"
    if sha256_file(weight_path) != DINO_SHA256:
        raise ValueError("pinned DINOv2 model weight SHA-256 mismatch")

    args.output.mkdir(parents=True)
    torchgeo_files = {key: split_texts[key] for key in split_texts}
    config = {
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "started_utc": datetime.now(UTC).isoformat(),
        "archive_sha256": ARCHIVE_SHA256,
        "torchgeo_split_revision": TORCHGEO_SPLIT_REVISION,
        "torchgeo_split_sha256": SPLIT_HASHES,
        "dino_revision": "facebook/dinov2-small@ed25f3a31f01632728cabb09d1542f84ab7b0056",
        "dino_weight_sha256": DINO_SHA256,
        "representation": "DINOv2 CLS token concatenated with mean of 256 patch tokens",
        "resize": [224, 224],
        "batch_size": args.batch_size,
        "feature_dimensions": 768,
        "class_order": CLASS_ORDER,
        "opened_splits": ["train", "validation"],
        "spatial_test_media_decoded": False,
        "hotgpu_reservation": {
            "reserved_mib": int(os.environ["TINY_OMNI_GPU_GATE_RESERVED_MIB"]),
            "free_mib_at_child_start": int(os.environ["TINY_OMNI_GPU_GATE_FREE_MIB"]),
            "total_mib": int(os.environ["TINY_OMNI_GPU_GATE_TOTAL_MIB"]),
        },
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "numpy", "Pillow")
        },
    }
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

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
    media_hashes: dict[str, str] = {}
    row_hashes: dict[str, str] = {}
    archive_records: dict[str, tuple[str, object]] = {}
    with ZipFile(args.archive) as zipped:
        expected_members = set()
        for info in zipped.infolist():
            if info.is_dir():
                continue
            member_path = Path(info.filename)
            if member_path.suffix.lower() != ".jpg":
                continue
            basename = member_path.name
            if basename in expected_members:
                raise ValueError(f"duplicate EuroSAT RGB basename: {basename}")
            expected_members.add(basename)
            label = eurosat_class_from_filename(basename)
            if label != member_path.parent.name:
                raise ValueError(f"class-folder mismatch for {info.filename}")
            archive_records[basename] = (info.filename, label)
        if len(expected_members) != 27_000:
            raise ValueError(f"expected 27,000 RGB images, found {len(expected_members)}")
        splits = validate_spatial_splits(torchgeo_files, expected_members=expected_members)
        split_manifest_sha = {split: sha256_file(split_paths[split]) for split in split_paths}
        split_counts = {split: len(rows) for split, rows in splits.items()}
        for split in ("train", "validation"):
            rows = splits[split]
            if not rows:
                raise ValueError(f"empty {split} split")
            label_counts = {name: 0 for name in CLASS_ORDER}
            for basename in rows:
                member, label = archive_records[basename]
                if label not in label_counts:
                    raise ValueError(f"unexpected class label: {label}")
                label_counts[label] += 1
            array_path = args.output / f"{split}-features.npy"
            array = np.lib.format.open_memmap(
                array_path, mode="w+", dtype=np.float32, shape=(len(rows), 768)
            )
            manifest_rows = []
            for offset in range(0, len(rows), args.batch_size):
                batch_names = rows[offset : offset + args.batch_size]
                images = []
                for basename in batch_names:
                    member, label = archive_records[basename]
                    raw = zipped.read(member)
                    media_hash = hashlib.sha256(raw).hexdigest()
                    media_hashes[basename] = media_hash
                    with Image.open(BytesIO(raw)) as image:
                        images.append(
                            image.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
                        )
                    manifest_rows.append(
                        {
                            "id": f"{label}/{basename}",
                            "filename": basename,
                            "label": label,
                            "archive_member": member,
                            "media_sha256": media_hash,
                        }
                    )
                encoded = processor(
                    images=images, do_resize=False, do_center_crop=False, return_tensors="pt"
                ).to(device)
                with torch.inference_mode():
                    tokens = model(**encoded).last_hidden_state
                features = cls_patch_mean(tokens.float().cpu().numpy())
                if not np.isfinite(features).all():
                    raise ValueError("non-finite DINOv2 feature")
                array[offset : offset + len(batch_names)] = features
                if offset % 1024 == 0 or offset + len(batch_names) == len(rows):
                    print(
                        json.dumps(
                            {
                                "split": split,
                                "completed": offset + len(batch_names),
                                "total": len(rows),
                                "elapsed_seconds": round(time.perf_counter() - started, 2),
                            }
                        ),
                        flush=True,
                    )
            array.flush()
            del array
            manifest_path = args.output / f"{split}-records.jsonl"
            manifest_path.write_text(
                "".join(
                    json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                    for row in manifest_rows
                ),
                encoding="utf-8",
            )
            row_hashes[split] = sha256_file(manifest_path)
            (args.output / f"{split}-class-counts.json").write_text(
                json.dumps(label_counts, sort_keys=True, indent=2) + "\n", encoding="utf-8"
            )
    if len(set(media_hashes.values())) != len(media_hashes):
        raise ValueError("duplicate image bytes found among opened train/validation media")
    media_manifest_path = args.output / "media-sha256.json"
    media_manifest_path.write_text(
        json.dumps(media_hashes, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    result = {
        **config,
        "finished_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "split_counts": split_counts,
        "split_manifest_file_sha256": split_manifest_sha,
        "train_records_sha256": row_hashes["train"],
        "validation_records_sha256": row_hashes["validation"],
        "unique_opened_media_sha256": len(set(media_hashes.values())),
        "opened_media_hashes_unique": len(set(media_hashes.values())) == len(media_hashes),
        "spatial_test_media_decoded": False,
        "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated(),
        "feature_file_sha256": {
            split: sha256_file(args.output / f"{split}-features.npy")
            for split in ("train", "validation")
        },
        "media_sha256_manifest_sha256": sha256_file(media_manifest_path),
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
