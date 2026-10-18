"""Evaluate one frozen selected DINOv2 EuroSAT readout once on the spatial test split."""

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
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import numpy as np
import torch
from PIL import Image
from torch import nn
from transformers import AutoImageProcessor, AutoModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tiny_omni_decision.clevr4_readout import classification_quality
from tiny_omni_decision.dinov2_features import cls_patch_mean
from tiny_omni_decision.eurosat import (
    eurosat_class_from_filename,
    require_hotgpu_reservation,
    validate_spatial_splits,
)

ARCHIVE_SHA256 = "b4f5b234ecb7d7ff9c6cddb046543b4717c53fd6e9815be6c0e80cc614f51b90"
DINO_SHA256 = "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1"
DINO_REVISION = "facebook/dinov2-small@ed25f3a31f01632728cabb09d1542f84ab7b0056"
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
SPLIT_HASHES = {
    "train": "2db7d455afb8dcbca898ea19a00f1f90c091734efdbba89e22aaf24056da243f",
    "validation": "6c758477604b7057a0fd990d7f6327b63b99a6725aac11a6a9d0174a7fdd8f0b",
    "test": "de22dec83d350cac3b3e4ca8e285cb6733c81ab94bf5bcf9213a567993402452",
}
TORCHGEO_REVISION = "1c11c73a87b40b0485d103231a97829991b8e22f"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def summarize(logits: torch.Tensor, targets: torch.Tensor) -> dict:
    probabilities = torch.softmax(logits.float(), dim=-1)
    result = classification_quality(probabilities, targets)
    predictions = logits.argmax(-1)
    recalls = []
    per_class = {}
    for index, name in enumerate(CLASS_ORDER):
        mask = targets.eq(index)
        support = int(mask.sum())
        accuracy = float(predictions[mask].eq(targets[mask]).float().mean()) if support else None
        per_class[name] = {"support": support, "accuracy": accuracy}
        if accuracy is not None:
            recalls.append(accuracy)
    result["macro_accuracy"] = float(np.mean(recalls))
    result["balanced_accuracy"] = result["macro_accuracy"]
    result["per_class"] = per_class
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--selected-run", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    gate_values = require_hotgpu_reservation(os.environ)
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    selected_result_path = args.selected_run / "result.json"
    selected_result = json.loads(selected_result_path.read_text(encoding="utf-8"))
    if (
        selected_result.get("result_status")
        != "TRAINING_50_EPOCHS_COMPLETED_BEST_RELOAD_VERIFIED_POSTHOC"
    ):
        raise ValueError("selected readout has not passed posthoc best reload verification")
    selected_checkpoint = args.selected_run / "best-readout.pt"
    if sha256_file(selected_checkpoint) != selected_result["best_readout_sha256"]:
        raise ValueError("selected best checkpoint hash mismatch")
    archive_sha = sha256_file(args.archive)
    if archive_sha != ARCHIVE_SHA256:
        raise ValueError("EuroSAT RGB archive SHA-256 mismatch")
    split_paths = {
        "train": args.split_dir / "eurosat-spatial-train.txt",
        "validation": args.split_dir / "eurosat-spatial-val.txt",
        "test": args.split_dir / "eurosat-spatial-test.txt",
    }
    split_texts = {split: path.read_text(encoding="utf-8") for split, path in split_paths.items()}
    split_sha = {split: sha256_file(path) for split, path in split_paths.items()}
    if split_sha != SPLIT_HASHES:
        raise ValueError("pinned TorchGeo spatial split hash mismatch")
    if sha256_file(args.model / "model.safetensors") != DINO_SHA256:
        raise ValueError("pinned DINOv2 model weight SHA-256 mismatch")

    args.output.mkdir(parents=True)
    torch.set_num_threads(2)
    device = torch.device("cuda")
    encoder, loading = AutoModel.from_pretrained(
        args.model, local_files_only=True, dtype=torch.float32, output_loading_info=True
    )
    if loading.get("missing_keys") or loading.get("unexpected_keys"):
        raise ValueError("DINOv2 checkpoint did not load exactly")
    encoder.eval().requires_grad_(False).to(device)
    processor = AutoImageProcessor.from_pretrained(args.model, local_files_only=True)
    readout_state = torch.load(selected_checkpoint, weights_only=True, map_location="cpu")
    readout = nn.Linear(768, len(CLASS_ORDER))
    readout.load_state_dict(readout_state["model"])
    readout.eval()
    mean, std = readout_state["mean"], readout_state["std"]

    feature_result_path = args.features / "result.json"
    if sha256_file(feature_result_path) != selected_result["feature_result_sha256"]:
        raise ValueError("feature source differs from the selected training run")
    train_records = [
        json.loads(line)
        for line in (args.features / "train-records.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    train_targets = torch.tensor([CLASS_ORDER.index(row["label"]) for row in train_records])
    train_counts = torch.bincount(train_targets, minlength=len(CLASS_ORDER)).float()
    train_prior_logits = train_counts.clamp_min(1).log().expand(5400, -1)
    started_utc = datetime.now(UTC).isoformat()
    config = {
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "started_utc": started_utc,
        "dataset_archive_sha256": archive_sha,
        "torchgeo_split_revision": TORCHGEO_REVISION,
        "test_split_manifest_sha256": split_sha["test"],
        "dino_revision": DINO_REVISION,
        "dino_weight_sha256": DINO_SHA256,
        "selected_run_result_sha256": sha256_file(selected_result_path),
        "selected_checkpoint_sha256": sha256_file(selected_checkpoint),
        "selected_epoch": selected_result["selected_epoch"],
        "prequantization_parameters": selected_result["prequantization_parameters"],
        "readout_trainable_parameters": selected_result["trainable_readout_parameters"],
        "test_examples": 5400,
        "resize": [224, 224],
        "batch_size": args.batch_size,
        "option_policy": "fixed ten-class EuroSAT option IDs; not arbitrary text options",
        "test_partition_public_not_blind": True,
        "test_was_not_used_for_checkpoint_selection": True,
        "feature_result_sha256": sha256_file(feature_result_path),
        "hotgpu_reservation": {
            "reserved_mib": gate_values["TINY_OMNI_GPU_GATE_RESERVED_MIB"],
            "free_mib_at_child_start": gate_values["TINY_OMNI_GPU_GATE_FREE_MIB"],
            "total_mib": gate_values["TINY_OMNI_GPU_GATE_TOTAL_MIB"],
        },
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "numpy", "Pillow")
        },
    }
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    opened_media = {}
    records = []
    feature_path = args.output / "spatial-test-features.npy"
    with ZipFile(args.archive) as zipped:
        archive_infos = [info for info in zipped.infolist() if not info.is_dir()]
        archive_members = {}
        expected_names = set()
        for info in archive_infos:
            path = PurePosixPath(info.filename)
            if path.suffix.lower() != ".jpg":
                continue
            if path.name in expected_names:
                raise ValueError(f"duplicate image basename: {path.name}")
            expected_names.add(path.name)
            label = eurosat_class_from_filename(path.name)
            if label != path.parent.name:
                raise ValueError(f"class folder mismatch: {info.filename}")
            archive_members[path.name] = (info.filename, label)
        splits = validate_spatial_splits(split_texts, expected_members=expected_names)
        test_names = splits["test"]
        if len(test_names) != 5400:
            raise ValueError(f"expected 5,400 spatial-test images, found {len(test_names)}")
        features = np.lib.format.open_memmap(
            feature_path, mode="w+", dtype=np.float32, shape=(len(test_names), 768)
        )
        for offset in range(0, len(test_names), args.batch_size):
            names = test_names[offset : offset + args.batch_size]
            images = []
            batch_records = []
            for name in names:
                member, label = archive_members[name]
                raw = zipped.read(member)
                media_hash = hashlib.sha256(raw).hexdigest()
                opened_media[name] = media_hash
                with Image.open(BytesIO(raw)) as image:
                    images.append(image.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC))
                batch_records.append(
                    {
                        "id": f"{label}/{name}",
                        "filename": name,
                        "label": label,
                        "archive_member": member,
                        "media_sha256": media_hash,
                    }
                )
            encoded = processor(
                images=images, do_resize=False, do_center_crop=False, return_tensors="pt"
            ).to(device)
            with torch.inference_mode():
                tokens = encoder(**encoded).last_hidden_state
            batch_features = cls_patch_mean(tokens.float().cpu().numpy())
            if not np.isfinite(batch_features).all():
                raise ValueError("non-finite spatial-test features")
            features[offset : offset + len(names)] = batch_features
            records.extend(batch_records)
            if offset % 1024 == 0 or offset + len(names) == len(test_names):
                print(
                    json.dumps(
                        {
                            "completed": offset + len(names),
                            "total": len(test_names),
                            "elapsed_seconds": round(time.perf_counter() - started, 2),
                        }
                    ),
                    flush=True,
                )
        features.flush()
        del features

    if len(set(opened_media.values())) != len(opened_media):
        raise ValueError("duplicate test media bytes found")
    test_array = np.load(feature_path, mmap_mode="r")
    test_tensor = torch.from_numpy(np.asarray(test_array).copy())
    with torch.inference_mode():
        logits = readout((test_tensor - mean) / std)
    targets = torch.tensor([CLASS_ORDER.index(row["label"]) for row in records])
    test_metrics = summarize(logits, targets)
    train_prior_metrics = summarize(train_prior_logits, targets)
    probabilities = torch.softmax(logits.float(), dim=-1)
    prediction_path = args.output / "spatial-test-predictions.jsonl"
    with prediction_path.open("w", encoding="utf-8", newline="\n") as stream:
        for row, probs in zip(records, probabilities.tolist(), strict=True):
            pred = CLASS_ORDER[int(np.argmax(probs))]
            stream.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "target": row["label"],
                        "prediction": pred,
                        "options": CLASS_ORDER,
                        "probabilities": probs,
                        "media_sha256": row["media_sha256"],
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
    record_path = args.output / "spatial-test-records.jsonl"
    record_path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in records),
        encoding="utf-8",
    )
    result = {
        **config,
        "finished_utc": datetime.now(UTC).isoformat(),
        "metrics": test_metrics,
        "train_frequency_prior_metrics": train_prior_metrics,
        "media_manifest_sha256": sha256_file(args.output / "spatial-test-records.jsonl"),
        "feature_sha256": sha256_file(feature_path),
        "predictions_sha256": sha256_file(prediction_path),
        "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
