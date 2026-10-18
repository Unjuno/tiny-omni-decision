"""Create an isolated DINOv2 CLS + 4x4 patch-cell cache for the frozen Clevr-4 train split."""

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

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))
sys.path.insert(0, str(REPOSITORY / "src"))

from scripts.hotgpu_preflight import validate_gpu_preflight  # noqa: E402
from tiny_omni_decision.dinov2_features import require_image_splits, spatial_cells  # noqa: E402

WEIGHTS_SHA256 = "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1"
ARCHIVE_SHA512 = (
    "7e8454f9f44d7b527eb20252a687b15b4cd28f3e9c79cf6147317971034c367b"
    "64bf6bba98d2a66f64b26f1e7eea2105a8acf221198342e89a844fd0007cc94c"
)


def sha(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def require_gpu_slot() -> dict[str, int | str]:
    usage = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,utilization.gpu,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    memory, utilization, total = [int(value.strip()) for value in usage.split(",")]
    pid_output = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True,
    )
    pids = {
        int(line.strip())
        for line in pid_output.splitlines()
        if line.strip() and line.strip().casefold() != "no running processes found"
    }
    return validate_gpu_preflight(
        memory_used_mib=memory,
        utilization_pct=utilization,
        total_mib=total,
        active_pids=pids,
        own_pid=os.getpid(),
        environment=os.environ,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "index", "media-hashed-dir", "model", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if sha(args.archive, "sha512") != ARCHIVE_SHA512:
        raise ValueError("archive hash mismatch")
    if sha(args.model / "model.safetensors") != WEIGHTS_SHA256:
        raise ValueError("DINOv2 weight hash mismatch")
    index_report = json.loads((args.index / "index-report.json").read_text(encoding="utf-8"))
    media_report = json.loads(
        (args.media_hashed_dir / "media-hash-report.json").read_text(encoding="utf-8")
    )
    if index_report["archive_sha512"] != ARCHIVE_SHA512:
        raise ValueError("index archive does not match pinned source")
    if (
        media_report.get("status") != "MEDIA_HASHED"
        or media_report.get("official_val_media_opened") != 0
    ):
        raise ValueError("media integrity gate failed or official validation was opened")
    manifests = {
        s: rows(args.media_hashed_dir / f"{s}-media-manifest.jsonl")
        for s in ("train", "validation")
    }
    labels = {s: rows(args.index / f"{s}-labels.jsonl") for s in manifests}
    for split in manifests:
        if [r["id"] for r in labels[split]] != [r["id"] for r in manifests[split]]:
            raise ValueError(f"{split} label/media ID mismatch")
        name = f"{split}-media-manifest.jsonl"
        if sha(args.media_hashed_dir / name) != media_report["media_manifest_sha256"][name]:
            raise ValueError(f"{split} media manifest hash mismatch")
    require_image_splits(
        [{"id": r["id"], "sha256": r["media_sha256"]} for r in manifests["train"]],
        [{"id": r["id"], "sha256": r["media_sha256"]} for r in manifests["validation"]],
    )
    if (
        media_report.get("train_validation_media_overlap") != 0
        or media_report.get("overlap_with_previous_train_media") != 0
    ):
        raise ValueError("media identity overlap gate failed")
    if args.smoke:
        manifests = {split: items[:256] for split, items in manifests.items()}
    gpu_slot = require_gpu_slot()
    args.output.mkdir(parents=True)
    model = AutoModel.from_pretrained(args.model, local_files_only=True, dtype=torch.float32).eval()
    model.requires_grad_(False).to("cuda")
    processor = AutoImageProcessor.from_pretrained(args.model, local_files_only=True)
    config = {
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "started_utc": datetime.now(UTC).isoformat(),
        "archive_sha512": ARCHIVE_SHA512,
        "index_report_sha256": sha(args.index / "index-report.json"),
        "media_report_sha256": sha(args.media_hashed_dir / "media-hash-report.json"),
        "model_weight_sha256": WEIGHTS_SHA256,
        "model_revision": "facebook/dinov2-small@ed25f3a31f01632728cabb09d1542f84ab7b0056",
        "representation": "CLS plus 4x4 means of 16x16 DINOv2 patch grid; float32",
        "splits": "internal asset-hash train/validation derived only from official train partition",
        "official_public_validation_opened": False,
        "smoke_subset": args.smoke,
        "seed": 17,
        "batch_size": args.batch_size,
        "gpu_preflight": gpu_slot,
        "gpu": torch.cuda.get_device_name(),
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "versions": {
            package: importlib.metadata.version(package)
            for package in ("torch", "transformers", "numpy", "Pillow")
        },
        "exact_argv": sys.argv,
    }
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    feature_media = {}
    with ZipFile(args.archive) as archive:
        names = set(archive.namelist())
        for split, items in manifests.items():
            array_path = args.output / f"{split}-cells.npy"
            array = np.lib.format.open_memmap(
                array_path, mode="w+", dtype=np.float32, shape=(len(items), 17, 384)
            )
            for offset in range(0, len(items), args.batch_size):
                batch = items[offset : offset + args.batch_size]
                images = []
                for row in batch:
                    member = row["archive_member"]
                    if member not in names:
                        raise ValueError(f"missing archive member {member}")
                    raw = archive.read(member)
                    digest = hashlib.sha256(raw).hexdigest()
                    if digest != row["media_sha256"]:
                        raise ValueError(f"image bytes changed for {row['id']}")
                    feature_media[row["id"]] = digest
                    with Image.open(BytesIO(raw)) as image:
                        images.append(
                            image.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
                        )
                encoded = processor(
                    images=images, do_resize=False, do_center_crop=False, return_tensors="pt"
                ).to("cuda")
                with torch.inference_mode():
                    tokens = model(**encoded).last_hidden_state.float().cpu().numpy()
                array[offset : offset + len(batch)] = spatial_cells(tokens)
                if offset % 4096 == 0 or offset + len(batch) == len(items):
                    print(
                        json.dumps(
                            {"split": split, "done": offset + len(batch), "total": len(items)}
                        ),
                        flush=True,
                    )
            array.flush()
            del array
            (args.output / f"{split}-ids.json").write_text(
                json.dumps([r["id"] for r in items], separators=(",", ":")) + "\n", encoding="utf-8"
            )
    (args.output / "source-media-sha256.json").write_text(
        json.dumps(feature_media, sort_keys=True) + "\n", encoding="utf-8"
    )
    result = {
        **config,
        "finished_utc": datetime.now(UTC).isoformat(),
        "train_images": len(manifests["train"]),
        "validation_images": len(manifests["validation"]),
        "official_public_validation_opened": False,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        "elapsed_seconds": time.perf_counter() - started,
        "unique_media_assets": len(feature_media),
        "sha256": {
            name: sha(args.output / name)
            for name in (
                "train-cells.npy",
                "validation-cells.npy",
                "train-ids.json",
                "validation-ids.json",
                "source-media-sha256.json",
            )
        },
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
