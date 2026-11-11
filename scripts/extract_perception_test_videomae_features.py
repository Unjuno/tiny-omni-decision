"""Extract pinned frozen VideoMAE V2 token grids for fixed train/valid media."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from scripts.extract_perception_test_videoprism_features import (
    _load_fixed_inputs,
    sha256_file,
)

MODEL_REVISION = "706cc172d65ebd4dedbee3f9c0183a93df9fa125"
SOURCE_COMMIT = "29eab1e8a588d1b3ec0cdec7b03a86cca491b74b"
SOURCE_SHA256 = "d132c97369f25207e380e78c59294fd00b93d2f6aa0fb5c4ded19eff26a5a3cb"
CHECKPOINT_SHA256 = "24fb71687fa3671b8387cadfbcbab0f72af695692e93cf1ecc82caa888626172"
PREFLIGHT_SHA256 = "e04504116449b100d9c415d63b5fa0204c550c83cf75ef2b4d6fc523c476c910"
MODEL_PARAMETERS = 22_153_286
FRAME_COUNT = 16
FRAME_SIZE = 224
TOKEN_SHAPE = (8, 14, 14, 384)


def uniform_indices(total: int, requested: int = FRAME_COUNT) -> list[int]:
    if total < requested or requested < 1:
        raise ValueError("video must contain at least the requested number of frames")
    if requested == 1:
        return [0]
    step = (total - 1) / (requested - 1)
    return [round(i * step) for i in range(requested)]


def _decode_frames(path: Path, requested: int = FRAME_COUNT) -> np.ndarray:
    import av

    with av.open(str(path)) as container:
        streams = [stream for stream in container.streams if stream.type == "video"]
        if not streams:
            raise ValueError(f"video stream missing: {path}")
        stream = streams[0]
        total = int(stream.frames or 0)
        if total <= 0:
            if stream.duration is None or stream.time_base is None or not stream.average_rate:
                raise ValueError("video frame count cannot be derived from stream metadata")
            total = round(float(stream.duration * stream.time_base * stream.average_rate))
        if total < requested:
            raise ValueError(f"video metadata reports only {total} frames; need {requested}")
        wanted = set(uniform_indices(total, requested))
        selected: dict[int, np.ndarray] = {}
        for index, frame in enumerate(container.decode(stream)):
            if index in wanted:
                selected[index] = frame.reformat(
                    width=FRAME_SIZE, height=FRAME_SIZE, format="rgb24"
                ).to_ndarray()
            if index >= max(wanted):
                break
    if set(selected) != wanted:
        raise ValueError(f"decoder returned {len(selected)}/{requested} frames")
    return np.stack([selected[index] for index in sorted(wanted)])


def preprocess_frames(frames: np.ndarray) -> np.ndarray:
    values = np.asarray(frames)
    if values.shape != (FRAME_COUNT, FRAME_SIZE, FRAME_SIZE, 3) or values.dtype != np.uint8:
        raise ValueError("decoded frames do not match the pinned RGB uint8 input shape")
    return values.astype(np.float32) / 127.5 - 1.0


def _receipt_set_hash(rows: list[dict[str, Any]]) -> str:
    canonical = "\n".join(
        json.dumps(row, sort_keys=True, separators=(",", ":"))
        for row in sorted(rows, key=lambda item: item["video_id"])
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _load_preflight_progress(preflight_path: Path, progress_path: Path, split: str):
    preflight_hash = sha256_file(preflight_path)
    if preflight_hash != PREFLIGHT_SHA256:
        raise ValueError("media preflight hash differs from the pinned candidate")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if (
        preflight.get("status") != "PASS_METADATA_ONLY_MEDIA_PREFLIGHT"
        or preflight.get("official_test_archive_opened") is not False
    ):
        raise ValueError("media preflight does not attest the fixed train/validation-only scope")
    expected = preflight.get("splits", {}).get(split, {}).get("members")
    if not isinstance(expected, list) or not expected:
        raise ValueError(f"pinned preflight has no {split} media members")
    expected_by_id = {row["video_id"]: row for row in expected}
    if len(expected_by_id) != len(expected):
        raise ValueError("duplicate video IDs in media preflight")

    raw = progress_path.read_bytes()
    complete_bytes = raw if raw.endswith(b"\n") else raw[: raw.rfind(b"\n") + 1]
    receipts = [json.loads(line) for line in complete_bytes.splitlines() if line.strip()]
    all_ids = [row.get("video_id") for row in receipts]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("duplicate IDs in acquisition progress")
    selected = [row for row in receipts if row.get("split") == split]
    actual_by_id = {row["video_id"]: row for row in selected}
    if set(actual_by_id) != set(expected_by_id):
        raise ValueError(
            f"{split} media receipts incomplete: {len(actual_by_id)}/{len(expected_by_id)}"
        )
    ordered = []
    for video_id, expected_row in expected_by_id.items():
        receipt = actual_by_id[video_id]
        for key in ("split", "member", "expanded_bytes", "compressed_bytes", "crc32"):
            if receipt.get(key) != expected_row.get(key):
                raise ValueError(f"{split} progress receipt differs from preflight: {video_id}")
        if (
            receipt.get("crc32_verified") is not True
            or receipt.get("expanded_size_verified") is not True
        ):
            raise ValueError(f"{split} progress receipt lacks ZIP/size verification: {video_id}")
        relative = Path(str(receipt.get("local_relative_path", "")))
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError(f"unsafe path in acquisition receipt: {video_id}")
        ordered.append(receipt)
    ordered.sort(key=lambda row: row["video_id"])
    return preflight_hash, _receipt_set_hash(ordered), hashlib.sha256(raw).hexdigest(), ordered


def _load_model(model_dir: Path, dependency_dir: Path, device_name: str):
    source = model_dir / "modeling_finetune.py"
    checkpoint = model_dir / "vit_s_k710_dl_from_giant.pth"
    if sha256_file(source) != SOURCE_SHA256:
        raise ValueError("VideoMAE source SHA-256 mismatch")
    if sha256_file(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("VideoMAE checkpoint SHA-256 mismatch")
    if device_name not in {"cuda", "cpu"}:
        raise ValueError("device must be cuda or cpu")
    dependency_path = str(dependency_dir.resolve(strict=True))
    if dependency_path not in sys.path:
        sys.path.insert(0, dependency_path)
    import torch

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(device_name)
    spec = importlib.util.spec_from_file_location("perception_test_videomae_source", source)
    if spec is None or spec.loader is None:
        raise ImportError("could not load the pinned VideoMAE source file")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    torch.set_num_threads(4)
    model = module.vit_small_patch16_224(num_classes=710).eval().requires_grad_(False)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state["module"], strict=True)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters != MODEL_PARAMETERS:
        raise ValueError(f"unexpected VideoMAE parameter count: {parameters}")
    model = model.to(device)
    capture: dict[str, Any] = {}

    def hook(_module, _inputs, output):
        capture["tokens"] = output.detach()

    handle = model.blocks[-1].register_forward_hook(hook)
    return torch, model, device, capture, handle, parameters, source, checkpoint


def _forward_feature(torch, model, device, capture, frames: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(preprocess_frames(frames).copy())
    tensor = tensor.permute(3, 0, 1, 2).unsqueeze(0).to(device)
    capture.pop("tokens", None)
    with torch.inference_mode():
        pooled = model.forward_features(tensor)
    tokens = capture.get("tokens")
    if tokens is None or tuple(tokens.shape) != (1, int(np.prod(TOKEN_SHAPE[:3])), TOKEN_SHAPE[-1]):
        raise ValueError("VideoMAE final-block token shape differs from pinned contract")
    if tuple(pooled.shape) != (1, TOKEN_SHAPE[-1]):
        raise ValueError("VideoMAE pooled representation has unexpected shape")
    torch.testing.assert_close(
        pooled,
        model.fc_norm(tokens.mean(1)),
        rtol=0,
        atol=0,
    )
    grid = tokens.reshape((1, *TOKEN_SHAPE)).squeeze(0).float().cpu().numpy()
    pooled_np = pooled.squeeze(0).float().cpu().numpy()
    if not np.isfinite(grid).all() or not np.isfinite(pooled_np).all():
        raise FloatingPointError("VideoMAE produced non-finite representations")
    return grid


def extract_split(
    *,
    preflight_path: Path,
    media_manifest_path: Path | None = None,
    acquisition_progress_path: Path | None = None,
    media_root: Path,
    model_dir: Path,
    dependency_dir: Path,
    output_dir: Path,
    split: str,
    device_name: str = "cuda",
) -> dict[str, Any]:
    if split not in {"train", "valid"}:
        raise ValueError("only train and public validation splits are supported")
    if (media_manifest_path is None) == (acquisition_progress_path is None):
        raise ValueError("provide exactly one complete media manifest or acquisition progress log")
    required_paths = [preflight_path, media_root, model_dir, dependency_dir]
    if media_manifest_path is not None:
        required_paths.append(media_manifest_path)
    if acquisition_progress_path is not None:
        required_paths.append(acquisition_progress_path)
    for path in required_paths:
        path.resolve(strict=True)
    if output_dir.drive.upper() != "D:":
        raise ValueError("feature outputs must stay on D:")
    if media_manifest_path is not None:
        preflight_hash, _manifest_hash, media_rows = _load_fixed_inputs(
            preflight_path, media_manifest_path, split
        )
        media_source_kind = "complete_media_manifest"
        media_source_file_sha256 = sha256_file(media_manifest_path)
    else:
        (
            preflight_hash,
            source_receipt_hash,
            media_source_file_sha256,
            media_rows,
        ) = _load_preflight_progress(preflight_path, acquisition_progress_path, split)
        media_source_kind = "verified_acquisition_progress"
    receipt_set_sha256 = _receipt_set_hash(media_rows)
    if media_manifest_path is None and source_receipt_hash != receipt_set_sha256:
        raise ValueError("acquisition progress receipt-set hash mismatch")
    output_dir.mkdir(parents=True, exist_ok=True)
    identity = {
        "schema_version": 1,
        "status": "RUNNING",
        "split": split,
        "preflight_sha256": preflight_hash,
        "media_receipt_set_sha256": receipt_set_sha256,
        "project_code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True
        ).strip(),
        "extractor_script_sha256": sha256_file(Path(__file__).resolve()),
        "source_commit": SOURCE_COMMIT,
        "source_sha256": SOURCE_SHA256,
        "model_revision": MODEL_REVISION,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "frame_count": FRAME_COUNT,
        "frame_size": FRAME_SIZE,
        "preprocessing": (
            "uniform 16 endpoint-inclusive RGB frames, resize 224, /255 then mean/std 0.5"
        ),
        "token_shape_per_video": list(TOKEN_SHAPE),
        "labels_loaded": False,
        "device_requested": device_name,
    }
    metadata_path = output_dir / "run.json"
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if any(existing.get(key) != value for key, value in identity.items() if key != "status"):
            raise ValueError("existing feature metadata differs; refusing resume")
    else:
        metadata_path.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")

    receipt_path = output_dir / "feature-progress.jsonl"
    receipts: dict[str, dict[str, Any]] = {}
    if receipt_path.exists():
        for line in receipt_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["video_id"] in receipts:
                raise ValueError("duplicate feature receipt")
            receipts[row["video_id"]] = row
    media_by_id = {row["video_id"]: row for row in media_rows}
    if not set(receipts).issubset(media_by_id):
        raise ValueError("feature receipts include unexpected video IDs")
    for video_id, row in receipts.items():
        if row.get("media_sha256") != media_by_id[video_id]["sha256"]:
            raise ValueError(f"feature receipt media hash mismatch: {video_id}")
        feature_path = output_dir / f"{video_id}.npy"
        if not feature_path.is_file() or sha256_file(feature_path) != row.get("feature_sha256"):
            raise ValueError(f"feature shard missing or hash mismatch: {video_id}")

    torch, model, device, capture, handle, parameters, source, checkpoint = _load_model(
        model_dir, dependency_dir, device_name
    )
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    pending = [row for row in media_rows if row["video_id"] not in receipts]
    try:
        with receipt_path.open("a", encoding="utf-8", newline="\n") as progress:
            for index, item in enumerate(pending, start=1):
                video_id = item["video_id"]
                relative = Path(item["local_relative_path"])
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("unsafe path in media receipt")
                video_path = media_root / relative
                if (
                    not video_path.is_file()
                    or video_path.stat().st_size != item["expanded_bytes"]
                    or sha256_file(video_path) != item["sha256"]
                ):
                    raise ValueError(f"media differs from fixed receipt: {video_id}")
                tick = time.perf_counter()
                frames = _decode_frames(video_path)
                feature = _forward_feature(torch, model, device, capture, frames)
                destination = output_dir / f"{video_id}.npy"
                temporary = output_dir / f"{video_id}.npy.tmp"
                if destination.exists() or temporary.exists():
                    raise FileExistsError(f"refusing to overwrite feature shard: {video_id}")
                with temporary.open("xb") as stream:
                    np.save(stream, feature, allow_pickle=False)
                    stream.flush()
                    os.fsync(stream.fileno())
                temporary.replace(destination)
                receipt = {
                    "video_id": video_id,
                    "split": split,
                    "media_sha256": item["sha256"],
                    "feature_sha256": sha256_file(destination),
                    "feature_bytes": destination.stat().st_size,
                    "feature_shape": list(feature.shape),
                    "seconds": time.perf_counter() - tick,
                }
                progress.write(json.dumps(receipt, sort_keys=True) + "\n")
                progress.flush()
                os.fsync(progress.fileno())
                receipts[video_id] = receipt
                if index % 25 == 0 or index == len(pending):
                    print(f"{len(receipts)}/{len(media_rows)} {split} videos", flush=True)
    finally:
        handle.remove()

    if set(receipts) != set(media_by_id):
        raise RuntimeError("feature extraction stopped before the split was complete")
    report = {
        **identity,
        "status": "COMPLETE",
        "media_source_kind": media_source_kind,
        "media_source_file_sha256": media_source_file_sha256,
        "model_parameters": parameters,
        "model_source_sha256": sha256_file(source),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu_used": device.type == "cuda",
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "peak_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
        "peak_reserved_bytes": (
            torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0
        ),
        "video_count": len(receipts),
        "elapsed_seconds_this_invocation": time.perf_counter() - started,
        "feature_receipts": [receipts[key] for key in sorted(receipts)],
    }
    report_path = output_dir / "feature-manifest.json"
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite feature manifest: {report_path}")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report["feature_manifest_sha256"] = sha256_file(report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path, required=True)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--media-manifest", type=Path)
    source_group.add_argument("--acquisition-progress", type=Path)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dependency-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "valid"), required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    report = extract_split(
        preflight_path=args.preflight,
        media_manifest_path=args.media_manifest,
        acquisition_progress_path=args.acquisition_progress,
        media_root=args.media_root,
        model_dir=args.model_dir,
        dependency_dir=args.dependency_dir,
        output_dir=args.output_dir,
        split=args.split,
        device_name=args.device,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "split": report["split"],
                "video_count": report["video_count"],
                "device": report["device"],
                "peak_allocated_bytes": report["peak_allocated_bytes"],
                "feature_manifest_sha256": report["feature_manifest_sha256"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
