"""Extract resumable whole-clip frozen EfficientAT features for FSD50K."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import subprocess
import sys
import time
import wave
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.build_fsd50k_annotation_decision_manifest import (
    canonical_sha256,
)
from scripts.validate_fsd50k_annotation_acquisition import (
    load_frozen_manifest,
    validate_acquisition,
)

ENCODER_REVISION = "a425fdce92572e602a1d5634799bd9f1f2efa806"
ENCODER_WEIGHTS_SHA256 = "49b0e63577021994257b3de2ac5f1686d71554b93c8149f9d2568ad5760b904e"
ENCODER_PARAMETERS = 1_961_239
FEATURE_WIDTH = 384
SAMPLE_RATE = 32_000
SHARD_SIZE = 32
TEMPORAL_POLICY = "whole_clip_native_audio_resampled_to_32khz"
DEFAULT_ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def decode_mono_pcm_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as source:
        if source.getcomptype() != "NONE" or source.getnchannels() != 1:
            raise ValueError(f"FSD50K input must be uncompressed mono PCM: {path}")
        channels = source.getnchannels()
        width = source.getsampwidth()
        sample_rate = source.getframerate()
        frame_count = source.getnframes()
        raw = source.readframes(frame_count)
    if sample_rate <= 0 or width not in {1, 2, 3, 4} or not raw:
        raise ValueError(f"unsupported or empty FSD50K WAV: {path}")
    if width == 1:
        samples = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 3:
        packed = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        values = packed[:, 0] | (packed[:, 1] << 8) | (packed[:, 2] << 16)
        samples = ((values ^ 0x800000) - 0x800000).astype(np.float32) / 8388608.0
    else:
        samples = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    if channels != 1 or len(samples) != frame_count or not np.isfinite(samples).all():
        raise ValueError(f"invalid decoded mono PCM samples: {path}")
    return samples, sample_rate


def encode_full_clip(
    waveform: np.ndarray,
    source_rate: int,
    *,
    encoder: Any,
    torchaudio_module: Any,
) -> tuple[np.ndarray, int]:
    if waveform.ndim != 1 or not waveform.size or source_rate <= 0:
        raise ValueError("whole-clip input must be a nonempty mono waveform")
    if not np.isfinite(waveform).all():
        raise ValueError("whole-clip input contains nonfinite audio")
    values = torch.from_numpy(np.asarray(waveform, dtype=np.float32).copy()).unsqueeze(0)
    if source_rate != SAMPLE_RATE:
        values = torchaudio_module.functional.resample(values, source_rate, SAMPLE_RATE)
    values = values.squeeze(0).to(encoder.device)
    if not torch.isfinite(values).all():
        raise ValueError("resampled whole-clip input contains nonfinite audio")
    with torch.inference_mode():
        _, features = encoder.model(encoder.mel(values.unsqueeze(0)).unsqueeze(1))
    features = features.float()
    if features.shape != (1, FEATURE_WIDTH) or not torch.isfinite(features).all():
        raise ValueError(f"invalid full-clip EfficientAT feature shape/value: {features.shape}")
    return features[0].detach().cpu().numpy().astype(np.float32, copy=False), len(values)


def _expected_rows(
    preflight_dir: Path,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    report, splits = load_frozen_manifest(preflight_dir)
    ordered = [row for split in ("train", "calibration", "validation") for row in splits[split]]
    ids = [row["sample_id"] for row in ordered]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate sample IDs in frozen FSD50K order")
    return report, splits, ordered


def _load_receipt(media_dir: Path) -> tuple[dict[str, dict[str, Any]], str]:
    receipt_path = media_dir / "files.jsonl"
    records = [
        json.loads(line)
        for line in receipt_path.read_text(encoding="utf-8").splitlines()
    ]
    by_id = {row.get("sample_id"): row for row in records}
    if len(records) != len(by_id):
        raise ValueError("duplicate sample IDs in media acquisition receipt")
    return by_id, file_sha256(receipt_path)


def _feature_row(
    index: int,
    manifest_row: dict[str, Any],
    media_receipt: dict[str, Any],
    duration_seconds: float,
    source_rate: int,
    feature: np.ndarray,
) -> dict[str, Any]:
    feature_bytes = np.asarray(feature, dtype="<f4").tobytes()
    return {
        "feature_index": index,
        "sample_id": manifest_row["sample_id"],
        "source_id": manifest_row["source_id"],
        "group_id": manifest_row["group_id"],
        "split": manifest_row["split"],
        "license": manifest_row["license"],
        "targets": manifest_row["targets"],
        "media_sha256": media_receipt["sha256"],
        "feature_sha256": hashlib.sha256(feature_bytes).hexdigest(),
        "duration_seconds": duration_seconds,
        "source_sample_rate_hz": source_rate,
    }


def _build_identity(
    preflight_report: dict[str, Any],
    verification: dict[str, Any],
    verification_path: Path,
    receipt_sha: str,
    ordered_rows: list[dict[str, Any]],
    source_root: Path,
    *,
    device: str,
) -> dict[str, Any]:
    source_head = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
    ).strip()
    source_status = subprocess.check_output(
        ["git", "-C", str(source_root), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip()
    if source_head != ENCODER_REVISION or source_status:
        raise ValueError("EfficientAT source is not the clean pinned checkout")
    return {
        "preflight_manifest_sha256": preflight_report["manifest_sha256"],
        "acquisition_verification_sha256": file_sha256(verification_path),
        "acquisition_receipt_sha256": receipt_sha,
        "sample_order_sha256": canonical_sha256([row["sample_id"] for row in ordered_rows]),
        "encoder_revision": ENCODER_REVISION,
        "encoder_source_clean": True,
        "extractor_sha256": file_sha256(Path(__file__).resolve()),
        "encoder_weights_sha256": ENCODER_WEIGHTS_SHA256,
        "encoder_parameters": ENCODER_PARAMETERS,
        "feature_width": FEATURE_WIDTH,
        "feature_dtype": "float32",
        "source_sample_rate_policy": "decode_native_pcm_then_resample_to_32000",
        "temporal_policy": TEMPORAL_POLICY,
        "device": device,
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "torchaudio_version": "2.6.0+cu124",
        "cuda_runtime_version": torch.version.cuda,
        "official_eval_used": False,
        "verified_files": verification["verified_files"],
    }


def _load_resume(output: Path, identity: dict[str, Any], ordered_rows: list[dict[str, Any]]):
    config_path = output / "effective-config.json"
    progress_path = output / "progress.json"
    shards_dir = output / "shards"
    if not config_path.is_file() or not progress_path.is_file() or not shards_dir.is_dir():
        raise FileExistsError(f"output exists without valid resumable feature state: {output}")
    saved_config = json.loads(config_path.read_text(encoding="utf-8"))
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    identity_hash = canonical_sha256(identity)
    if saved_config.get("identity") != identity or progress.get("identity_sha256") != identity_hash:
        raise ValueError("existing FSD feature output belongs to different frozen inputs")
    if progress.get("status") == "PASS":
        final_path = output / "feature-run.json"
        if (
            not final_path.is_file()
            or file_sha256(final_path) != progress.get("feature_run_sha256")
        ):
            raise ValueError("completed FSD feature receipt failed hash validation")
        final = json.loads(final_path.read_text(encoding="utf-8"))
        if (
            final.get("status") != "PASS"
            or final.get("identity_sha256") != identity_hash
            or final.get("features_sha256") != file_sha256(output / "features.npy")
            or final.get("feature_rows_sha256") != file_sha256(output / "feature-rows.jsonl")
        ):
            raise ValueError("completed FSD feature arrays failed receipt hash validation")
        return progress, int(progress["completed_assets"]), []
    if any(
        (output / name).exists()
        for name in (
            "features.npy",
            "features.npy.tmp",
            "feature-rows.jsonl",
            "feature-rows.jsonl.tmp",
            "feature-run.json",
        )
    ):
        raise ValueError("final FSD feature artifacts exist before terminal PASS; refusing resume")
    completed = int(progress.get("completed_assets", -1))
    shards = progress.get("shards", [])
    expected_shard_count = (completed + SHARD_SIZE - 1) // SHARD_SIZE
    if (
        progress.get("status") != "RUNNING"
        or completed < 0
        or completed > len(ordered_rows)
        or (completed != len(ordered_rows) and completed % SHARD_SIZE != 0)
        or len(shards) != expected_shard_count
    ):
        raise ValueError("FSD feature progress metadata is inconsistent")
    recovered_rows: list[dict[str, Any]] = []
    expected_names = set()
    for index, shard_record in enumerate(shards):
        first = index * SHARD_SIZE
        count = min(SHARD_SIZE, completed - first)
        path = shards_dir / shard_record["name"]
        expected_names.add(path.name)
        if (
            shard_record.get("first_feature_index") != first
            or shard_record.get("feature_count") != count
            or not path.is_file()
            or file_sha256(path) != shard_record.get("sha256")
        ):
            raise ValueError(f"FSD feature shard failed resume verification: {path}")
        with np.load(path, allow_pickle=False) as saved:
            ids = saved["sample_ids"].tolist()
            values = saved["features"]
            rows = [json.loads(value) for value in saved["rows"].tolist()]
        expected_ids = [row["sample_id"] for row in ordered_rows[first : first + count]]
        if (
            ids != expected_ids
            or values.shape != (count, FEATURE_WIDTH)
            or not np.isfinite(values).all()
        ):
            raise ValueError(f"FSD feature shard IDs or arrays mismatch: {path}")
        if [row.get("feature_index") for row in rows] != list(range(first, first + count)):
            raise ValueError(f"FSD feature shard row indexes mismatch: {path}")
        if [row.get("sample_id") for row in rows] != expected_ids:
            raise ValueError(f"FSD feature shard row IDs mismatch: {path}")
        recovered_rows.extend(rows)
    if any(path.name not in expected_names for path in shards_dir.glob("*.npz")):
        raise ValueError("untracked FSD feature shard exists")
    if any(shards_dir.glob("*.tmp")):
        raise ValueError("temporary FSD feature shard exists; refusing unsafe resume")
    return progress, completed, recovered_rows


def _load_encoder(source_root: Path, checkpoint: Path, dependency_root: Path, device: str):
    if file_sha256(checkpoint) != ENCODER_WEIGHTS_SHA256:
        raise ValueError("pinned EfficientAT checkpoint hash mismatch")
    sys.path.insert(0, str(dependency_root.resolve()))
    sys.path.insert(0, str(source_root.resolve()))
    import torchaudio

    if torch.__version__ != "2.6.0+cu124" or torchaudio.__version__ != "2.6.0+cu124":
        raise ValueError("EfficientAT runtime differs from pinned PyTorch/torchaudio 2.6.0+cu124")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    original_cuda_available = torch.cuda.is_available
    if device == "cpu":
        torch.cuda.is_available = lambda: False
    try:
        from scripts.extract_avatar_efficientat_features import EfficientATEncoder

        with contextlib.redirect_stdout(io.StringIO()):
            encoder = EfficientATEncoder(source_root, checkpoint, dependency_root)
    finally:
        torch.cuda.is_available = original_cuda_available
    if encoder.device.type != device or encoder.parameter_count != ENCODER_PARAMETERS:
        raise ValueError("EfficientAT loaded on an unexpected device/parameter count")
    return encoder, torchaudio


def extract_features(
    preflight_dir: Path,
    media_dir: Path,
    verification_path: Path,
    output_dir: Path,
    source_root: Path,
    checkpoint: Path,
    dependency_root: Path,
    *,
    device: str = "cuda",
) -> dict[str, Any]:
    preflight, _splits, ordered_rows = _expected_rows(preflight_dir)
    verification = validate_acquisition(preflight_dir, media_dir)
    if verification_path.exists():
        saved_verification = json.loads(verification_path.read_text(encoding="utf-8"))
        if saved_verification != verification:
            raise ValueError("existing acquisition verification differs from full media recheck")
    else:
        verification_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(verification_path, verification)
    media_receipts, receipt_sha = _load_receipt(media_dir)
    if len(media_receipts) != len(ordered_rows):
        raise ValueError("media receipt count differs from frozen FSD50K rows")
    identity = _build_identity(
        preflight,
        verification,
        verification_path,
        receipt_sha,
        ordered_rows,
        source_root,
        device=device,
    )
    if output_dir.exists():
        progress, completed, recovered_rows = _load_resume(output_dir, identity, ordered_rows)
        if progress.get("status") == "PASS":
            return json.loads((output_dir / "feature-run.json").read_text(encoding="utf-8"))
    else:
        output_dir.mkdir(parents=True)
        (output_dir / "shards").mkdir()
        identity_hash = canonical_sha256(identity)
        atomic_json(
            output_dir / "effective-config.json",
            {
                "identity": identity,
                "identity_sha256": identity_hash,
                "created_utc": datetime.now(UTC).isoformat(),
            },
        )
        progress = {
            "status": "RUNNING",
            "identity_sha256": identity_hash,
            "completed_assets": 0,
            "shards": [],
        }
        atomic_json(output_dir / "progress.json", progress)
        completed = 0
        recovered_rows = []

    encoder, torchaudio = _load_encoder(source_root, checkpoint, dependency_root, device)
    all_features = []
    if completed:
        for record in progress["shards"]:
            with np.load(output_dir / "shards" / record["name"], allow_pickle=False) as saved:
                all_features.append(saved["features"])
    feature_rows = recovered_rows
    feature_seconds = 0.0
    decode_seconds = 0.0
    peak_vram = int(progress.get("peak_allocated_vram_bytes", 0))
    starting_completed = completed
    started = time.perf_counter()
    batch_features = []
    batch_rows = []
    batch_ids = []
    accounted_feature_seconds = 0.0
    accounted_decode_seconds = 0.0

    def flush_shard() -> None:
        nonlocal completed, progress, batch_features, batch_rows, batch_ids
        nonlocal accounted_feature_seconds, accounted_decode_seconds
        if not batch_features:
            return
        values = np.stack(batch_features).astype(np.float32, copy=False)
        filename = f"shard-{completed // SHARD_SIZE:05d}.npz"
        shard_path = output_dir / "shards" / filename
        temporary = shard_path.with_suffix(".npz.tmp")
        with temporary.open("xb") as stream:
            np.savez_compressed(
                stream,
                sample_ids=np.asarray(batch_ids, dtype=np.str_),
                features=values,
                rows=np.asarray(
                    [json.dumps(row, sort_keys=True, separators=(",", ":")) for row in batch_rows],
                    dtype=np.str_,
                ),
            )
        temporary.replace(shard_path)
        count = len(batch_ids)
        progress["shards"].append(
            {
                "name": filename,
                "sha256": file_sha256(shard_path),
                "first_feature_index": completed,
                "feature_count": count,
            }
        )
        completed += count
        progress["completed_assets"] = completed
        progress["peak_allocated_vram_bytes"] = peak_vram
        progress["feature_seconds_active"] = float(
            progress.get("feature_seconds_active", 0.0)
        ) + feature_seconds - accounted_feature_seconds
        progress["decode_seconds_active"] = float(
            progress.get("decode_seconds_active", 0.0)
        ) + decode_seconds - accounted_decode_seconds
        accounted_feature_seconds = feature_seconds
        accounted_decode_seconds = decode_seconds
        progress["last_updated_utc"] = datetime.now(UTC).isoformat()
        atomic_json(output_dir / "progress.json", progress)
        all_features.append(values)
        feature_rows.extend(batch_rows)
        batch_features.clear()
        batch_rows.clear()
        batch_ids.clear()

    for index, row in enumerate(ordered_rows[completed:], start=completed):
        media_path = media_dir / row["split"] / f"{row['sample_id']}.wav"
        receipt = media_receipts.get(row["sample_id"])
        if (
            receipt is None
            or receipt.get("split") != row["split"]
            or receipt.get("sha256") is None
        ):
            raise ValueError(f"missing or mismatched media receipt: {row['sample_id']}")
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(encoder.device)
        decode_start = time.perf_counter()
        waveform, source_rate = decode_mono_pcm_wav(media_path)
        duration_seconds = len(waveform) / source_rate
        decode_seconds += time.perf_counter() - decode_start
        encode_start = time.perf_counter()
        feature, _resampled_samples = encode_full_clip(
            waveform, source_rate, encoder=encoder, torchaudio_module=torchaudio
        )
        feature_seconds += time.perf_counter() - encode_start
        if device == "cuda":
            peak_vram = max(peak_vram, int(torch.cuda.max_memory_allocated(encoder.device)))
        batch_features.append(feature)
        batch_ids.append(row["sample_id"])
        batch_rows.append(
            _feature_row(index, row, receipt, duration_seconds, source_rate, feature)
        )
        if len(batch_ids) == SHARD_SIZE or index + 1 == len(ordered_rows):
            flush_shard()
        if completed and completed % 256 == 0:
            print(f"completed {completed}/{len(ordered_rows)} frozen FSD features", flush=True)

    if completed != len(ordered_rows):
        raise ValueError(f"FSD feature extraction ended at {completed}/{len(ordered_rows)}")
    features = np.concatenate(all_features, axis=0)
    if features.shape != (len(ordered_rows), FEATURE_WIDTH) or not np.isfinite(features).all():
        raise ValueError("final FSD feature matrix is incomplete or nonfinite")
    if [row["sample_id"] for row in feature_rows] != [row["sample_id"] for row in ordered_rows]:
        raise ValueError("final FSD feature row order differs from frozen manifest")
    feature_path = output_dir / "features.npy"
    rows_path = output_dir / "feature-rows.jsonl"
    feature_temp = feature_path.with_suffix(".npy.tmp")
    with feature_temp.open("xb") as stream:
        np.save(stream, features, allow_pickle=False)
    feature_temp.replace(feature_path)
    rows_temp = rows_path.with_suffix(".jsonl.tmp")
    rows_temp.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in feature_rows),
        encoding="utf-8",
    )
    rows_temp.replace(rows_path)
    feature_run = {
        "status": "PASS",
        "identity_sha256": canonical_sha256(identity),
        "preflight_manifest_sha256": preflight["manifest_sha256"],
        "acquisition_verification_sha256": file_sha256(verification_path),
        "acquisition_receipt_sha256": receipt_sha,
        "sample_order_sha256": canonical_sha256([row["sample_id"] for row in ordered_rows]),
        "features_sha256": file_sha256(feature_path),
        "feature_rows_sha256": file_sha256(rows_path),
        "encoder_revision": ENCODER_REVISION,
        "encoder_weights_sha256": ENCODER_WEIGHTS_SHA256,
        "encoder_parameters": ENCODER_PARAMETERS,
        "feature_width": FEATURE_WIDTH,
        "feature_dtype": "float32",
        "temporal_policy": TEMPORAL_POLICY,
        "source_sample_rate_policy": identity["source_sample_rate_policy"],
        "samples": len(ordered_rows),
        "unique_assets": len({row["source_id"] for row in ordered_rows}),
        "unique_uploaders": len({row["group_id"] for row in ordered_rows}),
        "duration_over_30s": sum(row["duration_seconds"] > 30 for row in feature_rows),
        "device": str(encoder.device),
        "peak_allocated_vram_bytes": peak_vram,
        "feature_seconds_this_process": feature_seconds,
        "decode_seconds_this_process": decode_seconds,
        "wall_seconds_this_process": time.perf_counter() - started,
        "feature_seconds_active_total": progress.get("feature_seconds_active", 0.0),
        "decode_seconds_active_total": progress.get("decode_seconds_active", 0.0),
        "resume_completed_before_process": starting_completed,
        "official_eval_used": False,
    }
    feature_run_path = output_dir / "feature-run.json"
    atomic_json(feature_run_path, feature_run)
    progress.update(
        status="PASS",
        completed_assets=len(ordered_rows),
        feature_run_sha256=file_sha256(feature_run_path),
    )
    atomic_json(output_dir / "progress.json", progress)
    return feature_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preflight",
        type=Path,
        default=DEFAULT_ROOT / "fsd50k-annotation-presence-v1-preflight-20261115-attempt-01",
    )
    parser.add_argument("--media", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("C:/CodexArtifacts/pretrained-reuse-r1/efficientat-audit-v1/source"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("C:/CodexArtifacts/pretrained-reuse-r1/efficientat-audit-v1/dymn04_as.pt"),
    )
    parser.add_argument(
        "--dependency-root",
        type=Path,
        default=Path("C:/CodexArtifacts/pretrained-reuse-r1/python-deps"),
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    result = extract_features(
        args.preflight,
        args.media,
        args.verification,
        args.output,
        args.source_root,
        args.checkpoint,
        args.dependency_root,
        device=args.device,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
