"""Extract label-free BSD10k features with pinned frozen EfficientAT DyMN-0.4."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.extract_perception_source_candidate_audio_features import (  # noqa: E402
    MODEL_PARAMETERS,
    SOURCE_COMMIT,
    WEIGHTS_SHA256,
    _load_model,
    _state_sha,
    decode_resampled_mono_wav,
)

SAMPLE_RATE_HZ = 32_000
WINDOW_SAMPLES = 2 * SAMPLE_RATE_HZ
HOP_SAMPLES = SAMPLE_RATE_HZ
FEATURE_WIDTH = 384


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fixed_audio_windows(waveform: np.ndarray) -> np.ndarray:
    """Make complete two-second windows at a one-second hop, right-padding only the tail."""
    values = np.asarray(waveform, dtype=np.float32)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("waveform must be nonempty, finite, and mono")
    starts = list(range(0, max(len(values) - WINDOW_SAMPLES + 1, 1), HOP_SAMPLES))
    while starts[-1] + WINDOW_SAMPLES < len(values):
        starts.append(starts[-1] + HOP_SAMPLES)
    windows = np.zeros((len(starts), WINDOW_SAMPLES), dtype=np.float32)
    for index, start in enumerate(starts):
        clip = values[start : start + WINDOW_SAMPLES]
        windows[index, : len(clip)] = clip
    return windows


def extract_split(
    split_manifest_path: Path,
    download_result_path: Path,
    media_dir: Path,
    output_dir: Path,
    weights: Path,
    source: Path,
    batch_size: int = 24,
    limit: int | None = None,
) -> dict:
    if not os.environ.get("TINY_OMNI_GPU_GATE_RESERVED_MIB"):
        raise RuntimeError("feature extraction must be run under run_with_hotgpu_guard.py")
    if not 1 <= batch_size <= 64:
        raise ValueError("batch size must be between 1 and 64")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite frozen feature output: {output_dir}")

    manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "tiny-omni-decision/bsd10k-group-split-v1":
        raise ValueError("unexpected BSD10k split manifest")
    result = json.loads(download_result_path.read_text(encoding="utf-8"))
    split = result.get("split")
    if split not in {"train", "validation", "test"}:
        raise ValueError("download result must identify a supported split")
    split_ids = {str(row["sound_id"]) for row in manifest["examples"] if row["split"] == split}
    ordered_split_ids = sorted(split_ids, key=int)
    if limit is not None:
        if limit < 1:
            raise ValueError("feature smoke limit must be positive")
        expected_ids = set(ordered_split_ids[:limit])
    else:
        expected_ids = split_ids
    files = result.get("downloaded_files")
    actual_ids = [str(row.get("sound_id", "")) for row in files or []]
    if (
        result.get("source_record_id") != manifest.get("source_record_id")
        or result.get("selected_ids") != len(split_ids)
        or result.get("downloaded_count") != len(actual_ids)
        or len(actual_ids) != len(set(actual_ids))
        or not expected_ids.issubset(set(actual_ids))
        or (limit is None and set(actual_ids) != expected_ids)
    ):
        raise ValueError("download result does not exactly match the frozen split membership")

    file_rows = {str(row["sound_id"]): row for row in files}
    for sound_id, row in file_rows.items():
        path = media_dir / f"{sound_id}.wav"
        if path.stat().st_size != row["bytes"] or sha256_file(path) != row["sha256"]:
            raise ValueError(f"downloaded media hash mismatch for sound {sound_id}")

    torch, model, mel, parameter_count, source_commit = _load_model(weights, source)
    if parameter_count != MODEL_PARAMETERS or source_commit != SOURCE_COMMIT:
        raise ValueError("EfficientAT source or parameter count differs from pinned audit")
    before_state = _state_sha(model)
    features = np.empty((len(expected_ids), FEATURE_WIDTH), dtype=np.float32)
    ordered_ids = sorted(expected_ids, key=int)
    started = time.perf_counter()
    window_count = 0
    peak_allocated = 0
    peak_reserved = 0
    with torch.inference_mode():
        for record_index, sound_id in enumerate(ordered_ids, start=1):
            path = media_dir / f"{sound_id}.wav"
            waveform, start_us = decode_resampled_mono_wav(path, sample_rate_hz=SAMPLE_RATE_HZ)
            if start_us != 0:
                raise ValueError(f"BSD10k WAV unexpectedly has a nonzero start time: {sound_id}")
            windows = fixed_audio_windows(waveform)
            embeddings = []
            for start in range(0, len(windows), batch_size):
                batch = torch.from_numpy(windows[start : start + batch_size]).cuda()
                mel_input = mel(batch).unsqueeze(1)
                logits, output_features = model(mel_input)
                if (
                    logits.shape != (len(batch), 527)
                    or output_features.shape != (len(batch), FEATURE_WIDTH)
                    or not torch.isfinite(output_features).all()
                ):
                    raise ValueError(f"EfficientAT produced invalid features for sound {sound_id}")
                embeddings.extend(output_features.float().cpu().numpy())
                torch.cuda.synchronize()
                peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated())
                peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved())
            pooled = np.mean(np.stack(embeddings), axis=0, dtype=np.float64).astype(np.float32)
            norm = float(np.linalg.norm(pooled))
            if not np.isfinite(norm) or norm <= 1e-12:
                raise ValueError(f"EfficientAT pooled feature has invalid norm: {sound_id}")
            features[record_index - 1] = pooled / norm
            window_count += len(windows)
            if record_index % 25 == 0 or record_index == len(ordered_ids):
                print(
                    json.dumps(
                        {
                            "split": split,
                            "completed_records": record_index,
                            "total_records": len(ordered_ids),
                            "windows": window_count,
                            "elapsed_seconds": time.perf_counter() - started,
                        }
                    ),
                    flush=True,
                )
    after_state = _state_sha(model)
    if before_state != after_state:
        raise ValueError("frozen EfficientAT weights changed during feature extraction")

    output_dir.mkdir(parents=True)
    feature_path = output_dir / "features.npy"
    index_path = output_dir / "sample-index.json"
    np.save(feature_path, features, allow_pickle=False)
    index_path.write_text(
        json.dumps(
            {"split": split, "sound_ids": ordered_ids, "window_count": window_count},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    report = {
        "status": "PASS_SMOKE_FEATURES" if limit is not None else "PASS_FROZEN_FEATURES",
        "split": split,
        "partial_smoke": limit is not None,
        "split_manifest_sha256": sha256_file(split_manifest_path),
        "download_result_sha256": sha256_file(download_result_path),
        "weights_sha256": WEIGHTS_SHA256,
        "source_commit": source_commit,
        "model_parameters": parameter_count,
        "model_state_sha256_before": before_state,
        "model_state_sha256_after": after_state,
        "records": len(ordered_ids),
        "window_seconds": 2,
        "hop_seconds": 1,
        "window_count": window_count,
        "feature_shape": list(features.shape),
        "feature_sha256": sha256_file(feature_path),
        "sample_index_sha256": sha256_file(index_path),
        "dtype": str(next(model.parameters()).dtype),
        "device": torch.cuda.get_device_name(0),
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "elapsed_seconds": time.perf_counter() - started,
        "labels_loaded": False,
        "metrics_computed": False,
        "precision": "FP32",
    }
    (output_dir / "run.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--download-result", type=Path, required=True)
    parser.add_argument("--media-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    report = extract_split(
        args.split_manifest.resolve(strict=True),
        args.download_result.resolve(strict=True),
        args.media_dir.resolve(strict=True),
        args.output.resolve(),
        args.weights.resolve(strict=True),
        args.source.resolve(strict=True),
        args.batch_size,
        args.limit,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
