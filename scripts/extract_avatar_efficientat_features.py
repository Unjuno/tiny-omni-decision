"""Extract pinned EfficientAT features for the audio side of AVATAR choices."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from itertools import groupby
from pathlib import Path
from typing import Any

import numpy as np

AUDIO_MODEL_REVISION = "a425fdce92572e602a1d5634799bd9f1f2efa806"
CHECKPOINT_SHA256 = "49b0e63577021994257b3de2ac5f1686d71554b93c8149f9d2568ad5760b904e"
SOURCE_ID = re.compile(r"[A-Za-z0-9_-]{11}_\d{5}/\d{5}\Z")
SAMPLE_RATE = 32_000
WINDOW_SECONDS = 2.0
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SECONDS)
BATCH_SIZE = 4
PINNED_ARCHIVE_SHA256 = "5ce5be7fa6dda85889750f6c7cc2fad807527063e7d7090c664170a66b078a68"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def write_json_atomic(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_audio_assets(train_path: Path, validation_path: Path) -> list[dict[str, Any]]:
    by_key: dict[str, tuple[float, float, float]] = {}
    for split, path in (("train", train_path), ("validation", validation_path)):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") != split:
                raise ValueError(f"{path}:{line_number}: unexpected split marker")
            key = row.get("audio_asset_key")
            center = row.get("audio_window_center_seconds")
            duration = row.get("audio_window_duration_seconds")
            if not isinstance(key, str) or SOURCE_ID.fullmatch(key) is None:
                raise ValueError(f"{path}:{line_number}: invalid audio asset key")
            if not isinstance(center, (int, float)) or center < 0:
                raise ValueError(f"{path}:{line_number}: invalid audio window center")
            if duration != WINDOW_SECONDS:
                raise ValueError(f"{path}:{line_number}: unexpected audio window duration")
            declared_padding = (
                float(row["audio_window_left_padding_seconds"]),
                float(row["audio_window_right_padding_seconds"]),
            )
            value = (float(center), *declared_padding)
            previous = by_key.setdefault(key, value)
            if any(abs(left - right) > 1e-9 for left, right in zip(previous, value, strict=True)):
                raise ValueError(f"conflicting audio window for asset {key}")
    return [
        {
            "asset_key": key,
            "center_seconds": by_key[key][0],
            "declared_left_padding_seconds": by_key[key][1],
            "declared_right_padding_seconds": by_key[key][2],
        }
        for key in sorted(by_key)
    ]


def crop_audio_window(
    waveform: np.ndarray,
    center_seconds: float,
    *,
    sample_rate: int = SAMPLE_RATE,
    duration_seconds: float = WINDOW_SECONDS,
    right_boundary_seconds: float | None = None,
) -> tuple[np.ndarray, int, int]:
    if waveform.ndim != 1 or sample_rate <= 0 or duration_seconds <= 0:
        raise ValueError("waveform must be mono and window parameters positive")
    window_samples = round(sample_rate * duration_seconds)
    start_sample = round((center_seconds - duration_seconds / 2) * sample_rate)
    result = np.zeros(window_samples, dtype=np.float32)
    source_start = max(0, start_sample)
    source_limit = len(waveform)
    if right_boundary_seconds is not None:
        source_limit = min(source_limit, round(right_boundary_seconds * sample_rate))
    source_end = min(source_limit, start_sample + window_samples)
    if source_end > source_start:
        target_start = source_start - start_sample
        length = source_end - source_start
        result[target_start : target_start + length] = waveform[source_start:source_end]
    left_padding = max(0, -start_sample)
    right_padding = max(0, start_sample + window_samples - source_limit)
    return result, left_padding, right_padding


def trim_to_audio_stream_end(
    waveform: np.ndarray, *, end_seconds: float, sample_rate: int = SAMPLE_RATE
) -> np.ndarray:
    if waveform.ndim != 1 or end_seconds <= 0 or sample_rate <= 0:
        raise ValueError("audio stream end and mono waveform must be valid")
    end_samples = round(end_seconds * sample_rate)
    if len(waveform) < end_samples:
        raise ValueError(
            "decoded resampled audio is shorter than declared audio stream end: "
            f"{len(waveform)} < {end_samples} samples"
        )
    return waveform[:end_samples].astype(np.float32, copy=False)


def decode_mono_32khz(path: Path) -> np.ndarray:
    try:
        import av
    except ImportError as exc:
        message = "PyAV is required for media decoding; install the project ML extra"
        raise RuntimeError(message) from exc
    with av.open(str(path)) as container:
        streams = [stream for stream in container.streams if stream.type == "audio"]
        if not streams:
            raise ValueError(f"no audio stream: {path}")
        stream = streams[0]
        if stream.duration is None or stream.time_base is None:
            raise ValueError(f"audio stream duration unavailable: {path}")
        stream_end_seconds = float(stream.duration * stream.time_base)
        if stream.start_time is not None:
            stream_end_seconds += float(stream.start_time * stream.time_base)
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=SAMPLE_RATE)
        chunks: list[tuple[int, np.ndarray]] = []
        cursor = 0
        for frame in container.decode(stream):
            for converted in resampler.resample(frame):
                samples = np.asarray(converted.to_ndarray(), dtype=np.float32).reshape(-1)
                if converted.pts is not None and converted.time_base is not None:
                    start = round(float(converted.pts * converted.time_base) * SAMPLE_RATE)
                else:
                    start = cursor
                chunks.append((start, samples))
                cursor = max(cursor, start + len(samples))
        for converted in resampler.resample(None):
            samples = np.asarray(converted.to_ndarray(), dtype=np.float32).reshape(-1)
            if converted.pts is not None and converted.time_base is not None:
                start = round(float(converted.pts * converted.time_base) * SAMPLE_RATE)
            else:
                start = cursor
            chunks.append((start, samples))
            cursor = max(cursor, start + len(samples))
    if not chunks or cursor <= 0:
        raise ValueError(f"audio stream decoded no samples: {path}")
    waveform = np.zeros(cursor, dtype=np.float32)
    for start, samples in chunks:
        first = max(start, 0)
        last = min(start + len(samples), len(waveform))
        if last > first:
            waveform[first:last] = samples[first - start : last - start]
    if not np.isfinite(waveform).all():
        raise ValueError(f"non-finite resampled audio: {path}")
    return trim_to_audio_stream_end(waveform, end_seconds=stream_end_seconds)


class EfficientATEncoder:
    def __init__(self, source_root: Path, checkpoint: Path, dependency_root: Path):
        if sha256_file(checkpoint) != CHECKPOINT_SHA256:
            raise ValueError("EfficientAT checkpoint hash mismatch")
        commit = subprocess.check_output(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
        ).strip()
        if commit != AUDIO_MODEL_REVISION:
            raise ValueError(f"unexpected EfficientAT source revision: {commit}")
        sys.path.insert(0, str(dependency_root.resolve()))
        sys.path.insert(0, str(source_root.resolve()))
        import torch
        import torchaudio
        from models.dymn.model import get_model
        from models.preprocess import AugmentMelSTFT

        if torch.__version__ != "2.6.0+cu124" or torchaudio.__version__ != "2.6.0+cu124":
            raise ValueError("runtime differs from the measured PyTorch/torchaudio pair")
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = get_model(width_mult=0.4, pretrained_name=None, T_max=1.0)
        self.model.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True
        )
        self.model.eval().requires_grad_(False).to(self.device)
        self.mel = AugmentMelSTFT(fmax=15000).eval().to(self.device)
        self.parameter_count = sum(parameter.numel() for parameter in self.model.parameters())
        self.free_vram_at_load = None
        self.peak_allocated_vram_bytes = 0
        if self.device.type == "cuda":
            self.free_vram_at_load = int(torch.cuda.mem_get_info(self.device)[0])

    def encode(self, waveforms: np.ndarray) -> np.ndarray:
        if waveforms.shape != (BATCH_SIZE, WINDOW_SAMPLES):
            raise ValueError(f"unexpected EfficientAT input shape: {waveforms.shape}")
        torch = self.torch
        values = torch.from_numpy(waveforms.copy()).to(self.device)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        with torch.inference_mode():
            _, features = self.model(self.mel(values).unsqueeze(1))
        result = features.float().cpu().numpy()
        if result.shape != (BATCH_SIZE, 384) or not np.isfinite(result).all():
            raise ValueError(f"invalid EfficientAT features: {result.shape}")
        if self.device.type == "cuda":
            self.peak_allocated_vram_bytes = max(
                self.peak_allocated_vram_bytes,
                int(torch.cuda.max_memory_allocated(self.device)),
            )
        return result


def _verify_media_receipt(media_dir: Path, receipt_path: Path, assets: list[dict[str, Any]]) -> str:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("status") != "PASS"
        or receipt.get("source_archive_sha256") != PINNED_ARCHIVE_SHA256
    ):
        raise ValueError("AVATAR media receipt does not match the pinned archive")
    records = {row["name"]: row for row in receipt.get("media_sha256", [])}
    needed = {row["asset_key"].rsplit("/", 1)[0] + ".mp4" for row in assets}
    if needed - set(records):
        raise ValueError("media receipt is missing clips required by the frozen audio inputs")
    for name in sorted(needed):
        path = media_dir / name
        record = records[name]
        if (
            not path.is_file()
            or path.stat().st_size != record["bytes"]
            or sha256_file(path) != record["sha256"]
        ):
            raise ValueError(f"extracted media failed receipt verification: {path}")
    return sha256_file(receipt_path)


def _run(args: argparse.Namespace) -> dict[str, Any]:
    train = args.train_input.resolve(strict=True)
    validation = args.validation_input.resolve(strict=True)
    media_dir = args.media_dir.resolve(strict=True)
    receipt_path = args.media_receipt.resolve(strict=True)
    source = args.source_root.resolve(strict=True)
    checkpoint = args.checkpoint.resolve(strict=True)
    dependencies = args.dependency_root.resolve(strict=True)
    assets = load_audio_assets(train, validation)
    if args.max_assets is not None:
        if args.max_assets < 1:
            raise ValueError("--max-assets must be positive")
        assets = assets[: args.max_assets]
    receipt_sha256 = _verify_media_receipt(media_dir, receipt_path, assets)
    identity = {
        "repository_head": subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parent.parent), "rev-parse", "HEAD"],
            text=True,
        ).strip(),
        "extractor_sha256": sha256_file(Path(__file__).resolve()),
        "train_input_sha256": sha256_file(train),
        "validation_input_sha256": sha256_file(validation),
        "media_receipt_sha256": receipt_sha256,
        "asset_order_sha256": canonical_sha256(assets),
        "asset_count": len(assets),
        "max_assets": args.max_assets,
        "model_revision": AUDIO_MODEL_REVISION,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "sample_rate_hz": SAMPLE_RATE,
        "window_seconds": WINDOW_SECONDS,
        "window_samples": WINDOW_SAMPLES,
        "feature_dim": 384,
        "batch_size": BATCH_SIZE,
        "input_preprocessing": (
            "decode source audio, PyAV resample to mono fltp 32kHz, trim resampler tail "
            "to actual audio-stream end, crop around annotation time, and zero-pad beyond "
            "available audio; retain metadata-padding differences for audit"
        ),
    }
    identity_hash = canonical_sha256(identity)
    output = args.output_dir.resolve()
    shards_dir = output / "shards"
    if output.exists():
        config_path = output / "effective-config.json"
        progress_path = output / "progress.json"
        if not config_path.is_file() or not progress_path.is_file() or not shards_dir.is_dir():
            raise FileExistsError(f"output exists without a valid resumable extraction: {output}")
        saved = json.loads(config_path.read_text(encoding="utf-8"))
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if saved.get("identity") != identity or progress.get("identity_sha256") != identity_hash:
            raise ValueError("existing audio output belongs to a different extraction identity")
        if progress.get("status") == "PASS":
            result = output / "result.json"
            if not result.is_file() or sha256_file(result) != progress.get("result_sha256"):
                raise ValueError("completed audio result failed hash validation")
            return json.loads(result.read_text(encoding="utf-8"))
        completed = int(progress.get("completed_assets", -1))
        shards = progress.get("shards", [])
        expected = (completed + BATCH_SIZE - 1) // BATCH_SIZE
        if (
            progress.get("status") != "RUNNING"
            or completed < 0
            or completed > len(assets)
            or (completed != len(assets) and completed % BATCH_SIZE != 0)
            or len(shards) != expected
        ):
            raise ValueError("audio progress metadata is inconsistent")
        for index, record in enumerate(shards):
            path = shards_dir / record["name"]
            first = index * BATCH_SIZE
            count = min(BATCH_SIZE, completed - first)
            if (
                record.get("first_asset_index") != first
                or record.get("asset_count") != count
                or not path.is_file()
                or sha256_file(path) != record.get("sha256")
            ):
                raise ValueError(f"audio shard failed resume validation: {path}")
            with np.load(path, allow_pickle=False) as saved_shard:
                if saved_shard["asset_keys"].tolist() != [
                    row["asset_key"] for row in assets[first : first + count]
                ]:
                    raise ValueError(f"audio shard IDs mismatch: {path}")
        known = {row["name"] for row in shards}
        if any(path.name not in known for path in shards_dir.glob("*.npz")):
            raise ValueError("untracked audio shard exists")
        start_asset = completed
    else:
        output.mkdir(parents=True)
        shards_dir.mkdir()
        write_json_atomic(
            output / "effective-config.json",
            {
                "identity": identity,
                "identity_sha256": identity_hash,
                "created_utc": datetime.now(UTC).isoformat(),
            },
        )
        progress = {
            "status": "RUNNING",
            "identity_sha256": identity_hash,
            "asset_count": len(assets),
            "completed_assets": 0,
            "shards": [],
        }
        write_json_atomic(output / "progress.json", progress)
        start_asset = 0

    model_start = time.perf_counter()
    encoder = EfficientATEncoder(source, checkpoint, dependencies)
    model_load_seconds = time.perf_counter() - model_start
    inference_start = time.perf_counter()
    pending_audio: list[np.ndarray] = []
    pending_keys: list[str] = []
    pending_centers: list[float] = []
    pending_padding: list[tuple[int, int]] = []
    pending_declared_padding: list[tuple[int, int]] = []
    decoded_waveform: np.ndarray | None = None
    completed = start_asset
    asset_indices = {row["asset_key"]: index for index, row in enumerate(assets)}

    def flush_batch() -> None:
        nonlocal completed, progress
        if not pending_audio:
            return
        count = len(pending_audio)
        batch = np.zeros((BATCH_SIZE, WINDOW_SAMPLES), dtype=np.float32)
        batch[:count] = np.stack(pending_audio)
        features = encoder.encode(batch)[:count]
        first = completed
        keys = pending_keys.copy()
        filename = f"shard-{first // BATCH_SIZE:05d}.npz"
        path = shards_dir / filename
        temporary = path.with_suffix(".npz.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                asset_keys=np.asarray(keys, dtype=np.str_),
                centers_seconds=np.asarray(pending_centers, dtype=np.float64),
                features=features,
                padding_samples=np.asarray(pending_padding, dtype=np.int64),
                declared_padding_samples=np.asarray(pending_declared_padding, dtype=np.int64),
            )
        temporary.replace(path)
        record = {
            "name": filename,
            "sha256": sha256_file(path),
            "first_asset_index": first,
            "asset_count": count,
        }
        progress["shards"].append(record)
        completed += count
        progress["completed_assets"] = completed
        progress["last_updated_utc"] = datetime.now(UTC).isoformat()
        write_json_atomic(output / "progress.json", progress)
        pending_audio.clear()
        pending_keys.clear()
        pending_centers.clear()
        pending_padding.clear()
        pending_declared_padding.clear()
        if len(progress["shards"]) % 32 == 0 or completed == len(assets):
            print(f"completed {completed}/{len(assets)} audio features", flush=True)

    decode_seconds = 0.0
    for clip, group in groupby(assets, key=lambda row: row["asset_key"].rsplit("/", 1)[0]):
        group_rows = list(group)
        last_index = asset_indices[group_rows[-1]["asset_key"]]
        if last_index < start_asset:
            continue
        clip_path = media_dir / f"{clip}.mp4"
        decode_start = time.perf_counter()
        decoded_waveform = decode_mono_32khz(clip_path)
        decode_seconds += time.perf_counter() - decode_start
        for row in group_rows:
            asset_index = asset_indices[row["asset_key"]]
            if asset_index < start_asset:
                continue
            expected_left = round(row["declared_left_padding_seconds"] * SAMPLE_RATE)
            expected_right = round(row["declared_right_padding_seconds"] * SAMPLE_RATE)
            waveform, left, right = crop_audio_window(decoded_waveform, row["center_seconds"])
            if left != expected_left:
                raise ValueError(
                    f"audio padding differs from frozen input for {row['asset_key']}: "
                    f"left padding got {left}, expected {expected_left}"
                )
            pending_audio.append(waveform)
            pending_keys.append(row["asset_key"])
            pending_centers.append(row["center_seconds"])
            pending_padding.append((left, right))
            pending_declared_padding.append((expected_left, expected_right))
            if len(pending_audio) == BATCH_SIZE:
                flush_batch()
    flush_batch()
    if completed != len(assets):
        raise ValueError(f"audio extraction ended at {completed}/{len(assets)} assets")

    all_keys: list[str] = []
    times: list[np.ndarray] = []
    features: list[np.ndarray] = []
    padding: list[np.ndarray] = []
    declared_padding: list[np.ndarray] = []
    for record in progress["shards"]:
        shard = shards_dir / record["name"]
        if sha256_file(shard) != record["sha256"]:
            raise ValueError(f"audio shard hash mismatch: {shard}")
        with np.load(shard, allow_pickle=False) as values:
            all_keys.extend(values["asset_keys"].tolist())
            times.append(values["centers_seconds"])
            features.append(values["features"])
            padding.append(values["padding_samples"])
            declared_padding.append(values["declared_padding_samples"])
    if all_keys != [row["asset_key"] for row in assets]:
        raise ValueError("final audio feature order differs from frozen input order")
    feature_path = output / "features.npz"
    with feature_path.open("xb") as stream:
        np.savez_compressed(
            stream,
            asset_keys=np.asarray(all_keys, dtype=np.str_),
            centers_seconds=np.concatenate(times),
            features=np.concatenate(features),
            padding_samples=np.concatenate(padding),
            declared_padding_samples=np.concatenate(declared_padding),
        )
    actual_padding = np.concatenate(padding)
    declared_padding_array = np.concatenate(declared_padding)
    padding_delta = actual_padding - declared_padding_array
    result = {
        "status": "PASS",
        "identity_sha256": identity_hash,
        "asset_count": len(all_keys),
        "unique_clip_count": len({key.rsplit("/", 1)[0] for key in all_keys}),
        "feature_shape": [len(all_keys), 384],
        "feature_file_sha256": sha256_file(feature_path),
        "feature_file_bytes": feature_path.stat().st_size,
        "device": str(encoder.device),
        "parameter_count": encoder.parameter_count,
        "initial_free_vram_bytes": encoder.free_vram_at_load,
        "peak_allocated_vram_bytes": (encoder.peak_allocated_vram_bytes),
        "model_load_seconds": model_load_seconds,
        "audio_decode_seconds": decode_seconds,
        "end_to_end_wall_seconds": time.perf_counter() - inference_start,
        "windows_with_audio_stream_padding_mismatch": int(np.any(padding_delta != 0, axis=1).sum()),
        "windows_with_audio_shorter_than_annotation_duration": int((padding_delta[:, 1] > 0).sum()),
        "windows_with_audio_longer_than_annotation_duration": int((padding_delta[:, 1] < 0).sum()),
        "max_abs_padding_delta_samples": int(np.abs(padding_delta).max(initial=0)),
        "sealed_audit_loaded": False,
    }
    result_path = output / "result.json"
    write_json_atomic(result_path, result)
    progress["status"] = "PASS"
    progress["result_sha256"] = sha256_file(result_path)
    write_json_atomic(output / "progress.json", progress)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-input", type=Path, required=True)
    parser.add_argument("--validation-input", type=Path, required=True)
    parser.add_argument("--media-dir", type=Path, required=True)
    parser.add_argument("--media-receipt", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dependency-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-assets", type=int)
    args = parser.parse_args()
    print(json.dumps(_run(args), indent=2), flush=True)


if __name__ == "__main__":
    main()
