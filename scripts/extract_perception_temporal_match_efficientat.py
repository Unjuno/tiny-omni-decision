"""Extract audio candidate embeddings under the cooperative HotUGPU gate."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np

WEIGHTS_SHA256 = "49b0e63577021994257b3de2ac5f1686d71554b93c8149f9d2568ad5760b904e"
SOURCE_COMMIT = "a425fdce92572e602a1d5634799bd9f1f2efa806"
MODEL_PARAMETERS = 1_961_239
SAMPLE_RATE = 32_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def completed_run_metadata(
    initial: dict[str, Any], result: dict[str, Any], result_sha256: str
) -> dict[str, Any]:
    """Publish terminal status only after the complete feature receipt is durable."""
    return {
        **initial,
        "status": result["status"],
        "result_sha256": result_sha256,
        "examples": result["examples"],
        "candidate_audio_windows": result["candidate_audio_windows"],
        "elapsed_seconds": result["elapsed_seconds"],
    }


def read_audio_window(path: Path, window: list[float], torchaudio: Any, torch: Any):
    start_s, end_s = map(float, window)
    if end_s <= start_s or abs((end_s - start_s) - 2.0) > 1e-5:
        raise ValueError(f"candidate audio window must be exactly 2 seconds: {window}")
    with wave.open(str(path), "rb") as source:
        rate = source.getframerate()
        channels = source.getnchannels()
        frames = source.getnframes()
        if (
            source.getsampwidth() != 4
            or source.getcomptype() != "NONE"
            or rate not in (44_100, 48_000)
            or channels not in (1, 2)
        ):
            raise ValueError(f"unsupported WAV representation: {path}")
        first = round(start_s * rate)
        last = round(end_s * rate)
        if first < 0 or last > frames or last <= first:
            raise ValueError(
                f"audio crop is outside media duration: {path}:{first}:{last}/{frames}"
            )
        source.setpos(first)
        raw = source.readframes(last - first)
    pcm = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2_147_483_648.0
    if pcm.size != (last - first) * channels:
        raise ValueError(f"truncated WAV crop: {path}")
    mono = pcm.reshape(-1, channels).mean(axis=1)
    waveform = torch.from_numpy(mono.copy()).unsqueeze(0)
    if rate != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, orig_freq=rate, new_freq=SAMPLE_RATE)
    expected = round((end_s - start_s) * SAMPLE_RATE)
    if abs(waveform.shape[-1] - expected) > 1:
        raise ValueError(f"resampled crop length mismatch: {waveform.shape[-1]} != {expected}")
    waveform = waveform[..., :expected]
    return waveform, {
        "source_sample_rate_hz": rate,
        "channels": channels,
        "source_sample_range": [first, last],
        "resampled_sample_count": int(waveform.shape[-1]),
        "window_s": [start_s, end_s],
    }


def load_hash_maps(old_media: Path, train_report: Path, fresh_valid_report: Path):
    old = json.loads((old_media / "acquisition_manifest.json").read_text(encoding="utf-8"))
    train = json.loads(train_report.read_text(encoding="utf-8"))
    fresh = json.loads(fresh_valid_report.read_text(encoding="utf-8"))
    old_hashes = {
        (row["split"], row["video_id"], Path(row["member"]).suffix): row["sha256"]
        for row in old["files"]
    }
    train_hashes = {
        (row["split"], row["video_id"], Path(row["member"]).suffix): row["sha256"]
        for row in train["files"]
    }
    fresh_hashes = {
        ("valid", row["video_id"], ".wav"): row["audio_sha256"]
        for row in fresh["examples_manifest"]
    }
    return old_hashes, train_hashes, fresh_hashes


def find_audio(
    split: str,
    video_id: str,
    *,
    old_media: Path,
    added_train_media: Path,
    fresh_valid_media: Path,
    old_hashes: dict[tuple[str, str, str], str],
    new_train_hashes: dict[tuple[str, str, str], str],
    fresh_valid_hashes: dict[tuple[str, str, str], str],
) -> tuple[Path, str]:
    candidates = [(old_media / split / f"{video_id}.wav", old_hashes)]
    if split == "train":
        candidates.append((added_train_media / "train" / f"{video_id}.wav", new_train_hashes))
    else:
        candidates.append((fresh_valid_media / "valid" / f"{video_id}.wav", fresh_valid_hashes))
    key = (split, video_id, ".wav")
    for path, hashes in candidates:
        if path.is_file() and key in hashes:
            return path, hashes[key]
    raise FileNotFoundError(f"no pinned audio asset for {split}:{video_id}")


def state_sha(model: Any) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def device_name(torch: Any, device: Any) -> str:
    return torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"


def cuda_memory_metrics(torch: Any, device: Any) -> tuple[int | None, int | None]:
    if device.type != "cuda":
        return None, None
    return torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.device == "cuda" and not os.environ.get("TINY_OMNI_GPU_GATE_RESERVED_MIB"):
        raise RuntimeError("audio feature extraction must run under run_with_hotgpu_guard.py")
    manifest_path = args.manifest.resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    weights = args.weights.resolve(strict=True)
    source = args.source.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing feature output: {output}")
    if sha256_file(weights) != WEIGHTS_SHA256:
        raise ValueError("EfficientAT checkpoint SHA mismatch")
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if (
        commit != SOURCE_COMMIT
        or subprocess.check_output(
            ["git", "-C", str(source), "diff", "--name-only"], text=True
        ).strip()
    ):
        raise ValueError("EfficientAT source checkout is not pristine at the pinned commit")

    import torch
    import torchaudio

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; refusing GPU audio feature extraction")
        torch.cuda.set_device(0)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    sys.path.insert(0, str(source))
    with contextlib.redirect_stdout(io.StringIO()):
        from models.dymn.model import get_model
        from models.preprocess import AugmentMelSTFT
    model = get_model(width_mult=0.4, pretrained_name=None, T_max=1.0)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters != MODEL_PARAMETERS:
        raise ValueError(f"unexpected EfficientAT parameter count: {parameters}")
    model.eval().requires_grad_(False).to(device)
    mel = AugmentMelSTFT(freqm=0, timem=0).eval().requires_grad_(False).to(device)
    before_state = state_sha(model)
    old_hashes, train_hashes, fresh_hashes = load_hash_maps(
        args.old_media, args.train_acquisition, args.fresh_valid_media_report
    )
    rows = [row for split in ("train", "validation") for row in manifest[split]]
    candidate_count = sum(len(row["options"]) for row in rows)
    output.mkdir(parents=True)
    run_metadata = {
        "status": "RUNNING",
        "manifest_sha256": sha256_file(manifest_path),
        "weights_sha256": WEIGHTS_SHA256,
        "source_commit": commit,
        "parameters": parameters,
        "device": device_name(torch, device),
        "dtype": str(next(model.parameters()).dtype),
        "source_rate_hz_after_resample": SAMPLE_RATE,
        "input_window_seconds": 2.0,
        "examples": len(rows),
        "candidate_audio_windows": candidate_count,
        "gpu_gate": {
            key: value for key, value in os.environ.items() if key.startswith("TINY_OMNI_GPU_GATE_")
        } if device.type == "cuda" else None,
        "validation_scope": "development-only",
    }
    atomic_json(output / "run.json", run_metadata)
    metadata = []
    started = time.perf_counter()
    for row in rows:
        split = "train" if row["split"] == "train" else "valid"
        audio_path, expected_hash = find_audio(
            split,
            row["video_id"],
            old_media=args.old_media,
            added_train_media=args.added_train_media,
            fresh_valid_media=args.fresh_valid_media,
            old_hashes=old_hashes,
            new_train_hashes=train_hashes,
            fresh_valid_hashes=fresh_hashes,
        )
        if sha256_file(audio_path) != expected_hash:
            raise ValueError(f"audio media SHA mismatch: {audio_path}")
        for option in row["options"]:
            output_stem = f"{row['video_id']}__{option['event_id']}"
            feature_path = output / row["split"] / f"{output_stem}.npy"
            meta_path = output / row["split"] / f"{output_stem}.json"
            (output / row["split"]).mkdir(exist_ok=True)
            if feature_path.exists() or meta_path.exists():
                raise FileExistsError(
                    f"unexpected existing output for {row['sample_id']}:{option['event_id']}"
                )
            waveform, crop_meta = read_audio_window(
                audio_path, option["audio_window_s"], torchaudio, torch
            )
            crop_sha = hashlib.sha256(
                waveform.squeeze(0).contiguous().numpy().astype("<f4", copy=False).tobytes()
            ).hexdigest()
            mel_input = mel(waveform.to(device)).unsqueeze(1)
            if not torch.isfinite(mel_input).all():
                raise ValueError(f"non-finite mel input: {row['sample_id']}:{option['event_id']}")
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            tick = time.perf_counter()
            with torch.inference_mode():
                logits, embedding = model(mel_input)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - tick
            vector = embedding.squeeze(0).float().cpu().numpy()
            if logits.shape != (1, 527) or vector.shape != (384,) or not np.isfinite(vector).all():
                raise ValueError(
                    f"invalid EfficientAT output: {row['sample_id']}:{option['event_id']}"
                )
            feature_hash = hashlib.sha256(vector.tobytes()).hexdigest()
            peak_allocated, peak_reserved = cuda_memory_metrics(torch, device)
            with feature_path.open("xb") as target:
                np.save(target, vector, allow_pickle=False)
            record = {
                "sample_id": row["sample_id"],
                "split": row["split"],
                "video_id": row["video_id"],
                "event_id": option["event_id"],
                "role": option["role"],
                "target_index": row["target_index"],
                "audio_window_s": option["audio_window_s"],
                "media_audio_sha256": expected_hash,
                "resampled_window_pcm_sha256": crop_sha,
                **crop_meta,
                "feature_sha256": feature_hash,
                "feature_dim": 384,
                "feature_path": str(feature_path),
                "forward_seconds": elapsed,
                "peak_allocated_bytes": peak_allocated,
                "peak_reserved_bytes": peak_reserved,
            }
            atomic_json(meta_path, record)
            metadata.append(record)
            if len(metadata) % 32 == 0:
                print(
                    json.dumps(
                        {
                            "completed_candidate_windows": len(metadata),
                            "total_candidate_windows": candidate_count,
                            "elapsed_seconds": time.perf_counter() - started,
                            "mean_forward_seconds": float(
                                np.mean([item["forward_seconds"] for item in metadata[-32:]])
                            ),
                            "latest_sample_id": row["sample_id"],
                            "peak_allocated_bytes": max(
                                item["peak_allocated_bytes"] for item in metadata
                                if item["peak_allocated_bytes"] is not None
                            ) if device.type == "cuda" else None,
                        }
                    ),
                    flush=True,
                )
    after_state = state_sha(model)
    if before_state != after_state:
        raise RuntimeError("frozen EfficientAT parameters changed during feature extraction")
    result = {
        "status": "PASS_TEMPORAL_MATCH_EFFICIENTAT_FEATURES",
        "manifest_sha256": sha256_file(manifest_path),
        "weights_sha256": WEIGHTS_SHA256,
        "source_commit": commit,
        "device": device_name(torch, device),
        "parameters": parameters,
        "model_state_sha256_before": before_state,
        "model_state_sha256_after": after_state,
        "examples": len(rows),
        "candidate_audio_windows": len(metadata),
        "expected_candidate_audio_windows": candidate_count,
        "peak_allocated_bytes": max(
            item["peak_allocated_bytes"] for item in metadata
            if item["peak_allocated_bytes"] is not None
        ) if device.type == "cuda" else None,
        "peak_reserved_bytes": max(
            item["peak_reserved_bytes"] for item in metadata
            if item["peak_reserved_bytes"] is not None
        ) if device.type == "cuda" else None,
        "mean_forward_seconds": float(np.mean([item["forward_seconds"] for item in metadata])),
        "elapsed_seconds": time.perf_counter() - started,
        "gpu_gate": {
            key: value for key, value in os.environ.items() if key.startswith("TINY_OMNI_GPU_GATE_")
        } if device.type == "cuda" else None,
        "records": metadata,
        "validation_scope": "development-only; not blind evaluation",
    }
    result_path = output / "result.json"
    atomic_json(result_path, result)
    atomic_json(
        output / "run.json",
        completed_run_metadata(run_metadata, result, sha256_file(result_path)),
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--old-media", type=Path, required=True)
    parser.add_argument("--added-train-media", type=Path, required=True)
    parser.add_argument("--fresh-valid-media", type=Path, required=True)
    parser.add_argument("--train-acquisition", type=Path, required=True)
    parser.add_argument("--fresh-valid-media-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2), flush=True)


if __name__ == "__main__":
    main()
