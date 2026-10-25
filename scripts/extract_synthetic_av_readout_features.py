"""Extract and merge pinned frozen features for the synthetic AV comparison."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import platform
import subprocess
import sys
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tiny_omni_decision.synthetic_av_task import SEQUENCE_LENGTH, validate_manifest  # noqa: E402

SAMPLE_RATE_HZ = 32_000
VIDEO_MODEL_REVISION = "be719a406d563b66f0ac969e7c94bab8e997c81a"
VIDEO_SOURCE_COMMIT = "c6eba0ece3dd49a24ffc44582966997a4a59f1d8"
VIDEO_CHECKPOINT_SHA256 = "61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29"
AUDIO_SOURCE_COMMIT = "a425fdce92572e602a1d5634799bd9f1f2efa806"
AUDIO_CHECKPOINT_SHA256 = "49b0e63577021994257b3de2ac5f1686d71554b93c8149f9d2568ad5760b904e"
AUDIO_PARAMETER_COUNT = 1_961_239
CONFIG_PATH = ROOT / "configs/pretrained_reuse/synthetic_av_temporal_vs_conditional_v1.json"
VIDEO_PARAMETER_COUNT = 114_365_184


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_cuda_gate(device: str, environment: dict[str, str]) -> None:
    if device == "cuda" and not environment.get("TINY_OMNI_GPU_GATE_RESERVED_MIB"):
        raise RuntimeError("CUDA feature extraction requires the HotUGPU reservation")
    if device not in {"cpu", "cuda"}:
        raise ValueError(f"unsupported device: {device}")


def validate_video_features(sequence_ids: list[str], features: np.ndarray) -> None:
    values = np.asarray(features)
    if len(sequence_ids) == 0 or len(set(sequence_ids)) != len(sequence_ids):
        raise ValueError("video sequence IDs must be non-empty and unique")
    if values.shape != (len(sequence_ids), SEQUENCE_LENGTH, 768):
        raise ValueError(f"video feature shape must be [N,8,768], got {values.shape}")
    if values.dtype != np.float32 or not np.isfinite(values).all():
        raise ValueError("video features must be finite float32 values")


def validate_audio_features(symbol_ids: list[int], features: np.ndarray) -> None:
    values = np.asarray(features)
    if symbol_ids != [0, 1, 2, 3]:
        raise ValueError("audio symbol IDs must be ordered exactly [0,1,2,3]")
    if values.shape != (4, 384):
        raise ValueError(f"audio feature shape must be [4,384], got {values.shape}")
    if values.dtype != np.float32 or not np.isfinite(values).all():
        raise ValueError("audio features must be finite float32 values")


def audio_segment_hash(samples: np.ndarray) -> str:
    values = np.asarray(samples)
    if values.ndim != 1 or values.dtype != np.int16:
        raise ValueError("audio segments must be mono int16 arrays")
    return hashlib.sha256(values.astype("<i2", copy=False).tobytes()).hexdigest()


def validate_rendered_manifest(manifest: dict[str, Any], input_dir: Path) -> None:
    validate_manifest(manifest)
    videos = manifest.get("assets", {}).get("videos", {})
    audio = manifest.get("assets", {}).get("audio", {})
    if not videos or set(videos) != set(audio):
        raise ValueError("video and audio sequence asset identities must match")
    sequence_split: dict[str, str] = {}
    for group in manifest["groups"]:
        for sequence_id in group["sequence_ids"].values():
            previous = sequence_split.setdefault(sequence_id, group["split"])
            if previous != group["split"]:
                raise ValueError(f"sequence identity crosses splits: {sequence_id}")
    for sequence_id, record in videos.items():
        video_path = Path(input_dir) / record["path"]
        if not video_path.is_file() or _sha256(video_path) != record["sha256"]:
            raise ValueError(f"video asset hash mismatch: {sequence_id}")
        if record.get("frame_count") != SEQUENCE_LENGTH or len(record.get("frame_paths", [])) != 8:
            raise ValueError(f"video asset must contain eight ordered frames: {sequence_id}")
        if len(record.get("frame_sha256", [])) != SEQUENCE_LENGTH:
            raise ValueError(f"video frame hashes are incomplete: {sequence_id}")
        for relative_path, expected_hash in zip(
            record["frame_paths"], record["frame_sha256"], strict=True
        ):
            frame_path = Path(input_dir) / relative_path
            if not frame_path.is_file() or _sha256(frame_path) != expected_hash:
                raise ValueError(f"video frame hash mismatch: {relative_path}")
        audio_record = audio[sequence_id]
        audio_path = Path(input_dir) / audio_record["path"]
        if not audio_path.is_file() or _sha256(audio_path) != audio_record["sha256"]:
            raise ValueError(f"audio asset hash mismatch: {sequence_id}")
        if (
            audio_record.get("sample_rate_hz") != SAMPLE_RATE_HZ
            or audio_record.get("sample_count") != SEQUENCE_LENGTH * 16_000
            or len(audio_record.get("segments", [])) != SEQUENCE_LENGTH
        ):
            raise ValueError(f"audio asset has invalid time segmentation: {sequence_id}")


def audio_segments_by_symbol(
    manifest: dict[str, Any], input_dir: Path, sequence_ids: list[str] | None = None
) -> dict[int, np.ndarray]:
    segments: dict[int, np.ndarray] = {}
    segment_hashes: dict[int, str] = {}
    assets = manifest["assets"]["audio"]
    if sequence_ids is not None:
        assets = {key: assets[key] for key in sequence_ids}
    for record in assets.values():
        path = Path(input_dir) / record["path"]
        with wave.open(str(path), "rb") as source:
            if (
                source.getnchannels() != 1
                or source.getsampwidth() != 2
                or source.getframerate() != SAMPLE_RATE_HZ
            ):
                raise ValueError(f"unsupported generated WAV representation: {path}")
            for segment in record["segments"]:
                count = int(segment["sample_end"]) - int(segment["sample_start"])
                source.setpos(int(segment["sample_start"]))
                samples = np.frombuffer(source.readframes(count), dtype="<i2").copy()
                symbol_id = int(segment["symbol_id"])
                digest = audio_segment_hash(samples)
                if symbol_id in segment_hashes and segment_hashes[symbol_id] != digest:
                    raise ValueError(f"same audio symbol has different waveform bytes: {symbol_id}")
                segment_hashes[symbol_id] = digest
                segments.setdefault(symbol_id, samples)
    if sorted(segments) != [0, 1, 2, 3]:
        raise ValueError("audio manifest must contain all four tone symbols")
    return segments


def _git_commit(source: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()


def _require_clean_commit(source: Path, expected: str) -> str:
    commit = _git_commit(source)
    status = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain"], text=True
    ).strip()
    validate_source_status(status)
    if commit != expected:
        raise ValueError(f"source checkout must be at {expected}; got {commit}")
    return commit


def validate_source_status(status: str) -> None:
    generated_install_paths = {
        "build/",
        "videoprism.egg-info/",
        "videoprism/__pycache__/",
    }
    unexpected = []
    for line in status.splitlines():
        if line.startswith("?? ") and line[3:] in generated_install_paths:
            continue
        unexpected.append(line)
    if unexpected:
        raise ValueError(f"source checkout has unexpected changes: {'; '.join(unexpected)}")


def _selected_groups(
    manifest: dict[str, Any], limit_per_split: int | None
) -> list[dict[str, Any]]:
    split_groups: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    for group in manifest["groups"]:
        split_groups[group["split"]].append(group)
    if limit_per_split is not None:
        if limit_per_split < 1:
            raise ValueError("limit-groups-per-split must be positive")
        split_groups = {key: value[:limit_per_split] for key, value in split_groups.items()}
    selected = split_groups["train"] + split_groups["validation"]
    if not split_groups["train"] or not split_groups["validation"]:
        raise ValueError("feature extraction requires both train and validation sequences")
    return selected


def _selected_sequence_ids(groups: list[dict[str, Any]]) -> list[str]:
    sequences = [sequence for group in groups for sequence in group["sequence_ids"].values()]
    return list(dict.fromkeys(sequences))


def extract_video(args: argparse.Namespace) -> dict[str, Any]:
    input_dir = args.input_dir.resolve(strict=True)
    checkpoint = args.checkpoint.resolve(strict=True)
    source = args.source.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite feature output: {output}")
    if _sha256(checkpoint) != VIDEO_CHECKPOINT_SHA256:
        raise ValueError("VideoPrism checkpoint SHA-256 mismatch")
    source_commit = _require_clean_commit(source, VIDEO_SOURCE_COMMIT)
    manifest = json.loads((input_dir / "render-manifest.json").read_text(encoding="utf-8"))
    validate_rendered_manifest(manifest, input_dir)
    groups = _selected_groups(manifest, args.limit_groups_per_split)
    selected = _selected_sequence_ids(groups)

    import jax
    import jax.numpy as jnp
    import tensorflow as tf
    from videoprism import models as vp

    devices = [str(device) for device in jax.devices()]
    if not devices or any("CPU" not in device.upper() for device in devices):
        raise RuntimeError(
            f"this Windows run requires the pinned CPU VideoPrism backend: {devices}"
        )
    model = vp.get_model("videoprism_public_v1_base")
    variables = vp.load_pretrained_weights(
        "videoprism_public_v1_base", checkpoint_path=str(checkpoint)
    )
    encode = jax.jit(lambda video: model.apply(variables, video, train=False))
    from scripts.extract_frozen_videoprism_features import _decode_video, _preprocess
    from scripts.extract_videoprism_framewise_features import pool_framewise_tokens

    started = time.perf_counter()
    feature_rows = []
    sequences = manifest["assets"]["videos"]
    for sequence_id in selected:
        record = sequences[sequence_id]
        frames = _decode_video(input_dir / record["path"])
        if frames.shape[0] != SEQUENCE_LENGTH:
            raise ValueError(f"decoded frame count mismatch: {sequence_id}")
        result = encode(jnp.asarray(_preprocess(frames, tf)[None, ...]))
        jax.tree_util.tree_map(
            lambda value: (
                value.block_until_ready() if hasattr(value, "block_until_ready") else value
            ),
            result,
        )
        tokens = result[0] if isinstance(result, tuple) else result
        feature_rows.append(pool_framewise_tokens(np.asarray(tokens))[None, ...][0])

    features = np.stack(feature_rows).astype(np.float32, copy=False)
    validate_video_features(selected, features)
    output.mkdir(parents=True, exist_ok=False)
    cache = output / "video-features.npz"
    np.savez_compressed(cache, sequence_ids=np.asarray(selected), features=features)
    report = {
        "status": "PASS_SYNTHETIC_AV_VIDEOPRISM_FEATURES",
        "model_id": "google/videoprism-base-f16r288",
        "model_revision": VIDEO_MODEL_REVISION,
        "source_commit": source_commit,
        "checkpoint_sha256": VIDEO_CHECKPOINT_SHA256,
        "parameter_count": VIDEO_PARAMETER_COUNT,
        "extractor_sha256": _sha256(Path(__file__)),
        "experiment_config_sha256": _sha256(CONFIG_PATH),
        "input_manifest_sha256": _sha256(input_dir / "render-manifest.json"),
        "output_sha256": _sha256(cache),
        "feature_layout": "[unique_sequence, ordered_frame, 768]",
        "sequence_ids": selected,
        "feature_shape": list(features.shape),
        "feature_dtype": str(features.dtype),
        "devices": devices,
        "python_version": platform.python_version(),
        "jax_version": jax.__version__,
        "tensorflow_version": tf.__version__,
        "elapsed_seconds": time.perf_counter() - started,
        "smoke_subset": args.limit_groups_per_split is not None,
    }
    (output / "feature-report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def extract_audio(args: argparse.Namespace) -> dict[str, Any]:
    require_cuda_gate(args.device, os.environ)
    input_dir = args.input_dir.resolve(strict=True)
    checkpoint = args.checkpoint.resolve(strict=True)
    source = args.source.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite feature output: {output}")
    if _sha256(checkpoint) != AUDIO_CHECKPOINT_SHA256:
        raise ValueError("EfficientAT checkpoint SHA-256 mismatch")
    source_commit = _require_clean_commit(source, AUDIO_SOURCE_COMMIT)
    manifest = json.loads((input_dir / "render-manifest.json").read_text(encoding="utf-8"))
    validate_rendered_manifest(manifest, input_dir)
    groups = _selected_groups(manifest, args.limit_groups_per_split)
    selected = _selected_sequence_ids(groups)
    segments = audio_segments_by_symbol(manifest, input_dir, selected)

    import torch
    import torchaudio

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; refusing GPU feature extraction")
        torch.cuda.set_device(0)
    sys.path.insert(0, str(source))
    with contextlib.redirect_stdout(io.StringIO()):
        from models.dymn.model import get_model
        from models.preprocess import AugmentMelSTFT

    with contextlib.redirect_stdout(io.StringIO()):
        model = get_model(width_mult=0.4, pretrained_name=None, T_max=1.0)
    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True
    )
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters != AUDIO_PARAMETER_COUNT:
        raise ValueError(f"unexpected EfficientAT parameter count: {parameters}")
    model.eval().requires_grad_(False).to(device)
    mel = AugmentMelSTFT(freqm=0, timem=0).eval().requires_grad_(False).to(device)
    before_state = _torch_state_sha(model)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    feature_rows = []
    for symbol_id in range(4):
        waveform = torch.from_numpy(segments[symbol_id].astype(np.float32) / 32768.0).unsqueeze(0)
        mel_input = mel(waveform.to(device)).unsqueeze(1)
        with torch.inference_mode():
            logits, embedding = model(mel_input)
        if logits.shape != (1, 527) or embedding.shape != (1, 384):
            raise ValueError(f"unexpected EfficientAT output shapes for tone {symbol_id}")
        feature_rows.append(embedding.squeeze(0).float().cpu().numpy())
    features = np.stack(feature_rows).astype(np.float32, copy=False)
    validate_audio_features([0, 1, 2, 3], features)
    after_state = _torch_state_sha(model)
    if before_state != after_state:
        raise RuntimeError("frozen EfficientAT weights changed during extraction")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    output.mkdir(parents=True, exist_ok=False)
    cache = output / "audio-features.npz"
    np.savez_compressed(cache, symbol_ids=np.arange(4), features=features)
    report = {
        "status": "PASS_SYNTHETIC_AV_EFFICIENTAT_FEATURES",
        "model_id": "efficientat-dymn04-as",
        "source_commit": source_commit,
        "checkpoint_sha256": AUDIO_CHECKPOINT_SHA256,
        "extractor_sha256": _sha256(Path(__file__)),
        "experiment_config_sha256": _sha256(CONFIG_PATH),
        "parameter_count": parameters,
        "model_state_sha256_before": before_state,
        "model_state_sha256_after": after_state,
        "input_manifest_sha256": _sha256(input_dir / "render-manifest.json"),
        "output_sha256": _sha256(cache),
        "feature_layout": "[four_tone_symbols, 384]",
        "feature_shape": list(features.shape),
        "feature_dtype": str(features.dtype),
        "segment_sample_hashes": {
            str(symbol_id): audio_segment_hash(segments[symbol_id]) for symbol_id in range(4)
        },
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torchaudio_version": torchaudio.__version__,
        "elapsed_seconds": time.perf_counter() - started,
        "smoke_subset": args.limit_groups_per_split is not None,
        "peak_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
        "peak_reserved_bytes": (
            torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
        ),
        "gpu_gate": {
            key: value for key, value in os.environ.items() if key.startswith("TINY_OMNI_GPU_GATE_")
        }
        if device.type == "cuda"
        else None,
    }
    (output / "feature-report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def _torch_state_sha(model: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def merge_feature_caches(
    manifest: dict[str, Any],
    video_ids: list[str],
    video_features: np.ndarray,
    audio_ids: list[int],
    audio_features: np.ndarray,
) -> dict[str, np.ndarray]:
    validate_video_features(video_ids, video_features)
    validate_audio_features(audio_ids, audio_features)
    video_index = {sequence_id: index for index, sequence_id in enumerate(video_ids)}
    audio_index = {symbol_id: index for index, symbol_id in enumerate(audio_ids)}
    covered_ids = set(video_ids)
    examples = [
        row
        for row in manifest["examples"]
        if row["audio_sequence_id"] in covered_ids
        and row["video_sequence_id"] in covered_ids
    ]
    if not examples:
        raise ValueError("feature cache covers no complete groups")
    group_counts: dict[str, int] = {}
    for row in examples:
        group_counts[row["group_id"]] = group_counts.get(row["group_id"], 0) + 1
    if any(count != 8 for count in group_counts.values()):
        raise ValueError("feature cache must retain complete groups of eight examples")
    audio_indices = np.asarray(
        [[audio_index[symbol] for symbol in row["audio_symbols"]] for row in examples],
        dtype=np.int64,
    )
    video_indices = np.asarray(
        [video_index[row["video_sequence_id"]] for row in examples], dtype=np.int64
    )
    return {
        "example_ids": np.asarray([row["sample_id"] for row in examples]),
        "group_ids": np.asarray([row["group_id"] for row in examples]),
        "splits": np.asarray([row["split"] for row in examples]),
        "query_polarity": np.asarray(
            [int(row["query_polarity"] == "different_order") for row in examples],
            dtype=np.int64,
        ),
        "targets": np.asarray([row["target"] for row in examples], dtype=np.int64),
        "audio_symbol_indices": audio_indices,
        "video_sequence_indices": video_indices,
        "video_sequence_ids": np.asarray(video_ids),
        "video_features": np.asarray(video_features, dtype=np.float32),
        "audio_symbol_ids": np.asarray(audio_ids, dtype=np.int64),
        "audio_features": np.asarray(audio_features, dtype=np.float32),
    }


def merge_feature_cache_dirs(args: argparse.Namespace) -> dict[str, Any]:
    input_dir = args.input_dir.resolve(strict=True)
    video_dir = args.video_features.resolve(strict=True)
    audio_dir = args.audio_features.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite merged feature output: {output}")
    manifest_path = input_dir / "render-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_rendered_manifest(manifest, input_dir)
    video_report = json.loads((video_dir / "feature-report.json").read_text(encoding="utf-8"))
    audio_report = json.loads((audio_dir / "feature-report.json").read_text(encoding="utf-8"))
    if video_report.get("status") != "PASS_SYNTHETIC_AV_VIDEOPRISM_FEATURES":
        raise ValueError("VideoPrism feature receipt is incomplete")
    if audio_report.get("status") != "PASS_SYNTHETIC_AV_EFFICIENTAT_FEATURES":
        raise ValueError("EfficientAT feature receipt is incomplete")
    expected_manifest_hash = _sha256(manifest_path)
    if (
        video_report.get("input_manifest_sha256") != expected_manifest_hash
        or audio_report.get("input_manifest_sha256") != expected_manifest_hash
    ):
        raise ValueError("feature caches do not refer to the same rendered manifest")
    video_path = video_dir / "video-features.npz"
    audio_path = audio_dir / "audio-features.npz"
    if _sha256(video_path) != video_report.get("output_sha256"):
        raise ValueError("VideoPrism feature cache hash mismatch")
    if _sha256(audio_path) != audio_report.get("output_sha256"):
        raise ValueError("EfficientAT feature cache hash mismatch")
    with np.load(video_path, allow_pickle=False) as archive:
        video_ids = archive["sequence_ids"].astype(str).tolist()
        video_features = archive["features"].copy()
    with np.load(audio_path, allow_pickle=False) as archive:
        audio_ids = archive["symbol_ids"].astype(int).tolist()
        audio_features = archive["features"].copy()
    merged = merge_feature_caches(
        manifest, video_ids, video_features, audio_ids, audio_features
    )
    output.mkdir(parents=True, exist_ok=False)
    cache_path = output / "examples.npz"
    np.savez_compressed(cache_path, **merged)
    example_count = len(merged["example_ids"])
    report = {
        "status": "PASS_SYNTHETIC_AV_ALIGNED_FEATURE_CACHE",
        "input_manifest_sha256": expected_manifest_hash,
        "experiment_config_sha256": _sha256(CONFIG_PATH),
        "video_feature_sha256": video_report["output_sha256"],
        "audio_feature_sha256": audio_report["output_sha256"],
        "output_sha256": _sha256(cache_path),
        "example_count": example_count,
        "video_sequence_count": len(video_ids),
        "audio_symbol_count": len(audio_ids),
        "feature_shapes": {key: list(value.shape) for key, value in merged.items()},
        "selected_group_ids": sorted(set(merged["group_ids"].astype(str).tolist())),
        "train_example_count": int(np.count_nonzero(merged["splits"] == "train")),
        "validation_example_count": int(np.count_nonzero(merged["splits"] == "validation")),
        "smoke_subset": example_count != len(manifest["examples"]),
    }
    (output / "feature-report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="modality", required=True)
    for modality in ("video", "audio"):
        stage = subparsers.add_parser(modality)
        stage.add_argument("--input-dir", type=Path, required=True)
        stage.add_argument("--checkpoint", type=Path, required=True)
        stage.add_argument("--source", type=Path, required=True)
        stage.add_argument("--output", type=Path, required=True)
        stage.add_argument("--limit-groups-per-split", type=int)
        if modality == "audio":
            stage.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    merge = subparsers.add_parser("merge")
    merge.add_argument("--input-dir", type=Path, required=True)
    merge.add_argument("--video-features", type=Path, required=True)
    merge.add_argument("--audio-features", type=Path, required=True)
    merge.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.modality == "video":
        result = extract_video(args)
    elif args.modality == "audio":
        result = extract_audio(args)
    else:
        result = merge_feature_cache_dirs(args)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
