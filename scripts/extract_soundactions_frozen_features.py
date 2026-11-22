"""Extract label-blind audio or video sequences for allowed SoundActions splits."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from scripts.prepare_soundactions_media import CLASSES, sha256_file

VIDEO_SOURCE_COMMIT = "c6eba0ece3dd49a24ffc44582966997a4a59f1d8"
VIDEO_CHECKPOINT_SHA256 = "61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29"
AUDIO_SOURCE_COMMIT = "a425fdce92572e602a1d5634799bd9f1f2efa806"
AUDIO_CHECKPOINT_SHA256 = "49b0e63577021994257b3de2ac5f1686d71554b93c8149f9d2568ad5760b904e"
AUDIO_SAMPLE_RATE = 32_000
AUDIO_WINDOW_SAMPLES = 2 * AUDIO_SAMPLE_RATE
AUDIO_STRIDE_SAMPLES = AUDIO_SAMPLE_RATE
VIDEO_FRAMES = 8


def audio_window_starts(sample_count: int) -> list[int]:
    if sample_count < 1:
        raise ValueError("audio must contain samples")
    if sample_count <= AUDIO_WINDOW_SAMPLES:
        return [0]
    last_start = sample_count - AUDIO_WINDOW_SAMPLES
    starts = list(range(0, last_start + 1, AUDIO_STRIDE_SAMPLES))
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def _read_manifest(media_root: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    report_path = media_root / "acquisition-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "PASS_TRAIN_VALIDATION_ONLY":
        raise ValueError("media acquisition is not a verified train/validation-only package")
    if report.get("sealed_audit_read_or_extracted") is not False:
        raise ValueError("sealed-audit media was accessed during acquisition")
    rows = [
        json.loads(line)
        for line in (media_root / "train-validation.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [int(row["recording_id"]) for row in rows]
    if len(ids) != len(set(ids)) or not rows:
        raise ValueError("media index is empty or has duplicate recording IDs")
    if any(row.get("split") not in {"train", "validation"} for row in rows):
        raise ValueError("media index contains a forbidden split")
    if any(row.get("target") not in CLASSES for row in rows):
        raise ValueError("media index contains an unsupported label")
    return rows, report


def _verify_media(media_root: Path, rows: list[dict[str, object]]) -> None:
    for row in rows:
        for modality in ("audio", "video"):
            relative = Path(str(row[f"{modality}_path"]))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("media index path escapes its root")
            path = media_root / relative
            if not path.is_file() or sha256_file(path) != row[f"{modality}_sha256"]:
                raise ValueError(f"media hash mismatch: {path}")


def _run_audio(rows: list[dict[str, object]], media_root: Path, args) -> dict[str, object]:
    from scripts.extract_avatar_efficientat_features import (
        BATCH_SIZE,
        WINDOW_SAMPLES,
        EfficientATEncoder,
        decode_mono_32khz,
    )

    if WINDOW_SAMPLES != AUDIO_WINDOW_SAMPLES:
        raise ValueError("EfficientAT audio window contract changed")
    source = args.efficientat_source.resolve(strict=True)
    checkpoint = args.audio_checkpoint.resolve(strict=True)
    if sha256_file(checkpoint) != AUDIO_CHECKPOINT_SHA256:
        raise ValueError("EfficientAT checkpoint hash mismatch")
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True
    ).strip()
    if commit != AUDIO_SOURCE_COMMIT or dirty:
        raise ValueError("EfficientAT source checkout differs from the pinned pristine revision")
    with contextlib.redirect_stdout(io.StringIO()):
        encoder = EfficientATEncoder(source, checkpoint, args.dependency_root.resolve(strict=True))
    import torchaudio

    if encoder.parameter_count != 1_961_239 or any(
        p.requires_grad for p in encoder.model.parameters()
    ):
        raise ValueError("EfficientAT parameter count or frozen-state check failed")

    all_features = []
    offsets = [0]
    processed_windows = 0
    for index, row in enumerate(rows, start=1):
        path = media_root / str(row["audio_path"])
        waveform = decode_mono_32khz(path)
        starts = audio_window_starts(len(waveform))
        sample_features = []
        for first in range(0, len(starts), BATCH_SIZE):
            batch_starts = starts[first : first + BATCH_SIZE]
            batch = np.zeros((BATCH_SIZE, WINDOW_SAMPLES), dtype=np.float32)
            for slot, start in enumerate(batch_starts):
                window = waveform[start : start + WINDOW_SAMPLES]
                batch[slot, : len(window)] = window
            sample_features.extend(encoder.encode(batch)[: len(batch_starts)])
            processed_windows += len(batch_starts)
        matrix = np.asarray(sample_features, dtype=np.float32)
        if matrix.shape != (len(starts), 384) or not np.isfinite(matrix).all():
            raise ValueError(
                f"invalid EfficientAT feature sequence: recording {row['recording_id']}"
            )
        all_features.append(matrix)
        offsets.append(offsets[-1] + len(matrix))
        if index % 16 == 0 or index == len(rows):
            print(
                json.dumps(
                    {
                        "modality": "audio",
                        "records": index,
                        "total": len(rows),
                        "windows": processed_windows,
                    }
                ),
                flush=True,
            )

    return {
        "features": np.concatenate(all_features),
        "offsets": np.asarray(offsets, dtype=np.int32),
        "encoder": {
            "id": "EfficientAT-DyMN-0.4-AudioSet",
            "source_commit": commit,
            "checkpoint_sha256": sha256_file(checkpoint),
            "parameters": encoder.parameter_count,
            "device": str(encoder.device),
            "torch": encoder.torch.__version__,
            "torchaudio": torchaudio.__version__,
            "sample_rate_hz": AUDIO_SAMPLE_RATE,
            "window_seconds": 2,
            "stride_seconds": 1,
            "feature_shape": [384],
            "feature_windows": processed_windows,
            "peak_allocated_vram_bytes": encoder.peak_allocated_vram_bytes,
            "frozen": True,
            "python": sys.version,
            "numpy": np.__version__,
            "av": __import__("av").__version__,
        },
    }


def _decode_video_frames(path: Path, av) -> np.ndarray:
    with av.open(str(path)) as container:
        streams = [stream for stream in container.streams if stream.type == "video"]
        if not streams:
            raise ValueError(f"video stream missing: {path}")
        stream = streams[0]
        total_frames = int(stream.frames or 0)
        if total_frames > 0:
            targets = np.asarray(
                [round(i * (total_frames - 1) / (VIDEO_FRAMES - 1)) for i in range(VIDEO_FRAMES)],
                dtype=np.int64,
            )
            wanted = set(targets.tolist())
            selected = {}
            for frame_index, frame in enumerate(container.decode(stream)):
                if frame_index in wanted:
                    selected[frame_index] = frame.to_ndarray(format="rgb24")
                if frame_index >= int(targets[-1]):
                    break
            result = [selected[int(index)] for index in targets if int(index) in selected]
        else:
            selected = []
            if stream.duration is not None and stream.time_base is not None:
                duration = float(stream.duration * stream.time_base)
            elif container.duration is not None:
                duration = float(container.duration) / 1_000_000
            else:
                raise ValueError(f"video duration unavailable: {path}")
            targets = np.linspace(0, duration, VIDEO_FRAMES)
            target_index = 0
            for frame in container.decode(stream):
                if frame.pts is None or frame.time_base is None:
                    raise ValueError(f"video frame has no timestamp: {path}")
                timestamp = float(frame.pts * frame.time_base)
                while target_index < VIDEO_FRAMES and timestamp >= targets[target_index]:
                    selected.append(frame.to_ndarray(format="rgb24"))
                    target_index += 1
                if target_index == VIDEO_FRAMES:
                    break
            result = selected
    if len(result) != VIDEO_FRAMES:
        raise ValueError(f"decoded {len(result)}/{VIDEO_FRAMES} uniform frames: {path}")
    return np.stack(result)


def _run_video(rows: list[dict[str, object]], media_root: Path, args) -> dict[str, object]:
    checkpoint = args.video_checkpoint.resolve(strict=True)
    source = args.videoprism_source.resolve(strict=True)
    if sha256_file(checkpoint) != VIDEO_CHECKPOINT_SHA256:
        raise ValueError("VideoPrism checkpoint hash mismatch")
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True
    ).strip()
    if commit != VIDEO_SOURCE_COMMIT or dirty:
        raise ValueError("VideoPrism source checkout differs from the pinned pristine revision")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import av
    import jax
    import jax.numpy as jnp
    import tensorflow as tf

    if jax.default_backend() != "cpu":
        raise ValueError("VideoPrism video extraction must use the audited CPU backend")
    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    sys.path.insert(0, str(source))
    from videoprism import models as vp

    model = vp.get_model("videoprism_public_v1_base")
    variables = vp.load_pretrained_weights(
        "videoprism_public_v1_base", checkpoint_path=str(checkpoint)
    )
    parameters = sum(int(leaf.size) for leaf in jax.tree_util.tree_leaves(variables))
    if parameters != 114_365_184:
        raise ValueError("VideoPrism parameter count differs from the pinned model")
    encode = jax.jit(lambda batch: model.apply(variables, batch, train=False))
    sequences = []
    inference_seconds = []
    for index, row in enumerate(rows, start=1):
        frames = _decode_video_frames(media_root / str(row["video_path"]), av)
        height, width = frames.shape[1:3]
        scale = max(288 / height, 288 / width)
        resized = tf.image.resize(
            frames, (round(height * scale), round(width * scale)), method="bilinear"
        )
        prepared = (
            np.asarray(tf.image.resize_with_crop_or_pad(resized, 288, 288), dtype=np.float32)
            / 255.0
        )
        started = time.perf_counter()
        outputs = encode(jnp.asarray(prepared[None, ...]))
        jax.tree_util.tree_map(
            lambda value: (
                value.block_until_ready() if hasattr(value, "block_until_ready") else value
            ),
            outputs,
        )
        inference_seconds.append(time.perf_counter() - started)
        tokens = outputs[0] if isinstance(outputs, tuple) else outputs
        tokens = np.asarray(tokens)
        if tokens.shape != (1, VIDEO_FRAMES * 256, 768) or not np.isfinite(tokens).all():
            raise ValueError(f"invalid VideoPrism tokens for recording {row['recording_id']}")
        sequences.append(tokens.reshape(VIDEO_FRAMES, 256, 768).mean(axis=1).astype(np.float32))
        if index % 16 == 0 or index == len(rows):
            print(
                json.dumps({"modality": "video", "records": index, "total": len(rows)}),
                flush=True,
            )
    return {
        "features": np.stack(sequences),
        "encoder": {
            "id": "google/videoprism-base-f16r288",
            "revision": "be719a406d563b66f0ac969e7c94bab8e997c81a",
            "source_commit": commit,
            "checkpoint_sha256": sha256_file(checkpoint),
            "parameters": parameters,
            "device": str(jax.devices()[0]),
            "frame_count": VIDEO_FRAMES,
            "frame_size": 288,
            "pooling": "mean over the 256 spatial tokens independently per frame",
            "feature_shape": [VIDEO_FRAMES, 768],
            "mean_forward_seconds": float(np.mean(inference_seconds)),
            "p95_forward_seconds": float(np.percentile(inference_seconds, 95)),
            "frozen": True,
            "python": sys.version,
            "numpy": np.__version__,
            "jax": jax.__version__,
            "tensorflow": tf.__version__,
            "av": av.__version__,
        },
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen features: {output}")
    media_root = args.media_root.resolve(strict=True)
    rows, acquisition_report = _read_manifest(media_root)
    _verify_media(media_root, rows)
    started = time.perf_counter()
    result = (
        _run_audio(rows, media_root, args)
        if args.modality == "audio"
        else _run_video(rows, media_root, args)
    )
    output.mkdir(parents=True)
    features_path = output / f"{args.modality}-features.npz"
    labels = np.asarray([str(row["target"]) for row in rows])
    splits = np.asarray([str(row["split"]) for row in rows])
    record_ids = np.asarray([int(row["recording_id"]) for row in rows], dtype=np.int16)
    payload = {
        "recording_ids": record_ids,
        "splits": splits,
        "labels": labels,
        "features": result["features"],
    }
    if args.modality == "audio":
        payload["offsets"] = result["offsets"]
    np.savez_compressed(features_path, **payload)
    report = {
        "status": f"PASS_FROZEN_{args.modality.upper()}_FEATURES",
        "modality": args.modality,
        "acquisition_report_sha256": sha256_file(media_root / "acquisition-report.json"),
        "media_index_sha256": sha256_file(media_root / "train-validation.jsonl"),
        "feature_cache": features_path.name,
        "feature_cache_sha256": sha256_file(features_path),
        "feature_cache_bytes": features_path.stat().st_size,
        "recording_count": len(rows),
        "split_counts": {split: int(np.sum(splits == split)) for split in ("train", "validation")},
        "sealed_audit_accessed": False,
        "upstream_encoder": result["encoder"],
        "wall_seconds": time.perf_counter() - started,
        "acquisition_status": acquisition_report["status"],
    }
    (output / "feature-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modality", choices=("audio", "video"), required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--efficientat-source", type=Path)
    parser.add_argument("--audio-checkpoint", type=Path)
    parser.add_argument("--dependency-root", type=Path)
    parser.add_argument("--videoprism-source", type=Path)
    parser.add_argument("--video-checkpoint", type=Path)
    args = parser.parse_args()
    required = (
        ("efficientat_source", "audio_checkpoint", "dependency_root")
        if args.modality == "audio"
        else ("videoprism_source", "video_checkpoint")
    )
    if any(getattr(args, name) is None for name in required):
        required_flags = ", ".join("--" + name.replace("_", "-") for name in required)
        parser.error(f"{args.modality} mode requires: {required_flags}")
    print(json.dumps(run(args), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
