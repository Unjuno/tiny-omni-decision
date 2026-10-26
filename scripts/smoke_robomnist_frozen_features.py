"""CPU-only input/runtime diagnostic on the first fixed RoboMNIST training question.

Stream three CRC-checked media members into memory; retain only frozen temporal
features and an execution receipt. No fitting, option scoring, or audit access.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import io
import json
import platform
import subprocess
import sys
import time
import wave
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import requests

from scripts.audit_robomnist_repetition_payload import extract_member
from scripts.extract_perception_temporal_match_efficientat import (
    SOURCE_COMMIT,
    WEIGHTS_SHA256,
    sha256_file,
    state_sha,
)
from scripts.robomnist_container_features import stereo_tiles

ARTIFACTS = Path("C:/CodexArtifacts")
R1 = ARTIFACTS / "pretrained-reuse-r1"
R2 = ARTIFACTS / "pretrained-reuse-r2"
DIRECTORY = R2 / "robomnist-directory-audit-20261026-attempt-04.json"
VERSION = R2 / "robomnist-source-audit-01/article-version-1.json"
TRAIN = R2 / "robomnist-av-correspondence-seed17-attempt-05/train.json"
MODEL_DIR = R1 / "tinyclip-39m19m-audit-01/model"
AUDIO_SOURCE = R1 / "efficientat-audit-v1/source"
AUDIO_WEIGHTS = R1 / "efficientat-audit-v1/dymn04_as.pt"
DEPENDENCIES = R1 / "python-deps"
DIRECTORY_SHA = "538d526bfa049d4cef29568bcbe06b32a96a303a2c229d10cf0420641932d0cb"
VERSION_SHA = "18ef8b20c82056747c58572d24f262b2d12c9284e4ffd83c245ae64eca2ab120"
TRAIN_SHA = "0aa00362dc933c85926cd11bde8774ce96d1d350468e5927f226f04628b96049"
IMAGE_SHA = "112bdfb07ebc8994e9afba7a2a314e278493a675033595cf48e54cf8a17ae587"
CENTERS = np.linspace(1.0, 13.8, 8).tolist()


def select_runtime_refs(rows):
    if not rows:
        raise ValueError("nonempty fixed training manifest required")
    refs = rows[0]["source_refs"]
    if len(refs["audio_candidates"]) != 2:
        raise ValueError("two audio candidates required")
    return {"video": refs["video"], "audio_candidates": refs["audio_candidates"]}


def nearest_frame_indices(timestamps, centers):
    pts = np.asarray(timestamps, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    if len(pts) < 2 or not np.isfinite(pts).all() or np.any(np.diff(pts) <= 0):
        raise ValueError("video PTS must be finite and strictly increasing")
    if not np.isfinite(centers).all() or np.any(centers < pts[0]) or np.any(centers > pts[-1]):
        raise ValueError("video timestamp coverage does not contain sample centers")
    return [int(np.abs(pts - center).argmin()) for center in centers]


def audio_windows(waveform, sample_rate, centers, duration):
    if waveform.ndim != 1 or sample_rate <= 0 or duration <= 0 or not np.isfinite(waveform).all():
        raise ValueError("finite mono waveform and positive rate/duration required")
    size = round(duration * sample_rate)
    windows = []
    for center in centers:
        start = round((center - duration / 2) * sample_rate)
        if start < 0 or start + size > len(waveform):
            raise ValueError("audio timestamp coverage does not contain complete windows")
        windows.append(waveform[start : start + size])
    return np.stack(windows)


def decode_video(content):
    import av

    # Keep only the nearest frames while decoding in timestamp order, rather than
    # retaining 450 full-resolution frames in RAM.
    best_errors = [float("inf")] * len(CENTERS)
    frames = [None] * len(CENTERS)
    pts = []
    with av.open(io.BytesIO(content)) as container:
        if len(container.streams) != 1 or len(container.streams.video) != 1:
            raise ValueError("expected one video-only stream")
        for frame in container.decode(video=0):
            if frame.pts is None:
                raise ValueError("video frame has no PTS")
            timestamp = float(frame.pts * frame.time_base)
            pts.append(timestamp)
            for index, center in enumerate(CENTERS):
                error = abs(timestamp - center)
                if error < best_errors[index]:
                    best_errors[index] = error
                    frames[index] = frame.to_image()
    indices = nearest_frame_indices(pts, CENTERS)
    return frames, {
        "decoded_frames": len(pts),
        "first_pts_s": pts[0],
        "last_pts_s": pts[-1],
        "selected_pts_s": [pts[i] for i in indices],
    }


def decode_audio(content):
    with wave.open(io.BytesIO(content), "rb") as audio:
        if audio.getnchannels() != 1 or audio.getsampwidth() != 2:
            raise ValueError("expected mono PCM16 WAV")
        rate = audio.getframerate()
        samples = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2")
        waveform = samples.astype(np.float32) / 32768.0
    return waveform, rate


def checked_json(path, expected):
    if sha256_file(path) != expected:
        raise ValueError(f"pinned input hash mismatch: {path}")
    return json.loads(path.read_bytes())


def variation(features, time_axis=-2):
    norms = np.linalg.norm(features, axis=-1)
    if not np.isfinite(features).all() or np.any(norms <= 0):
        raise ValueError("features must be finite and nonzero")
    unit = np.moveaxis(features / norms[..., None], time_axis, -2)
    return {
        "shape": list(features.shape),
        "finite": True,
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
        "mean_adjacent_cosine_distance": float(
            (1 - np.sum(unit[..., 1:, :] * unit[..., :-1, :], axis=-1)).mean()
        ),
    }


def run(output):
    if output.exists():
        raise FileExistsError("choose a fresh smoke output directory")
    audit = checked_json(DIRECTORY, DIRECTORY_SHA)
    version = checked_json(VERSION, VERSION_SHA)
    rows = checked_json(TRAIN, TRAIN_SHA)
    if version["id"] != 28179383 or version["version"] != 1:
        raise ValueError("RoboMNIST v1 required")
    refs = select_runtime_refs(rows)
    for path, expected in (
        (MODEL_DIR / "model.safetensors", IMAGE_SHA),
        (AUDIO_WEIGHTS, WEIGHTS_SHA256),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"pinned weights hash mismatch: {path}")
    commit = subprocess.check_output(
        ["git", "-C", str(AUDIO_SOURCE), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(AUDIO_SOURCE), "status", "--porcelain"], text=True
    ).strip()
    if commit != SOURCE_COMMIT or dirty:
        raise ValueError("EfficientAT source must be pristine at the pinned commit")
    output.mkdir(parents=True)
    started = time.perf_counter()
    receipt = {
        "status": "RUNNING",
        "started_at": datetime.now(UTC).isoformat(),
        "script_sha256": sha256_file(Path(__file__)),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "device": "cpu",
        "dtype": "float32",
        "seed": 17,
        "threads": 4,
        "train_manifest_sha256": TRAIN_SHA,
        "directory_audit_sha256": DIRECTORY_SHA,
        "version_metadata_sha256": VERSION_SHA,
        "selection": "first training question",
        "sample_id": rows[0]["id"],
        "source_refs": refs,
        "time_centers_s": CENTERS,
        "audio_window_s": 2.0,
        "audio_resampling_hz": 32000,
        "video_policy": "nearest actual PTS per center",
        "image_frontend": "both full stereo lenses; letterbox224/BICUBIC/mean-fill; no crop",
        "implementation_sha256": {
            name: sha256_file(Path(__file__).parent / name)
            for name in [
                "audit_robomnist_repetition_payload.py",
                "extract_perception_temporal_match_efficientat.py",
                "robomnist_container_features.py",
            ]
        },
        "tinyclip_checkpoint_sha256": IMAGE_SHA,
        "tinyclip_revision": "07a4b0bc751cb64fecd2b661c048c1dd98d69444",
        "efficientat_checkpoint_sha256": WEIGHTS_SHA256,
        "efficientat_source_commit": commit,
        "training": False,
        "quality_evaluated": False,
        "sealed_audit_opened": False,
        "raw_media_saved": False,
        "limitations": [
            "one training question; no generalization or calibration result",
            "shared time centers do not establish audio/video clock synchronization",
            "filename-derived repetitions do not establish independent sessions",
        ],
    }
    run_path = output / "run.json"

    def publish():
        temporary = output / "run.json.tmp"
        temporary.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        temporary.replace(run_path)

    publish()
    try:
        remaining, expanded = [32 * 1024 * 1024], [0]
        payloads, member_receipts = [], []
        transfer_started = time.perf_counter()
        with requests.Session() as session:
            for ref in [refs["video"], *refs["audio_candidates"]]:
                entry = next(item for item in version["files"] if item["name"] == ref["archive"])
                directory = next(
                    item for item in audit["archives"] if item["archive"] == ref["archive"]
                )
                member = next(
                    item for item in directory["member_inventory"] if item["name"] == ref["member"]
                )
                print("fetch", ref["member"], flush=True)
                payload = extract_member(
                    session, entry["download_url"], entry["size"], member, remaining, expanded
                )
                member_receipts.append(
                    {
                        "member": ref["member"],
                        "crc32_verified": True,
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "uncompressed_bytes": len(payload),
                    }
                )
                payloads.append(payload)
        receipt.update(
            {
                "members": member_receipts,
                "network_bytes": 32 * 1024 * 1024 - remaining[0],
                "network_seconds": time.perf_counter() - transfer_started,
            }
        )
        publish()
        print("payload verified; CPU model load", flush=True)
        sys.path.insert(0, str(DEPENDENCIES))
        sys.path.insert(0, str(AUDIO_SOURCE))
        import torch
        import torchaudio
        from transformers import AutoModelForZeroShotImageClassification, AutoProcessor

        with contextlib.redirect_stdout(io.StringIO()):
            from models.dymn.model import get_model
            from models.preprocess import AugmentMelSTFT
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.manual_seed(17)
        processor = AutoProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
        image_model = (
            AutoModelForZeroShotImageClassification.from_pretrained(
                MODEL_DIR, local_files_only=True
            )
            .eval()
            .requires_grad_(False)
        )
        with contextlib.redirect_stdout(io.StringIO()):
            audio_model = get_model(width_mult=0.4, pretrained_name=None, T_max=1.0)
        audio_model.load_state_dict(
            torch.load(AUDIO_WEIGHTS, map_location="cpu", weights_only=True)
        )
        audio_model.eval().requires_grad_(False)
        mel = AugmentMelSTFT(freqm=0, timem=0).eval().requires_grad_(False)
        image_parameters = sum(p.numel() for p in image_model.parameters())
        audio_parameters = sum(p.numel() for p in audio_model.parameters())
        if (image_parameters, audio_parameters) != (83_101_185, 1_961_239):
            raise ValueError("model parameter count differs from R1 audit")
        before = {"image": state_sha(image_model), "audio": state_sha(audio_model)}
        decode_started = time.perf_counter()
        frames, video_timing = decode_video(payloads[0])
        decode_seconds = time.perf_counter() - decode_started
        inference_started = time.perf_counter()
        with torch.inference_mode():
            tiles = [
                tile
                for frame in frames
                for tile in stereo_tiles(np.asarray(frame), processor.image_processor.image_mean)
            ]
            images = processor(
                images=tiles, do_resize=False, do_center_crop=False, return_tensors="pt"
            )
            video_features = (
                image_model.visual_projection(
                    image_model.vision_model(**images, return_dict=True).pooler_output
                )
                .float()
                .numpy()
                .reshape(8, 2, 512)
            )
            audio_features, audio_timing = [], []
            for payload in payloads[1:]:
                waveform, sample_rate = decode_audio(payload)
                windows = audio_windows(waveform, sample_rate, CENTERS, 2.0)
                resampled = torchaudio.functional.resample(
                    torch.from_numpy(windows), sample_rate, 32000
                )
                _, embedding = audio_model(mel(resampled).unsqueeze(1))
                audio_features.append(embedding.float().numpy())
                audio_timing.append(
                    {
                        "source_rate_hz": sample_rate,
                        "source_samples": len(waveform),
                        "duration_s": len(waveform) / sample_rate,
                        "window_rms": np.sqrt((windows**2).mean(axis=1)).tolist(),
                    }
                )
        audio_features = np.stack(audio_features)
        if video_features.shape != (8, 2, 512) or audio_features.shape != (2, 8, 384):
            raise ValueError("unexpected temporal feature shape")
        inference_seconds = time.perf_counter() - inference_started
        after = {"image": state_sha(image_model), "audio": state_sha(audio_model)}
        if before != after:
            raise ValueError("frozen model state changed")
        feature_path = output / "features.npz"
        with feature_path.open("xb") as target:
            np.savez_compressed(
                target, video=video_features, audio=audio_features, centers_s=np.asarray(CENTERS)
            )
        receipt.update(
            {
                "status": "PASS_ONE_TRAIN_QUESTION_FROZEN_CPU_RUNTIME",
                "video_timing": video_timing,
                "audio_timing": audio_timing,
                "video_features": variation(video_features, time_axis=0),
                "audio_features": variation(audio_features),
                "parameters": {
                    "image_model_including_unused_text_tower": image_parameters,
                    "audio": audio_parameters,
                    "trainable": 0,
                },
                "state_before": before,
                "state_after": after,
                "feature_sha256": sha256_file(feature_path),
                "feature_bytes": feature_path.stat().st_size,
                "video_decode_seconds": decode_seconds,
                "inference_and_audio_preprocessing_seconds": inference_seconds,
                "attention_backend": image_model.config._attn_implementation,
                "cuda_initialized": torch.cuda.is_initialized(),
                "versions": {
                    name: importlib.metadata.version(name)
                    for name in ["torch", "torchaudio", "transformers", "av", "numpy"]
                },
                "python": platform.python_version(),
                "finished_at": datetime.now(UTC).isoformat(),
                "wall_seconds": time.perf_counter() - started,
            }
        )
    except Exception as error:
        receipt.update(
            {
                "status": "FAILED_RUNTIME",
                "error": repr(error),
                "wall_seconds": time.perf_counter() - started,
            }
        )
        publish()
        raise
    publish()
    print(receipt["status"], "feature_sha256", receipt["feature_sha256"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)


if __name__ == "__main__":
    main()
