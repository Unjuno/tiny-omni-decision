"""CPU-only in-memory smoke for one frozen RoboMNIST AV match pair."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import platform
import subprocess
import sys
import time
import urllib.error
import urllib.request
import wave
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tiny_omni_decision.robomnist_media import compressed_member_range  # noqa: E402

SPLIT_SHA256 = "3f4575ba41d944310e1de83cbdc68385390581723262245b97dc0d2c71db660f"
VIDEO_SHA256 = "61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29"
VIDEO_SOURCE_COMMIT = "c6eba0ece3dd49a24ffc44582966997a4a59f1d8"
VIDEO_PARAMETERS = 114_365_184
AUDIO_SHA256 = "49b0e63577021994257b3de2ac5f1686d71554b93c8149f9d2568ad5760b904e"
AUDIO_SOURCE_COMMIT = "a425fdce92572e602a1d5634799bd9f1f2efa806"
AUDIO_PARAMETERS = 1_961_239
PAIR_ID = "f1dda52cfe72a7bd84dfaececd80ad7a2a15f6bc5e32c50d2e723a0c668ae381"
LOCAL_INPUT_ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2/robomnist-input-smoke-01")
VIDEO_MODEL = Path("C:/CodexArtifacts/pretrained-reuse-r1/videoprism-base-f16r288.npz")
VIDEO_SOURCE = Path("C:/CodexArtifacts/pretrained-reuse-r1/videoprism-base-r1/source")
AUDIO_WEIGHTS = Path("C:/CodexArtifacts/pretrained-reuse-r1/efficientat-audit-v1/dymn04_as.pt")
AUDIO_SOURCE = Path("C:/CodexArtifacts/pretrained-reuse-r1/efficientat-audit-v1/source")
GLOBAL_SITE = Path("C:/Users/junny/AppData/Local/Programs/Python/Python311/Lib/site-packages")
PYTHON_DEPS = Path("C:/CodexArtifacts/pretrained-reuse-r1/python-deps")
FEATURE_FRAMES = 8
FEATURE_SIZE = 288
NETWORK_TIMEOUT_SECONDS = 120


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _request_range(url: str, start: int, length: int, total_size: int) -> bytes:
    if start < 0 or length < 0 or start + length > total_size:
        raise ValueError("requested range is outside the pinned source archive")
    if length == 0:
        return b""
    end = start + length - 1
    request = urllib.request.Request(
        url,
        headers={
            "Range": f"bytes={start}-{end}",
            "Accept-Encoding": "identity",
        },
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=NETWORK_TIMEOUT_SECONDS) as response:
                expected = f"bytes {start}-{end}/{total_size}"
                if response.status != 206 or response.headers.get("Content-Range") != expected:
                    raise ValueError("source did not honor the exact pinned HTTP range")
                data = response.read(length + 1)
            if len(data) != length:
                raise ValueError("source returned an incorrect HTTP range length")
            return data
        except urllib.error.HTTPError as exc:
            if exc.code not in (408, 425, 429, 500, 502, 503, 504) or attempt == 2:
                raise
            retry_after = exc.headers.get("Retry-After", "")
            try:
                delay = min(30, max(1, int(retry_after)))
            except ValueError:
                delay = min(30, 2**attempt)
            time.sleep(delay)
    raise AssertionError("bounded range retry loop fell through")


def _video_frames(payload: bytes) -> tuple[np.ndarray, dict[str, Any]]:
    import av

    selected: dict[int, np.ndarray] = {}
    times: dict[int, float] = {}
    total = None
    with av.open(io.BytesIO(payload)) as container:
        streams = [stream for stream in container.streams if stream.type == "video"]
        if len(streams) != 1:
            raise ValueError("RoboMNIST member must contain exactly one video stream")
        stream = streams[0]
        stream.codec_context.thread_count = 1
        declared = int(stream.frames or 0)
        for index, frame in enumerate(container.decode(stream)):
            if index >= 1024 or frame.pts is None:
                raise ValueError("video decode exceeded its bounded frame/PTS contract")
            rgb = frame.to_ndarray(format="rgb24")
            if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
                raise ValueError("video decoder returned an unsupported RGB frame")
            if declared > 0:
                targets = [
                    round(position * (declared - 1) / (FEATURE_FRAMES - 1))
                    for position in range(FEATURE_FRAMES)
                ]
                if index in targets:
                    selected[index] = rgb
                    times[index] = float(frame.time) if frame.time is not None else float("nan")
            else:
                # The first pass retains only a bounded set; unknown frame counts are rejected
                # after decoding rather than silently changing the sampling schedule.
                if index < FEATURE_FRAMES:
                    selected[index] = rgb
                    times[index] = float(frame.time) if frame.time is not None else float("nan")
                total = index + 1
        count = declared or total
    if count is None or count < FEATURE_FRAMES:
        raise ValueError("video has too few frames for the fixed eight-frame encoder input")
    indices = [
        round(position * (count - 1) / (FEATURE_FRAMES - 1)) for position in range(FEATURE_FRAMES)
    ]
    if declared <= 0:
        raise ValueError("video stream must expose its frame count for fixed sampling")
    if any(index not in selected for index in indices):
        raise ValueError("video decoder missed a fixed uniformly sampled frame")
    frames = np.stack([selected[index] for index in indices])
    pts = [times[index] for index in indices]
    if not np.isfinite(pts).all() or any(b <= a for a, b in zip(pts, pts[1:], strict=False)):
        raise ValueError("sampled video PTS must be finite and strictly increasing")
    return frames, {
        "decoded_frames": count,
        "sample_indices": indices,
        "sampled_pts_seconds": pts,
        "frame_shape": list(frames.shape[1:]),
    }


def _load_video_encoder(jax: Any, checkpoint: Path, source: Path):
    if jax.default_backend() != "cpu" or any(device.platform != "cpu" for device in jax.devices()):
        raise ValueError("VideoPrism must remain on the pinned CPU backend for this smoke")
    sys.path.insert(0, str(source))
    from videoprism import models

    model = models.get_model("videoprism_public_v1_base")
    variables = models.load_pretrained_weights(
        "videoprism_public_v1_base", checkpoint_path=str(checkpoint)
    )
    parameters = sum(int(leaf.size) for leaf in jax.tree_util.tree_leaves(variables))
    if parameters != VIDEO_PARAMETERS:
        raise ValueError(f"VideoPrism parameter count mismatch: {parameters}")
    encode = jax.jit(lambda value: model.apply(variables, value, train=False))
    return model, variables, encode, parameters


def _encode_video(frames: np.ndarray, *, jax: Any, tf: Any, encode: Any):
    from scripts.extract_frozen_videoprism_features import _preprocess
    from tiny_omni_decision.temporal_tokens import videoprism_features_by_layout

    batch = _preprocess(frames, tf)[None, ...]
    if batch.shape != (1, FEATURE_FRAMES, FEATURE_SIZE, FEATURE_SIZE, 3):
        raise ValueError(f"unexpected VideoPrism input shape: {batch.shape}")
    encoded = encode(jax.numpy.asarray(batch))
    jax.tree_util.tree_map(
        lambda value: value.block_until_ready() if hasattr(value, "block_until_ready") else value,
        encoded,
    )
    tokens = encoded[0] if isinstance(encoded, tuple) else encoded
    features = videoprism_features_by_layout(np.asarray(tokens), FEATURE_FRAMES, "per_frame_mean")
    if features.shape != (1, FEATURE_FRAMES, 768) or not np.isfinite(features).all():
        raise ValueError("VideoPrism did not return finite [1,8,768] features")
    return features[0].astype(np.float32, copy=False)


def _audio_windows(payload: bytes):
    import torch
    import torchaudio

    with wave.open(io.BytesIO(payload), "rb") as source:
        channels, width, rate, count = (
            source.getnchannels(),
            source.getsampwidth(),
            source.getframerate(),
            source.getnframes(),
        )
        if (
            source.getcomptype() != "NONE"
            or width not in (2, 3, 4)
            or rate not in (44_100, 48_000)
            or channels not in (1, 2)
            or count < rate * 2
        ):
            raise ValueError("WAV must be bounded PCM16/24/32 mono/stereo at 44.1/48 kHz")
        pcm = source.readframes(count)
    values = _pcm_to_float(pcm, width)
    waveform = values.reshape(-1, channels).mean(axis=1).copy()
    tensor = torch.from_numpy(waveform).unsqueeze(0)
    if rate != 32_000:
        tensor = torchaudio.functional.resample(tensor, orig_freq=rate, new_freq=32_000)
    if tensor.shape[-1] < 64_000:
        raise ValueError("resampled audio is shorter than one two-second window")
    duration = tensor.shape[-1] / 32_000
    centers = np.linspace(1.0, duration - 1.0, FEATURE_FRAMES)
    windows = []
    for center in centers:
        first = round((center - 1.0) * 32_000)
        window = tensor[:, first : first + 64_000]
        if window.shape[-1] != 64_000 or not torch.isfinite(window).all():
            raise ValueError("audio window is incomplete or non-finite")
        windows.append(window.squeeze(0))
    return torch.stack(windows), {
        "channels": channels,
        "sample_width_bytes": width,
        "source_rate_hz": rate,
        "source_frames": count,
        "duration_seconds": duration,
        "window_centers_seconds": centers.tolist(),
        "window_seconds": 2.0,
        "cross_modal_clock_claim": "none; audio windows are pooled independently of video PTS",
    }


def _pcm_to_float(pcm: bytes, width: int) -> np.ndarray:
    if width == 2:
        integers = np.frombuffer(pcm, dtype="<i2").astype(np.int32)
        scale = 2**15
    elif width == 3:
        packed = np.frombuffer(pcm, dtype=np.uint8)
        if packed.size % 3:
            raise ValueError("24-bit PCM byte count must be divisible by three")
        triples = packed.reshape(-1, 3).astype(np.int32)
        integers = triples[:, 0] | (triples[:, 1] << 8) | (triples[:, 2] << 16)
        integers = np.where(integers & 0x800000, integers - 0x1000000, integers)
        scale = 2**23
    elif width == 4:
        integers = np.frombuffer(pcm, dtype="<i4")
        scale = 2**31
    else:
        raise ValueError(f"unsupported PCM sample width: {width}")
    return integers.astype(np.float32) / scale


def _load_audio_encoder(torch: Any, torchaudio: Any, weights: Path, source: Path):
    sys.path.insert(0, str(source))
    from models.dymn.model import get_model
    from models.preprocess import AugmentMelSTFT

    with contextlib.redirect_stdout(io.StringIO()):
        model = get_model(width_mult=0.4, pretrained_name=None, T_max=1.0)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    count = sum(parameter.numel() for parameter in model.parameters())
    if count != AUDIO_PARAMETERS:
        raise ValueError(f"EfficientAT parameter count mismatch: {count}")
    model.eval().requires_grad_(False)
    mel = AugmentMelSTFT(freqm=0, timem=0).eval().requires_grad_(False)
    return model, mel, count


def _encode_audio(windows: Any, model: Any, mel: Any, torch: Any):
    with torch.inference_mode():
        mel_input = mel(windows).unsqueeze(1)
        logits, embedding = model(mel_input)
    values = embedding.detach().float().cpu().numpy()
    if logits.shape != (FEATURE_FRAMES, 527) or values.shape != (FEATURE_FRAMES, 384):
        raise ValueError("EfficientAT output dimensions differ from the pinned smoke contract")
    if not np.isfinite(values).all() or not np.isfinite(logits.detach().cpu().numpy()).all():
        raise ValueError("EfficientAT produced non-finite features or logits")
    return values


def _load_local_assets(manifest: dict[str, Any], result_path: Path):
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if (
        result.get("status") != "PASS_FIXED_NATIVE_INPUT_MECHANICS_ONLY"
        or result.get("training") is not False
        or result.get("gpu_used") is not False
        or result.get("sealed_audit_opened") is not False
        or result.get("quality_measured") is not False
        or result.get("config", {}).get("group", {}).get("group_id") != "Ar1_Vhigh_Ac0_Rp01_UNC062"
    ):
        raise ValueError("existing Rp01 raw input smoke is not a valid read-only source")
    group_id = "Ar1_Vhigh_Ac0_Rp01_UNC062"
    source_record = manifest["groups"][group_id]
    raw_by_member = {row["source_member"]: row for row in result["raw_files"]}
    payloads = {}
    for modality, member_key, filename in (
        ("video", "video_member", "camera-1.mp4"),
        ("audio", "audio_member", "microphone-1.wav"),
    ):
        metadata = source_record[member_key]
        receipt = raw_by_member.get(metadata["name"])
        path = LOCAL_INPUT_ROOT / filename
        if (
            receipt is None
            or not path.is_file()
            or path.stat().st_size != metadata["size"]
            or receipt["file_bytes"] != metadata["size"]
            or receipt["crc32"] != metadata["crc32"]
            or sha256_file(path) != receipt["sha256"]
        ):
            raise ValueError(f"existing source payload integrity failed: {modality}")
        payloads[modality] = path.read_bytes()
    return group_id, payloads, sha256_file(result_path)


def _pin_sources(prior_result_hash: str) -> dict[str, Any]:
    video_commit = subprocess.check_output(
        ["git", "-C", str(VIDEO_SOURCE), "rev-parse", "HEAD"], text=True
    ).strip()
    audio_commit = subprocess.check_output(
        ["git", "-C", str(AUDIO_SOURCE), "rev-parse", "HEAD"], text=True
    ).strip()
    if video_commit != VIDEO_SOURCE_COMMIT or audio_commit != AUDIO_SOURCE_COMMIT:
        raise ValueError("pretrained encoder source commit differs from the reviewed pins")
    if sha256_file(VIDEO_MODEL) != VIDEO_SHA256 or sha256_file(AUDIO_WEIGHTS) != AUDIO_SHA256:
        raise ValueError("pretrained encoder checkpoint hash differs from R1 pins")
    allowed_generated = ("build/", "videoprism.egg-info/", "videoprism/__pycache__/")
    video_status = subprocess.check_output(
        ["git", "-C", str(VIDEO_SOURCE), "status", "--porcelain", "--untracked-files=all"],
        text=True,
    ).splitlines()
    audio_status = subprocess.check_output(
        ["git", "-C", str(AUDIO_SOURCE), "status", "--porcelain", "--untracked-files=all"],
        text=True,
    ).splitlines()
    for source, status in ((VIDEO_SOURCE, video_status), (AUDIO_SOURCE, audio_status)):
        tracked_diff = subprocess.run(
            ["git", "-C", str(source), "diff", "--quiet", "HEAD", "--"], check=False
        )
        if tracked_diff.returncode != 0:
            raise ValueError("pinned pretrained encoder has tracked source modifications")
        unexpected = [
            row[3:].replace("\\", "/")
            for row in status
            if row.startswith("?? ")
            and not row[3:].replace("\\", "/").startswith(allowed_generated)
        ]
        if unexpected:
            raise ValueError(f"pinned encoder has unexpected untracked files: {unexpected}")
    return {
        "video_checkpoint_sha256": VIDEO_SHA256,
        "video_source_commit": video_commit,
        "audio_checkpoint_sha256": AUDIO_SHA256,
        "audio_source_commit": audio_commit,
        "existing_input_result_sha256": prior_result_hash,
        "videoprism_generated_untracked": [
            row[3:] for row in video_status if row.startswith("?? ")
        ],
        "tracked_source_diff": False,
    }


def run(manifest_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    manifest_hash = sha256_file(manifest_path)
    if manifest_hash != SPLIT_SHA256:
        raise ValueError("frozen attempt-03 manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pair = next(unit for unit in manifest["pair_units"]["train"] if unit["pair_unit_id"] == PAIR_ID)
    if pair["group_a"] != "Ar1_Vhigh_Ac0_Rp01_UNC062":
        raise ValueError("smoke pair is not the fixed train pair with the cached Rp01 asset")
    local_id, local_payloads, prior_result_hash = _load_local_assets(
        manifest, LOCAL_INPUT_ROOT / "result.json"
    )
    if local_id != pair["group_a"]:
        raise ValueError("cached media identity differs from the selected train pair")

    output.mkdir(parents=True)
    started_at = datetime.now(UTC).isoformat()
    started = time.perf_counter()
    status = {
        "status": "RUNNING",
        "started_at": started_at,
        "finished_at": None,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "manifest_sha256": manifest_hash,
        "pair_unit_id": PAIR_ID,
        "groups": [pair["group_a"], pair["group_b"]],
        "training": False,
        "validation_metrics_measured": False,
        "quality_evaluated": False,
        "sealed_audit_opened": False,
        "raw_media_saved": False,
        "gpu_used": False,
    }
    (output / "run.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")

    try:
        sys.path.insert(0, str(REPO_ROOT))
        sys.path.insert(0, str(REPO_ROOT / "src"))
        # Load JAX/TensorFlow in their already pinned VideoPrism venv first. Then expose
        # the existing Torch/Torchaudio wheels without changing either environment.
        import jax
        import numpy as vp_numpy
        import tensorflow as tf

        sys.path.append(str(GLOBAL_SITE))
        sys.path.append(str(PYTHON_DEPS))
        import torch
        import torchaudio

        tf.config.set_visible_devices([], "GPU")
        if vp_numpy.__version__ != "1.26.4" or np.__version__ != "1.26.4":
            raise ValueError("mixed runtime did not retain the pinned VideoPrism NumPy")
        if not GLOBAL_SITE.is_dir() or not PYTHON_DEPS.is_dir():
            raise FileNotFoundError("existing pinned Torch/Torchaudio wheels are unavailable")
        if (
            jax.default_backend() != "cpu"
            or torch.cuda.is_initialized()
            or tf.config.get_visible_devices("GPU")
        ):
            raise ValueError("smoke must remain CPU-only")
        tf.config.threading.set_intra_op_parallelism_threads(4)
        tf.config.threading.set_inter_op_parallelism_threads(2)
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)

        source_pins = _pin_sources(prior_result_hash)
        video_model, video_variables, video_encode, video_parameters = _load_video_encoder(
            jax, VIDEO_MODEL, VIDEO_SOURCE
        )
        audio_model, audio_mel, audio_parameters = _load_audio_encoder(
            torch, torchaudio, AUDIO_WEIGHTS, AUDIO_SOURCE
        )
        audio_state_before = hashlib.sha256(
            b"".join(
                name.encode() + tensor.detach().cpu().contiguous().numpy().tobytes()
                for name, tensor in sorted(audio_model.state_dict().items())
            )
        ).hexdigest()

        groups = []
        total_network_bytes = 0
        for group_id in pair["group_a"], pair["group_b"]:
            group = manifest["groups"][group_id]
            payloads = local_payloads if group_id == local_id else {}
            media_records = {}
            for modality, member_key in (("video", "video_member"), ("audio", "audio_member")):
                member = group[member_key]
                if modality in payloads:
                    content = payloads[modality]
                    range_bytes = 0
                else:
                    archive = {
                        "download_url": group["archive_url"],
                        "size": group["archive_size"],
                    }
                    header = _request_range(
                        archive["download_url"], member["offset"], 30, archive["size"]
                    )
                    range_start, tail_length = compressed_member_range(header, member)
                    tail = _request_range(
                        archive["download_url"], range_start, tail_length, archive["size"]
                    )
                    from tiny_omni_decision.robomnist_media import decode_pinned_member

                    content = decode_pinned_member(header, tail, member)
                    range_bytes = 30 + tail_length
                    total_network_bytes += range_bytes
                media_records[modality] = {
                    "member": member["name"],
                    "source_size": len(content),
                    "source_sha256": sha256_bytes(content),
                    "crc32_expected": member["crc32"],
                    "compressed_bytes": member["compressed_size"],
                    "range_bytes_read": range_bytes,
                    "source_origin": "existing verified input smoke"
                    if not range_bytes
                    else "pinned HTTP Range",
                }
                payloads[modality] = content

            frames, video_timing = _video_frames(payloads["video"])
            video_features = _encode_video(frames, jax=jax, tf=tf, encode=video_encode)
            windows, audio_timing = _audio_windows(payloads["audio"])
            audio_features = _encode_audio(windows, audio_model, audio_mel, torch)
            groups.append(
                {
                    "group_id": group_id,
                    "video_features": video_features,
                    "audio_features": audio_features,
                    "media": media_records,
                    "video_timing": video_timing,
                    "audio_timing": audio_timing,
                }
            )
            print(
                json.dumps(
                    {
                        "group_id": group_id,
                        "network_bytes_total": total_network_bytes,
                        "video_shape": list(video_features.shape),
                        "audio_shape": list(audio_features.shape),
                    }
                ),
                flush=True,
            )

        audio_state_after = hashlib.sha256(
            b"".join(
                name.encode() + tensor.detach().cpu().contiguous().numpy().tobytes()
                for name, tensor in sorted(audio_model.state_dict().items())
            )
        ).hexdigest()
        if audio_state_before != audio_state_after:
            raise RuntimeError("frozen EfficientAT weights changed during inference")
        if torch.cuda.is_initialized() or any(device.platform != "cpu" for device in jax.devices()):
            raise RuntimeError("an encoder unexpectedly initialized a GPU backend")
        feature_arrays = {
            "group_ids": np.asarray([row["group_id"] for row in groups]),
            "video_features": np.stack([row["video_features"] for row in groups]),
            "audio_features": np.stack([row["audio_features"] for row in groups]),
        }
        feature_path = output / "features.npz"
        np.savez_compressed(feature_path, **feature_arrays)
        metadata = {
            **status,
            "status": "PASS_TWO_GROUP_FROZEN_FEATURE_SMOKE",
            "finished_at": datetime.now(UTC).isoformat(),
            "run_seconds": time.perf_counter() - started,
            "network_bytes": total_network_bytes,
            "peak_working_set_bytes": __import__(
                "scripts.fetch_robomnist_input_smoke", fromlist=["peak_bytes"]
            ).peak_bytes(),
            "source_pins": source_pins,
            "runtime": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "jax": jax.__version__,
                "tensorflow": tf.__version__,
                "torch": torch.__version__,
                "torchaudio": torchaudio.__version__,
                "video_device": "CPU",
                "audio_device": "CPU",
                "jax_backend": jax.default_backend(),
                "torch_cuda_initialized": torch.cuda.is_initialized(),
            },
            "encoders": {
                "video": {"parameters": video_parameters, "feature_shape": [8, 768]},
                "audio": {
                    "parameters": audio_parameters,
                    "feature_shape": [8, 384],
                    "state_sha256_before": audio_state_before,
                    "state_sha256_after": audio_state_after,
                },
                "trainable_parameters": 0,
                "resident_encoder_parameters": video_parameters + audio_parameters,
            },
            "sampling": {
                "video": "eight uniformly spaced frames over full decoded clip",
                "audio": "eight uniformly spaced 2-second windows pooled independently",
                "audio_video_alignment": "not assumed; only episode association is used",
            },
            "groups_detail": [
                {
                    **{
                        key: value
                        for key, value in row.items()
                        if key not in {"video_features", "audio_features"}
                    },
                    "video_feature_sha256": sha256_bytes(row["video_features"].tobytes()),
                    "audio_feature_sha256": sha256_bytes(row["audio_features"].tobytes()),
                }
                for row in groups
            ],
            "feature_file": str(feature_path),
            "feature_file_sha256": sha256_file(feature_path),
            "feature_bytes": feature_path.stat().st_size,
            "quality_result": None,
            "limitations": [
                "one train pair only; no fitted readout or quality estimate",
                "prior RoboMNIST metadata split was exposed; prior summary records no media, "
                "training or inference",
                "episode-level association only; no audio/video frame-clock alignment claim",
            ],
        }
        (output / "result.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (output / "run.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return metadata
    except BaseException as exc:
        status.update(
            {
                "status": "INCOMPLETE",
                "finished_at": datetime.now(UTC).isoformat(),
                "run_seconds": time.perf_counter() - started,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        (output / "run.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.manifest.resolve(strict=True), args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
