"""Extract one pinned frozen encoder feature bank for causal HoloAssist windows."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from scripts.audit_holoassist_metadata import sha256_file
from scripts.hotgpu_preflight import validate_gpu_preflight
from scripts.pack_holoassist_feature_cache import ordered_id_hash, read_jsonl

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/pretrained_reuse/holoassist_temporal_readout_v1.json"
VIDEO_SOURCE_COMMIT = "c6eba0ece3dd49a24ffc44582966997a4a59f1d8"
VIDEO_CHECKPOINT_SHA256 = "61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29"
HOLOASSIST_SELECTION_SHA256 = "577adbafe6be869de3aff3b814acf91db7be582a42e0a7a83e7ee773dde0e8b7"
WHISPER_SOURCE_WEIGHTS_SHA256 = "7ebd0e69e78190ffe1438491fa05cc1f5c1aa3a4c4db3bc1723adbb551ea2395"
WHISPER_ENCODER_SHA256 = "d7f4b75d10f8ec4af79a32cc6dd20160f0c669c67bd104c1f1ab3f9d5d36a2a8"
WHISPER_PROCESSOR_SHA256 = "3e4dc53c73e7e3954fc4435034f5027733c61d7c123b38bbd1461598fd8503ef"
MINILM_WEIGHTS_SHA256 = "cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433"
MINILM_PROCESSOR_SHA256 = "1c9cc30099243da1f4b287a315c03814279467a6316929f0d16e7f9940b52b0f"


def tensorflow_distribution_version() -> str:
    """Resolve TensorFlow's installed distribution name for pinned CPU builds."""
    for distribution in ("tensorflow-cpu", "tensorflow"):
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            pass
    raise importlib.metadata.PackageNotFoundError("tensorflow-cpu or tensorflow")


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


def _hash_bundle(root: Path, names: list[str]) -> str:
    files = {name: sha256_file(root / name) for name in names}
    payload = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _selected_rows(windows_dir: Path) -> tuple[list[dict], dict[str, list[dict]], dict]:
    rows = {
        split: read_jsonl(windows_dir / f"{split}.jsonl")
        for split in ("train", "validation")
    }
    receipt = json.loads((windows_dir / "receipt.json").read_text(encoding="utf-8"))
    for split in ("train", "validation"):
        receipt_key = "train_jsonl_sha256" if split == "train" else "validation_jsonl_sha256"
        if receipt.get(receipt_key) != sha256_file(windows_dir / f"{split}.jsonl"):
            raise ValueError(f"{split} window corpus hash differs from its receipt")
    ordered = rows["train"] + rows["validation"]
    ids = [str(row["id"]) for row in ordered]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("window IDs must be nonempty and unique across both splits")
    return ordered, rows, receipt


def _require_frozen_selection(av_report: dict) -> None:
    if av_report.get("selection_sha256") != HOLOASSIST_SELECTION_SHA256:
        raise ValueError("AV validation report is not for the frozen HoloAssist selection")


def _verify_provenance(
    *,
    events_dir: Path,
    windows_dir: Path,
    av_report_path: Path,
    rows_by_split: dict[str, list[dict]],
    av_report: dict,
) -> dict:
    _require_frozen_selection(av_report)
    event_receipt = json.loads((events_dir / "receipt.json").read_text(encoding="utf-8"))
    window_receipt = json.loads((windows_dir / "receipt.json").read_text(encoding="utf-8"))
    if event_receipt.get("av_validation_report_sha256") != sha256_file(av_report_path):
        raise ValueError("event corpus was not built from this AV validation report")
    for split in ("train", "validation"):
        event_hash = sha256_file(events_dir / f"{split}.jsonl")
        if window_receipt.get("source_event_receipts", {}).get(split) != event_hash:
            raise ValueError(f"{split} windows do not match their source event corpus")
    validated = {
        (str(row["split"]), str(row["session_id"])): str(row["media_sha256"])
        for row in av_report.get("media", [])
    }
    expected = {
        (split, str(row["session_id"])): str(row["media_sha256"])
        for split, rows in rows_by_split.items()
        for row in rows
    }
    if len(validated) != len(av_report.get("media", [])):
        raise ValueError("AV validation report contains duplicate media rows")
    if any(validated.get(key) != digest for key, digest in expected.items()):
        raise ValueError("window media identity is missing or differs from AV-validated media")
    return {
        "events_receipt_sha256": sha256_file(events_dir / "receipt.json"),
        "windows_receipt_sha256": sha256_file(windows_dir / "receipt.json"),
        "av_validation_report_sha256": sha256_file(av_report_path),
        "selection_sha256": av_report["selection_sha256"],
        "media_manifest_sha256": av_report["media_manifest_sha256"],
        "window_rows": len(expected),
    }


def _media_paths(rows: list[dict], media_root: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for row in rows:
        relative = Path(str(row["media_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("media paths must remain relative to the selected-media root")
        path = (media_root / relative).resolve(strict=True)
        try:
            path.relative_to(media_root.resolve(strict=True))
        except ValueError as error:
            raise ValueError("media path escapes selected-media root") from error
        expected = str(row["media_sha256"])
        previous = paths.setdefault(expected, path)
        if previous != path:
            raise ValueError("one media hash maps to multiple selected paths")
    for digest, path in paths.items():
        if sha256_file(path) != digest:
            raise ValueError(f"selected media changed after AV validation: {path}")
    return paths


def decode_video_window(path: Path, start: float, end: float, frame_count: int = 8) -> np.ndarray:
    import av

    if (
        frame_count != 8
        or not math.isfinite(start + end)
        or start < 0
        or end <= start
        or end - start > 30.0
    ):
        raise ValueError("the frozen VideoPrism path requires an 8-frame positive causal window")
    with av.open(str(path)) as container:
        streams = [stream for stream in container.streams if stream.type == "video"]
        if len(streams) != 1:
            raise ValueError(f"expected exactly one video stream: {path}")
        stream = streams[0]
        time_base = float(stream.time_base)
        container.seek(max(0, int(start / time_base)), stream=stream, backward=True)
        target_times = np.linspace(start, end, num=frame_count, endpoint=True)
        selected: list[np.ndarray] = []
        previous = None
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            stamp = float(frame.pts * frame.time_base)
            if stamp < start:
                continue
            if stamp > end:
                if previous is not None:
                    previous_pixels = previous.to_ndarray(format="rgb24")
                    selected.extend([previous_pixels] * (frame_count - len(selected)))
                break
            crossed_targets = 0
            while (
                len(selected) + crossed_targets < frame_count
                and stamp >= target_times[len(selected) + crossed_targets]
            ):
                crossed_targets += 1
            if crossed_targets:
                pixels = frame.to_ndarray(format="rgb24")
                selected.extend([pixels] * crossed_targets)
            previous = frame
            if len(selected) == frame_count:
                break
        if len(selected) < frame_count:
            if previous is None:
                raise ValueError(f"no video frame decoded inside window: {path} {start}:{end}")
            previous_pixels = previous.to_ndarray(format="rgb24")
            selected.extend([previous_pixels] * (frame_count - len(selected)))
        frames = np.stack(selected)
    return frames


def decode_audio_window(path: Path, start: float, end: float) -> tuple[np.ndarray, int]:
    import av

    rate = 16_000
    if (
        not math.isfinite(start + end)
        or start < 0
        or end <= start
        or end - start > 30.0
    ):
        raise ValueError("audio window must be finite, positive, and at most 30 seconds")
    desired_samples = round((end - start) * rate)
    pcm = np.zeros(desired_samples, dtype=np.float32)
    with av.open(str(path)) as container:
        streams = [stream for stream in container.streams if stream.type == "audio"]
        if len(streams) != 1 or streams[0].rate != 48_000 or streams[0].channels != 1:
            raise ValueError(f"expected validated mono 48 kHz audio: {path}")
        stream = streams[0]
        container.seek(max(0, int(max(0.0, start - 1.0) * av.time_base)), backward=True)
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=rate)

        def retain(frame: Any) -> None:
            if frame.pts is None:
                raise ValueError(f"resampled audio lacks timestamps: {path}")
            frame_start = float(frame.pts * frame.time_base)
            values = frame.to_ndarray().reshape(-1).astype(np.float32, copy=False)
            output_start = round((frame_start - start) * rate)
            left = max(0, output_start)
            right = min(desired_samples, output_start + len(values))
            if right > left:
                pcm[left:right] = values[left - output_start : right - output_start]

        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            frame_start = float(frame.pts * frame.time_base)
            for converted in resampler.resample(frame):
                retain(converted)
            if frame_start > end + 1.0:
                break
        for converted in resampler.resample(None):
            retain(converted)
    if not np.isfinite(pcm).all():
        raise ValueError(f"non-finite resampled audio: {path}")
    return pcm, desired_samples


def _write_bank(path: Path, ids: list[str], features: np.ndarray, metadata: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite feature bank: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("xb") as output:
        np.savez_compressed(
            output,
            sample_id=np.asarray(ids, dtype=str),
            features=np.asarray(features, dtype=np.float32),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
    temporary.replace(path)


def _load_progress(
    path: Path,
    *,
    ids: list[str],
    width: int,
    fingerprint: dict[str, Any],
    resume: bool,
) -> tuple[np.ndarray, np.ndarray, bool]:
    if path.exists():
        if not resume:
            raise FileExistsError(f"feature progress exists; inspect and pass --resume: {path}")
        with np.load(path, allow_pickle=False) as archive:
            saved_ids = list(map(str, archive["sample_id"].tolist()))
            features = archive["features"].copy()
            completed = archive["completed"].copy()
            saved_fingerprint = json.loads(str(archive["fingerprint_json"].item()))
        if saved_ids != ids or saved_fingerprint != fingerprint:
            raise ValueError("feature progress belongs to different IDs, models, or provenance")
        if (
            features.shape != (len(ids), width)
            or features.dtype != np.float32
            or completed.shape != (len(ids),)
            or completed.dtype != np.bool_
            or not np.isfinite(features[completed]).all()
        ):
            raise ValueError("feature progress arrays are invalid")
        return features, completed, True
    if resume:
        raise FileNotFoundError(f"--resume requested but feature progress is missing: {path}")
    return (
        np.zeros((len(ids), width), dtype=np.float32),
        np.zeros(len(ids), dtype=np.bool_),
        False,
    )


def _save_progress(
    path: Path,
    *,
    ids: list[str],
    features: np.ndarray,
    completed: np.ndarray,
    fingerprint: dict[str, Any],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(
            f"stale feature progress temporary file needs inspection: {temporary}"
        )
    with temporary.open("xb") as output:
        np.savez_compressed(
            output,
            sample_id=np.asarray(ids, dtype=str),
            features=features,
            completed=completed,
            fingerprint_json=np.asarray(json.dumps(fingerprint, sort_keys=True)),
        )
    temporary.replace(path)


def _validate_whisper_runtime(model_dir: Path) -> Path:
    model_dir = model_dir.resolve(strict=True)
    runtime_receipt = json.loads((model_dir / "result.json").read_text(encoding="utf-8"))
    if (
        runtime_receipt.get("status") != "PASS"
        or runtime_receipt.get("reference_revision")
        != "openai/whisper-tiny@169d4a4341b33bc18d8881c4b69c2e104e1cc0af"
        or runtime_receipt.get("reference_files", {}).get("model.safetensors")
        != WHISPER_SOURCE_WEIGHTS_SHA256
        or runtime_receipt.get("encoder_sha256") != WHISPER_ENCODER_SHA256
        or runtime_receipt.get("processor_sha256") != WHISPER_PROCESSOR_SHA256
    ):
        raise ValueError("Whisper runtime receipt does not bind to pinned source weights")
    if sha256_file(model_dir / "encoder.safetensors") != WHISPER_ENCODER_SHA256:
        raise ValueError("extracted Whisper encoder file hash differs from pinned runtime")
    if sha256_file(model_dir / "preprocessor_config.json") != WHISPER_PROCESSOR_SHA256:
        raise ValueError("Whisper processor config hash differs from pinned revision")
    return model_dir


def _extract_video(args, config, rows, media_root, features, completed, persist):
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TF_NUM_INTRAOP_THREADS"] = "4"
    os.environ["TF_NUM_INTEROP_THREADS"] = "2"
    import jax
    import jax.numpy as jnp
    import psutil
    import tensorflow as tf

    source = args.upstream_source.resolve(strict=True)
    commit = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != VIDEO_SOURCE_COMMIT:
        raise ValueError(f"unexpected VideoPrism source commit: {commit}")
    source_dirty = subprocess.run(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if source_dirty.strip():
        raise ValueError("tracked VideoPrism source files differ from the pinned commit")
    checkpoint = args.checkpoint.resolve(strict=True)
    if sha256_file(checkpoint) != VIDEO_CHECKPOINT_SHA256:
        raise ValueError("VideoPrism checkpoint hash differs from frozen config")
    import sys

    sys.path.insert(0, str(source))
    from videoprism import models as vp

    model = vp.get_model("videoprism_public_v1_base")
    variables = vp.load_pretrained_weights(
        "videoprism_public_v1_base", checkpoint_path=str(checkpoint)
    )
    parameter_tree = variables.get("params", variables)
    parameter_count = int(
        sum(np.prod(value.shape) for value in jax.tree_util.tree_leaves(parameter_tree))
    )
    if parameter_count != int(config["encoders"]["video"]["parameters"]):
        raise ValueError("loaded VideoPrism parameter count differs from frozen inventory")
    encode = jax.jit(lambda video: model.apply(variables, video, train=False))
    started = time.perf_counter()
    for index, row in enumerate(rows):
        if completed[index]:
            continue
        frames = decode_video_window(
            media_root / row["media_path"],
            float(row["window_start_seconds"]),
            float(row["window_end_seconds"]),
        )
        height, width = frames.shape[1:3]
        scale = max(288 / height, 288 / width)
        resized = tf.image.resize(
            frames, (round(height * scale), round(width * scale)), method="bilinear"
        )
        prepared = tf.image.resize_with_crop_or_pad(resized, 288, 288)
        video = jnp.asarray(np.asarray(prepared, dtype=np.float32)[None] / 255.0)
        result = encode(video)
        jax.tree_util.tree_map(
            lambda value: value.block_until_ready()
            if hasattr(value, "block_until_ready")
            else value,
            result,
        )
        tokens = result[0] if isinstance(result, tuple) else result
        pooled = np.asarray(tokens).mean(axis=1).squeeze(0).astype(np.float32)
        if pooled.shape != (768,) or not np.isfinite(pooled).all():
            raise ValueError(f"invalid VideoPrism embedding for {row['id']}")
        features[index] = pooled
        completed[index] = True
        if index == 0 or (index + 1) % 64 == 0:
            persist()
    persist()
    return features, {
        "model_id": config["encoders"]["video"]["model_id"],
        "revision": config["encoders"]["video"]["revision"],
        "weights_sha256": VIDEO_CHECKPOINT_SHA256,
        "loaded_weights_sha256": VIDEO_CHECKPOINT_SHA256,
        "preprocessing": config["encoders"]["video"]["preprocessing"],
        "upstream_source_commit": commit,
        "parameters": parameter_count,
        "loaded_variable_elements": int(
            sum(np.prod(value.shape) for value in jax.tree_util.tree_leaves(variables))
        ),
        "window_frames": 8,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "jax": jax.__version__,
        "tensorflow": tf.__version__,
        "device": [str(device) for device in jax.devices()],
        "elapsed_seconds": time.perf_counter() - started,
        "peak_rss_bytes": getattr(psutil.Process().memory_info(), "peak_wset", None),
    }


def _extract_audio(args, config, rows, media_root, features, completed, persist):
    import torch
    from safetensors.torch import load_file
    from transformers import WhisperConfig, WhisperFeatureExtractor
    from transformers.models.whisper.modeling_whisper import WhisperEncoder

    model_dir = _validate_whisper_runtime(args.model_dir)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    speech_config = WhisperConfig.from_pretrained(model_dir, local_files_only=True)
    speech_config._attn_implementation = "eager"
    encoder = WhisperEncoder(speech_config)
    encoder.load_state_dict(load_file(model_dir / "encoder.safetensors"), strict=True)
    encoder.eval().requires_grad_(False).to(device)
    processor = WhisperFeatureExtractor.from_pretrained(model_dir, local_files_only=True)
    if processor.sampling_rate != 16_000 or processor.n_samples != 480_000:
        raise ValueError("unexpected pinned Whisper processor profile")
    if sum(parameter.numel() for parameter in encoder.parameters()) != 8_208_384:
        raise ValueError("unexpected Whisper encoder parameter count")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for index, row in enumerate(rows):
        if completed[index]:
            continue
        pcm, valid_samples = decode_audio_window(
            media_root / row["media_path"],
            float(row["window_start_seconds"]),
            float(row["window_end_seconds"]),
        )
        padded = np.pad(pcm, (0, max(0, 480_000 - len(pcm))))[:480_000]
        inputs = processor(
            padded, sampling_rate=16_000, return_tensors="pt"
        ).input_features.to(device)
        with torch.inference_mode():
            tokens = encoder(inputs).last_hidden_state[0]
        valid_tokens = min(tokens.shape[0], math.ceil(valid_samples / 320))
        pooled = tokens[:valid_tokens].mean(dim=0).float().cpu().numpy()
        if pooled.shape != (384,) or not np.isfinite(pooled).all():
            raise ValueError(f"invalid Whisper embedding for {row['id']}")
        features[index] = pooled
        completed[index] = True
        if index == 0 or (index + 1) % 64 == 0:
            persist()
    persist()
    return features, {
        "model_id": config["encoders"]["audio"]["model_id"],
        "revision": config["encoders"]["audio"]["revision"],
        "weights_sha256": WHISPER_SOURCE_WEIGHTS_SHA256,
        "loaded_weights_sha256": WHISPER_ENCODER_SHA256,
        "processor_sha256": WHISPER_PROCESSOR_SHA256,
        "preprocessing": config["encoders"]["audio"]["preprocessing"],
        "parameters": sum(parameter.numel() for parameter in encoder.parameters()),
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "peak_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
        "peak_cuda_reserved_bytes": (
            torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }


def _extract_text(args, config, rows, features, completed, persist):
    import torch
    from transformers import AutoModel, AutoTokenizer

    model_dir = args.model_dir.resolve(strict=True)
    if sha256_file(model_dir / "model.safetensors") != MINILM_WEIGHTS_SHA256:
        raise ValueError("MiniLM weight file differs from pinned revision")
    tokenizer_files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.txt",
        "modules.json",
        "sentence_bert_config.json",
    ]
    if _hash_bundle(model_dir, tokenizer_files) != MINILM_PROCESSOR_SHA256:
        raise ValueError("MiniLM tokenizer/config bundle differs from pinned revision")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    encoder = AutoModel.from_pretrained(
        model_dir, local_files_only=True, dtype=torch.float32
    ).eval().requires_grad_(False)
    if sum(parameter.numel() for parameter in encoder.parameters()) != 17_389_824:
        raise ValueError("unexpected MiniLM parameter count")
    cache: dict[str, np.ndarray] = {}
    started = time.perf_counter()
    for index, row in enumerate(rows):
        if completed[index]:
            continue
        question = str(row["question"])
        if question in cache:
            continue
        encoded = tokenizer(
            question, padding=False, truncation=False, return_tensors="pt"
        )
        if encoded["input_ids"].shape[1] > 512:
            raise ValueError(f"task question exceeds 512 tokens: {row['id']}")
        with torch.inference_mode():
            hidden = encoder(**encoded).last_hidden_state[0]
            mask = encoded["attention_mask"][0].bool()
            pooled = hidden[mask].mean(dim=0).float().numpy()
        if pooled.shape != (384,) or not np.isfinite(pooled).all():
            raise ValueError(f"invalid MiniLM embedding for {row['id']}")
        cache[question] = pooled
    for index, row in enumerate(rows):
        if completed[index]:
            continue
        features[index] = cache[str(row["question"])]
        completed[index] = True
        if index == 0 or (index + 1) % 64 == 0:
            persist()
    persist()
    return features, {
        "model_id": config["encoders"]["text"]["model_id"],
        "revision": config["encoders"]["text"]["revision"],
        "weights_sha256": MINILM_WEIGHTS_SHA256,
        "loaded_weights_sha256": MINILM_WEIGHTS_SHA256,
        "processor_sha256": MINILM_PROCESSOR_SHA256,
        "preprocessing": config["encoders"]["text"]["preprocessing"],
        "parameters": sum(parameter.numel() for parameter in encoder.parameters()),
        "device": "cpu",
        "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "unique_questions": len(cache),
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modality", choices=("video", "audio", "text"), required=True)
    parser.add_argument("--windows-dir", type=Path, required=True)
    parser.add_argument("--events-dir", type=Path, required=True)
    parser.add_argument("--av-validation-report", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--upstream-source", type=Path)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output-bank", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.device == "cuda" and args.modality != "audio":
        raise ValueError("only the frozen Whisper audio encoder supports CUDA extraction")
    if args.modality == "video" and (
        args.device != "cpu" or args.checkpoint is None or args.upstream_source is None
    ):
        raise ValueError("VideoPrism extraction requires CPU JAX, checkpoint and source root")
    if args.modality == "audio":
        if args.model_dir is None:
            raise ValueError("Whisper encoder model directory is required")
        args.model_dir = _validate_whisper_runtime(args.model_dir)
    elif args.modality == "text":
        if args.device != "cpu" or args.model_dir is None:
            raise ValueError("MiniLM extraction requires a local model directory and CPU")
        args.model_dir = args.model_dir.resolve(strict=True)
    output_bank = args.output_bank.resolve()
    if output_bank.exists():
        raise FileExistsError(f"refusing to overwrite feature bank: {output_bank}")
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    if dirty.strip():
        raise ValueError("commit the feature extraction pipeline before extracting task features")
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows, rows_by_split, window_receipt = _selected_rows(args.windows_dir.resolve(strict=True))
    av_report = json.loads(args.av_validation_report.read_text(encoding="utf-8"))
    if av_report.get("status") != "PASS_HOLOASSIST_TRAINVAL_AV_MEDIA":
        raise ValueError("passing full train/validation AV validation report required")
    provenance = _verify_provenance(
        events_dir=args.events_dir.resolve(strict=True),
        windows_dir=args.windows_dir.resolve(strict=True),
        av_report_path=args.av_validation_report.resolve(strict=True),
        rows_by_split=rows_by_split,
        av_report=av_report,
    )
    media_root = args.media_root.resolve(strict=True)
    _media_paths(rows, media_root)
    model_spec = config["encoders"][args.modality]
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    script_hash = sha256_file(Path(__file__))
    software_versions = {
        "python": sys.version.split()[0],
        "av": importlib.metadata.version("av"),
    }
    if args.modality == "video":
        software_versions.update(
            jax=importlib.metadata.version("jax"),
            tensorflow=tensorflow_distribution_version(),
        )
    else:
        software_versions.update(
            torch=importlib.metadata.version("torch"),
            transformers=importlib.metadata.version("transformers"),
        )
    fingerprint = {
        "modality": args.modality,
        "device": args.device,
        "config_sha256": sha256_file(args.config),
        "model_spec": model_spec,
        "ordered_sample_ids_sha256": ordered_id_hash([str(row["id"]) for row in rows]),
        "source_window_hashes": {
            split: sha256_file(args.windows_dir / f"{split}.jsonl")
            for split in ("train", "validation")
        },
        "av_data_provenance": provenance,
        "extractor_source_commit": source_commit,
        "extractor_script_sha256": script_hash,
        "software_versions": software_versions,
    }
    width = {"video": 768, "audio": 384, "text": 384}[args.modality]
    progress_path = output_bank.with_suffix(".progress.npz")
    args.output_bank.parent.mkdir(parents=True, exist_ok=True)
    features, completed, resumed = _load_progress(
        progress_path,
        ids=[str(row["id"]) for row in rows],
        width=width,
        fingerprint=fingerprint,
        resume=args.resume,
    )
    gpu_slot = require_gpu_slot() if args.device == "cuda" else None

    def persist_progress() -> None:
        _save_progress(
            progress_path,
            ids=[str(row["id"]) for row in rows],
            features=features,
            completed=completed,
            fingerprint=fingerprint,
        )

    if args.modality == "video":
        features, model_metadata = _extract_video(
            args, config, rows, media_root, features, completed, persist_progress
        )
    elif args.modality == "audio":
        features, model_metadata = _extract_audio(
            args, config, rows, media_root, features, completed, persist_progress
        )
    else:
        features, model_metadata = _extract_text(
            args, config, rows, features, completed, persist_progress
        )
    if not completed.all():
        raise ValueError("feature extraction ended with unfinished sample IDs")
    ids = [str(row["id"]) for row in rows]
    if features.shape != (len(rows), {"video": 768, "audio": 384, "text": 384}[args.modality]):
        raise ValueError("feature bank shape does not match selected sample IDs")
    metadata = {
        **model_metadata,
        "modality": args.modality,
        "source_window_receipt_sha256": sha256_file(args.windows_dir / "receipt.json"),
        "source_windows_sha256": {
            split: sha256_file(args.windows_dir / f"{split}.jsonl")
            for split in ("train", "validation")
        },
        "sample_count": len(ids),
        "ordered_sample_ids_sha256": ordered_id_hash(ids),
        "unique_session_count": len({str(row["session_id"]) for row in rows}),
        "unique_media_count": len({str(row["media_sha256"]) for row in rows}),
        "preprocessing_config_sha256": sha256_file(args.config),
        "extractor_source_commit": source_commit,
        "extractor_script_sha256": sha256_file(Path(__file__)),
        "resumed_from_progress": resumed,
        "pyav": importlib.metadata.version("av"),
        "av_data_provenance": provenance,
        "test_data_loaded": False,
        "hotgpu_reservation": gpu_slot,
    }
    if model_metadata.get("model_id") != model_spec["model_id"]:
        raise ValueError("loaded model identity differs from config")
    for field in (
        "model_id",
        "revision",
        "weights_sha256",
        "loaded_weights_sha256",
        "processor_sha256",
        "upstream_source_commit",
        "preprocessing",
    ):
        if field in model_spec and model_metadata.get(field) != model_spec[field]:
            raise ValueError(f"{args.modality} loaded model {field} differs from frozen config")
    _write_bank(args.output_bank.resolve(), ids, features, metadata)
    progress_path.unlink(missing_ok=True)
    print(
        json.dumps(
            {
                "status": "complete_frozen_feature_bank",
                "modality": args.modality,
                "output_bank": str(args.output_bank.resolve()),
                "output_sha256": sha256_file(args.output_bank),
                "feature_shape": list(features.shape),
                "metadata": metadata,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
