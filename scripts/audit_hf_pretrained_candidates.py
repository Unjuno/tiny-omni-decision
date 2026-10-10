"""Read-only local-load audit for pinned text and audio Hugging Face models."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import wave
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_wav_mono(path: Path) -> tuple[Any, int]:
    import numpy as np

    with wave.open(str(path), "rb") as audio:
        channels = audio.getnchannels()
        sample_width = audio.getsampwidth()
        sample_rate = audio.getframerate()
        frames = audio.readframes(audio.getnframes())
    if sample_width != 2:
        raise ValueError(f"smoke WAV must use 16-bit PCM; got {sample_width * 8}-bit")
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    if samples.size == 0 or not np.isfinite(samples).all():
        raise ValueError("WAV contains no finite samples")
    return samples, sample_rate


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-model", type=Path, required=True)
    parser.add_argument("--text-revision", required=True)
    parser.add_argument("--audio-model", type=Path, required=True)
    parser.add_argument("--audio-revision", required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def _model_summary(model: Any) -> dict[str, Any]:
    parameters = tuple(model.named_parameters())
    return {
        "parameter_count": sum(parameter.numel() for _, parameter in parameters),
        "parameter_tensors": len(parameters),
        "dtypes": sorted({str(parameter.dtype) for _, parameter in parameters}),
    }


def _sync(device: Any, torch: Any) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    args = _parse_args()
    text_dir = args.text_model.resolve()
    audio_dir = args.audio_model.resolve()
    audio_path = args.audio.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite audit report: {output_path}")
    required_files = (
        text_dir / "config.json",
        text_dir / "model.safetensors",
        audio_dir / "config.json",
        audio_dir / "model.safetensors",
        audio_path,
    )
    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(path)

    import torch
    from transformers import (
        AutoFeatureExtractor,
        AutoModel,
        AutoModelForAudioClassification,
        AutoTokenizer,
    )

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    torch.set_num_threads(4)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()

    stage_started = time.perf_counter()
    text_tokenizer = AutoTokenizer.from_pretrained(
        text_dir, local_files_only=True, trust_remote_code=False
    )
    text_tokenizer_load_seconds = time.perf_counter() - stage_started
    stage_started = time.perf_counter()
    text_model = (
        AutoModel.from_pretrained(text_dir, local_files_only=True, trust_remote_code=False)
        .eval()
        .to(device)
    )
    _sync(device, torch)
    text_model_load_seconds = time.perf_counter() - stage_started
    text_examples = ["赤い物体は左にあります。", "The red object is on the left."]
    text_inputs = text_tokenizer(text_examples, return_tensors="pt", padding=True, truncation=True)
    text_inputs = {key: value.to(device) for key, value in text_inputs.items()}
    _sync(device, torch)
    stage_started = time.perf_counter()
    with torch.inference_mode():
        text_output = text_model(**text_inputs).last_hidden_state
    _sync(device, torch)
    text_forward_seconds = time.perf_counter() - stage_started
    if not torch.isfinite(text_output).all():
        raise ValueError("text encoder output contains NaN or Inf")
    text_mask = text_inputs["attention_mask"].unsqueeze(-1).to(text_output.dtype)
    text_embeddings = (text_output * text_mask).sum(dim=1) / text_mask.sum(dim=1).clamp_min(1)
    text_summary = _model_summary(text_model)
    text_summary.update(
        {
            "repo_revision": args.text_revision,
            "tokenizer_load_seconds": text_tokenizer_load_seconds,
            "model_load_seconds": text_model_load_seconds,
            "forward_seconds": text_forward_seconds,
            "weights_bytes": (text_dir / "model.safetensors").stat().st_size,
            "weights_sha256": sha256_file(text_dir / "model.safetensors"),
            "input_languages_smoked": ["Japanese", "English"],
            "token_shape": list(text_output.shape),
            "pooled_shape": list(text_embeddings.shape),
            "finite": bool(torch.isfinite(text_embeddings).all().item()),
        }
    )

    waveform, sample_rate = _read_wav_mono(audio_path)
    stage_started = time.perf_counter()
    audio_processor = AutoFeatureExtractor.from_pretrained(
        audio_dir, local_files_only=True, trust_remote_code=False
    )
    audio_processor_load_seconds = time.perf_counter() - stage_started
    stage_started = time.perf_counter()
    audio_model = (
        AutoModelForAudioClassification.from_pretrained(
            audio_dir, local_files_only=True, trust_remote_code=False
        )
        .eval()
        .to(device)
    )
    _sync(device, torch)
    audio_model_load_seconds = time.perf_counter() - stage_started
    audio_inputs = audio_processor(
        waveform,
        sampling_rate=sample_rate,
        return_tensors="pt",
    )
    audio_inputs = {key: value.to(device) for key, value in audio_inputs.items()}
    _sync(device, torch)
    stage_started = time.perf_counter()
    with torch.inference_mode():
        audio_output = audio_model(**audio_inputs)
    _sync(device, torch)
    audio_forward_seconds = time.perf_counter() - stage_started
    audio_logits = audio_output.logits
    if not torch.isfinite(audio_logits).all():
        raise ValueError("audio encoder output contains NaN or Inf")
    audio_summary = _model_summary(audio_model)
    audio_summary.update(
        {
            "repo_revision": args.audio_revision,
            "processor_load_seconds": audio_processor_load_seconds,
            "model_load_seconds": audio_model_load_seconds,
            "forward_seconds": audio_forward_seconds,
            "weights_bytes": (audio_dir / "model.safetensors").stat().st_size,
            "weights_sha256": sha256_file(audio_dir / "model.safetensors"),
            "audio_sample_rate": sample_rate,
            "audio_samples": int(waveform.size),
            "input_tensor_shapes": {key: list(value.shape) for key, value in audio_inputs.items()},
            "logits_shape": list(audio_logits.shape),
            "finite": bool(torch.isfinite(audio_logits).all().item()),
        }
    )

    report = {
        "schema_version": 1,
        "status": "complete_read_only_text_audio_load_and_forward_audit",
        "device": str(device),
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "text": text_summary,
        "audio": audio_summary,
        "elapsed_seconds": time.perf_counter() - started,
        "cuda_peak_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
        "cuda_peak_reserved_bytes": (
            torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
        ),
        "note": "Smoke inputs validate loading and tensor execution, not task quality.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
