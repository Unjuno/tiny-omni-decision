"""Extract deterministic frozen Whisper-Tiny features for the fresh M1 split."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
import wave
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
from transformers import WhisperFeatureExtractor, WhisperModel

from tiny_omni_decision.dataset import sha256_file

SOURCE = "google/speech_commands@a751309c0fd613e8a5d30d77900f30e8b42bc2da"
AUDIO_REVISION = "169d4a4341b33bc18d8881c4b69c2e104e1cc0af"
AUDIO_MODEL_SHA256 = "7ebd0e69e78190ffe1438491fa05cc1f5c1aa3a4c4db3bc1723adbb551ea2395"
LABELS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _rows(path: Path, split: str) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows or any(row.get("split") != split or row.get("source") != SOURCE for row in rows):
        raise ValueError(f"invalid or empty {split} manifest")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError(f"duplicate {split} IDs")
    counts = Counter(row["target"] for row in rows)
    expected = 150 if split == "train" else 100
    if set(counts) != set(LABELS) or any(count != expected for count in counts.values()):
        raise ValueError(f"unexpected {split} class counts: {dict(counts)}")
    return rows


def _read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as stream:
        if (
            stream.getnchannels() != 1
            or stream.getsampwidth() != 2
            or stream.getframerate() != 16000
        ):
            raise ValueError(f"expected mono PCM16 16kHz WAV: {path}")
        samples = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").copy()
    if samples.size == 0:
        raise ValueError(f"empty audio file: {path}")
    return samples.astype(np.float32) / 32768.0


def _validate_media(sample_dir: Path, rows: list[dict[str, Any]]) -> None:
    for row in rows:
        path = (sample_dir / row["media_path"]).resolve()
        if not path.is_relative_to(sample_dir) or not path.is_file():
            raise ValueError(f"missing or escaped WAV: {path}")
        if sha256_file(path) != row["media_sha256"]:
            raise ValueError(f"WAV hash mismatch: {row['id']}")


def _extract(
    rows: list[dict[str, Any]],
    sample_dir: Path,
    encoder: torch.nn.Module,
    processor: WhisperFeatureExtractor,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, list[str], list[int], float]:
    features: list[torch.Tensor] = []
    ids: list[str] = []
    targets: list[int] = []
    started = time.perf_counter()
    encoder.eval()
    for start in range(0, len(rows), batch_size):
        current = rows[start : start + batch_size]
        waveforms = [_read_wav(sample_dir / row["media_path"]) for row in current]
        encoded = processor(
            waveforms,
            sampling_rate=16000,
            padding="max_length",
            max_length=processor.n_samples,
            truncation=True,
            return_tensors="pt",
        )
        input_features = encoded.input_features.to(device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = encoder(input_features).last_hidden_state
            pooled_rows = []
            for index, waveform in enumerate(waveforms):
                valid_tokens = min(hidden.shape[1], max(1, math.ceil(len(waveform) / 320)))
                pooled_rows.append(hidden[index, :valid_tokens].float().mean(dim=0))
            pooled = torch.stack(pooled_rows)
        if pooled.shape != (len(current), 384) or not torch.isfinite(pooled).all():
            raise ValueError("frozen Whisper encoder produced invalid pooled features")
        features.append(pooled.cpu())
        ids.extend(str(row["id"]) for row in current)
        targets.extend(LABELS.index(str(row["target"])) for row in current)
    return torch.cat(features), ids, targets, time.perf_counter() - started


def run(sample_dir: Path, model_dir: Path, output_dir: Path, batch_size: int) -> dict[str, Any]:
    sample_dir, model_dir, output_dir = (
        sample_dir.resolve(),
        model_dir.resolve(),
        output_dir.resolve(),
    )
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse output path: {output_dir}")
    train_rows = _rows(sample_dir / "train.jsonl", "train")
    validation_rows = _rows(sample_dir / "validation.jsonl", "validation")
    _validate_media(sample_dir, train_rows)
    _validate_media(sample_dir, validation_rows)
    train_groups = {row["speaker_group_sha256"] for row in train_rows}
    validation_groups = {row["speaker_group_sha256"] for row in validation_rows}
    train_media = {row["media_sha256"] for row in train_rows}
    validation_media = {row["media_sha256"] for row in validation_rows}
    if train_groups & validation_groups or train_media & validation_media:
        raise ValueError("train/validation speaker or audio-content overlap")

    model_hash = sha256_file(model_dir / "model.safetensors")
    if model_hash != AUDIO_MODEL_SHA256:
        raise ValueError(f"Whisper checkpoint hash mismatch: {model_hash}")
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    if config.get("_name_or_path") != "openai/whisper-tiny" or config.get("d_model") != 384:
        raise ValueError("unexpected Whisper model config")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("feature extraction requires the authorized local CUDA/BF16 device")
    device = torch.device("cuda")
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    if free_bytes < 1 * 1024**3:
        raise RuntimeError(f"insufficient free GPU memory before feature extraction: {free_bytes}")

    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats(device)
    load_started = time.perf_counter()
    model = (
        WhisperModel.from_pretrained(model_dir, local_files_only=True, torch_dtype=torch.float32)
        .eval()
        .to(device)
    )
    model.requires_grad_(False)
    encoder = model.encoder
    processor = WhisperFeatureExtractor.from_pretrained(model_dir, local_files_only=True)
    load_seconds = time.perf_counter() - load_started
    torch.cuda.synchronize(device)

    train_features, train_ids, train_targets, train_seconds = _extract(
        train_rows, sample_dir, encoder, processor, device, batch_size
    )
    validation_features, validation_ids, validation_targets, validation_seconds = _extract(
        validation_rows, sample_dir, encoder, processor, device, batch_size
    )
    torch.cuda.synchronize(device)
    if train_ids != [row["id"] for row in train_rows] or validation_ids != [
        row["id"] for row in validation_rows
    ]:
        raise ValueError("feature extraction changed frozen sample ordering")

    output_dir.mkdir(parents=True)
    cache_path = output_dir / "audio-features.pt"
    torch.save(
        {
            "source_revision": SOURCE,
            "encoder_revision": AUDIO_REVISION,
            "train_ids": train_ids,
            "validation_ids": validation_ids,
            "train_targets": train_targets,
            "validation_targets": validation_targets,
            "train_features": train_features,
            "validation_features": validation_features,
        },
        cache_path,
    )
    manifest_hashes = {
        split: sha256_file(sample_dir / f"{split}.jsonl") for split in ("train", "validation")
    }
    report = {
        "schema_version": 1,
        "status": "FRESH_M1_FROZEN_AUDIO_FEATURES_COMPLETE",
        "source": SOURCE,
        "source_train_manifest_sha256": manifest_hashes["train"],
        "source_validation_manifest_sha256": manifest_hashes["validation"],
        "source_train_ids_sha256": _sha256_text("\n".join(train_ids)),
        "source_validation_ids_sha256": _sha256_text("\n".join(validation_ids)),
        "encoder": {
            "repository_id": "openai/whisper-tiny",
            "revision": AUDIO_REVISION,
            "model_safetensors_sha256": model_hash,
            "feature_extractor_config_sha256": sha256_file(model_dir / "preprocessor_config.json"),
            "feature_width": 384,
            "pooling": (
                "mean over valid encoder tokens; valid token count ceil(waveform_samples / 320)"
            ),
            "forward_autocast": "CUDA bfloat16, frozen encoder; pooled result stored float32",
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "trainable_parameters": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
        },
        "train_examples": len(train_rows),
        "validation_examples": len(validation_rows),
        "train_unique_speakers": len(train_groups),
        "validation_unique_speakers": len(validation_groups),
        "train_features_shape": list(train_features.shape),
        "validation_features_shape": list(validation_features.shape),
        "training_ids_order_preserved": True,
        "validation_ids_order_preserved": True,
        "wall_seconds": {
            "model_and_processor_load": load_seconds,
            "train_feature_extraction": train_seconds,
            "validation_feature_extraction": validation_seconds,
        },
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device),
        "hardware": {
            "gpu": torch.cuda.get_device_name(device),
            "cuda_runtime": torch.version.cuda,
            "total_cuda_memory_bytes": total_bytes,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
            "batch_size": batch_size,
        },
        "feature_cache_sha256": sha256_file(cache_path),
        "feature_cache_bytes": cache_path.stat().st_size,
        "feature_extraction_only": True,
        "decision_evaluation_run": False,
        "training_run": False,
        "sealed_audit_loaded": False,
        "paid_compute": False,
    }
    (output_dir / "feature-report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    print(
        json.dumps(run(args.sample_dir, args.model_dir, args.output_dir, args.batch_size), indent=2)
    )


if __name__ == "__main__":
    main()
