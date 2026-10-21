"""One-shot evaluation of the frozen text-free audio readout on Speech Commands test."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import math
import platform
import subprocess
import time
import wave
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from tiny_omni_decision.dataset import SPEECH_COMMAND_LABELS, sha256_file
from tiny_omni_decision.io import load_structured_file
from tiny_omni_decision.schema import DatasetManifest

REVISION = "a751309c0fd613e8a5d30d77900f30e8b42bc2da"
ENCODER_REVISION = "169d4a4341b33bc18d8881c4b69c2e104e1cc0af"
EXPECTED_PARQUET_SHA256 = "6446af16af8ef28423c57aac7ec0faf091905e2580afcaca5e7733188b6ee85a"
EXPECTED_READOUT_SHA256 = "171f6a9303b2fd9ad63efe2bee6af119549fb4f26ea663e9c1297f51a678e1c3"
EXPECTED_BASELINE_READOUT_SHA256 = (
    "46808a882ab75f8f19a138e4a91d13ceb42cfdc82534c4608c42ee615774d3cb"
)
EXPECTED_CALIBRATION_REPORT_SHA256 = (
    "16e4396e8d1dc9f0a977dee18d8022f5093bbee0bf963359ed3a66c8a59d8d21"
)
EXPECTED_TEMPERATURE = 0.42702477413437295
EXPECTED_ENCODER_WEIGHTS_SHA256 = "7ebd0e69e78190ffe1438491fa05cc1f5c1aa3a4c4db3bc1723adbb551ea2395"
LABELS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _known_label(value: Any, class_names: list[str]) -> str | None:
    if isinstance(value, int) and not isinstance(value, bool):
        if not 0 <= value < len(class_names):
            raise ValueError(f"test label index is outside the pinned class list: {value}")
        value = class_names[value]
    if not isinstance(value, str):
        raise ValueError("test row label must be a string or integer")
    return value if value in LABELS else None


def build_test_records(rows: list[dict[str, Any]], class_names: list[str]) -> list[dict[str, Any]]:
    """Filter only the ten declared command classes and hash identifiers/content."""
    records: list[dict[str, Any]] = []
    for row in rows:
        label = _known_label(row.get("label"), class_names)
        if label is None:
            continue
        audio = row.get("audio")
        if not isinstance(audio, dict) or not isinstance(audio.get("bytes"), bytes):
            raise ValueError("test audio must be available as undecoded bytes")
        payload = audio["bytes"]
        if len(payload) < 12 or payload[:4] != b"RIFF" or payload[8:12] != b"WAVE":
            raise ValueError("test media is not a WAV payload")
        speaker = row.get("speaker_id")
        source_path = row.get("file") or audio.get("path")
        if not isinstance(speaker, str) or not speaker or not isinstance(source_path, str):
            raise ValueError("test row lacks source speaker and recording identifiers")
        records.append(
            {
                "id_sha256": sha256_bytes(source_path.encode()),
                "speaker_group_sha256": sha256_bytes(speaker.encode()),
                "media_sha256": sha256_bytes(payload),
                "target": label,
                "payload": payload,
            }
        )
    if not records:
        raise ValueError("official test split has no recognized ten-keyword examples")
    if len({row["id_sha256"] for row in records}) != len(records):
        raise ValueError("duplicate official test source IDs")
    if len({row["media_sha256"] for row in records}) != len(records):
        raise ValueError("duplicate official test audio content")
    if set(Counter(row["target"] for row in records)) != set(LABELS):
        raise ValueError("official test split does not contain all ten fixed classes")
    return records


def verify_disjoint(records, train_rows, validation_rows, calibration_groups) -> dict[str, int]:
    test_assets = {row["media_sha256"] for row in records}
    test_speakers = {row["speaker_group_sha256"] for row in records}
    known_ids = {row["id"] for row in [*train_rows, *validation_rows]}
    known_assets = {row["media_sha256"] for row in [*train_rows, *validation_rows]}
    known_speakers = {row["speaker_group_sha256"] for row in [*train_rows, *validation_rows]} | set(
        calibration_groups
    )
    if test_assets & known_assets or test_speakers & known_speakers:
        raise ValueError("official test media or speaker identity overlaps fitted/development data")
    # Source IDs are represented differently by source adapters; only compare exact hashes
    # when a source row has a media-independent canonical ID supplied by both sides.
    return {
        "test_train_validation_asset_overlap": len(test_assets & known_assets),
        "test_train_validation_calibration_speaker_overlap": len(test_speakers & known_speakers),
        "test_unique_examples": len(records),
        "known_sample_id_count": len(known_ids),
    }


def _load_rows(path: Path, expected_split: str, sample_dir: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not rows or any(
        row.get("split") != expected_split
        or row.get("source") != f"google/speech_commands@{REVISION}"
        for row in rows
    ):
        raise ValueError(f"invalid frozen {expected_split} sample rows")
    if len({row.get("id") for row in rows}) != len(rows):
        raise ValueError(f"duplicate frozen {expected_split} sample IDs")
    if any(row.get("target") not in LABELS for row in rows):
        raise ValueError(f"invalid target in frozen {expected_split} sample rows")
    for row in rows:
        media = (sample_dir / row["media_path"]).resolve()
        if not media.is_relative_to(sample_dir) or not media.is_file():
            raise ValueError(f"missing or escaped frozen sample media: {media}")
        if sha256_file(media) != row.get("media_sha256"):
            raise ValueError(f"frozen sample media hash mismatch: {media}")
    return rows


def _metrics(logits, targets) -> dict[str, float | int]:
    import torch
    from torch.nn import functional as F

    probabilities = torch.softmax(logits.float(), dim=-1)
    confidence, predictions = probabilities.max(dim=-1)
    correct = predictions.eq(targets)
    one_hot = F.one_hot(targets, num_classes=len(LABELS)).float()
    ece = 0.0
    for bin_index in range(15):
        mask = (confidence >= bin_index / 15) & (
            confidence < (bin_index + 1) / 15 if bin_index < 14 else confidence <= 1
        )
        if mask.any():
            ece += mask.float().mean().item() * abs(
                correct[mask].float().mean().item() - confidence[mask].mean().item()
            )
    return {
        "count": int(len(targets)),
        "accuracy": correct.float().mean().item(),
        "nll": F.cross_entropy(logits.float(), targets).item(),
        "brier": (probabilities - one_hot).square().sum(dim=-1).mean().item(),
        "ece_15_bins": ece,
        "mean_confidence": confidence.mean().item(),
    }


def _speaker_group_split(
    speaker_group_ids: list[str], *, seed: int = 17, calibration_fraction: float = 0.2
) -> tuple[list[int], list[int]]:
    if (
        not speaker_group_ids
        or any(not isinstance(value, str) or not value for value in speaker_group_ids)
        or not math.isfinite(calibration_fraction)
        or not 0 < calibration_fraction < 1
    ):
        raise ValueError("invalid speaker groups or calibration fraction")
    groups = sorted(set(speaker_group_ids))
    if len(groups) < 2:
        raise ValueError("at least two speaker groups are required")
    ranked = sorted(
        groups,
        key=lambda group: hashlib.sha256(f"{seed}:calibration:{group}".encode()).hexdigest(),
    )
    calibration_groups = set(ranked[: math.ceil(len(ranked) * calibration_fraction)])
    calibration = [
        index for index, group in enumerate(speaker_group_ids) if group in calibration_groups
    ]
    fit = [
        index for index, group in enumerate(speaker_group_ids) if group not in calibration_groups
    ]
    return fit, calibration


def _decode_pcm(payload: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(payload), "rb") as stream:
        if (
            stream.getnchannels() != 1
            or stream.getsampwidth() != 2
            or stream.getframerate() != 16000
            or stream.getcomptype() != "NONE"
        ):
            raise ValueError("expected mono 16-bit 16 kHz PCM WAV")
        pcm = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").copy()
    if not len(pcm):
        raise ValueError("empty speech command recording")
    return pcm.astype(np.float32) / 32768.0


def grouped_bootstrap(
    records: list[dict[str, Any]],
    targets: Any,
    logits_by_model: dict[str, Any],
    *,
    repetitions: int = 10_000,
    seed: int = 17,
) -> dict[str, Any]:
    """Paired speaker-group bootstrap CIs for accuracy and proper scores."""
    import torch

    groups = sorted({str(row["speaker_group_sha256"]) for row in records})
    if repetitions < 1 or len(groups) < 2 or len(records) != len(targets):
        raise ValueError("invalid grouped bootstrap inputs")
    group_lookup = {group: index for index, group in enumerate(groups)}
    group_index = np.asarray(
        [group_lookup[str(row["speaker_group_sha256"])] for row in records], dtype=np.int64
    )
    target = targets.detach().cpu().numpy().astype(np.int64, copy=False)
    n_groups = len(groups)
    counts = np.bincount(group_index, minlength=n_groups).astype(np.float64)
    prepared: dict[str, dict[str, np.ndarray]] = {}
    for name, logits in logits_by_model.items():
        probabilities = torch.softmax(logits.detach().float(), dim=-1).cpu().numpy()
        confidence = probabilities.max(axis=1)
        correct = probabilities.argmax(axis=1) == target
        nll = -np.log(np.clip(probabilities[np.arange(len(target)), target], 1e-12, 1.0))
        one_hot = np.eye(probabilities.shape[1], dtype=np.float64)[target]
        brier = np.square(probabilities - one_hot).sum(axis=1)
        bins = np.minimum((confidence * 15).astype(np.int64), 14)
        flat_bins = group_index * 15 + bins
        bin_count = np.bincount(flat_bins, minlength=n_groups * 15).reshape(n_groups, 15)
        bin_correct = np.bincount(
            flat_bins, weights=correct.astype(np.float64), minlength=n_groups * 15
        ).reshape(n_groups, 15)
        bin_confidence = np.bincount(
            flat_bins, weights=confidence, minlength=n_groups * 15
        ).reshape(n_groups, 15)
        prepared[name] = {
            "correct": np.bincount(
                group_index, weights=correct.astype(np.float64), minlength=n_groups
            ),
            "nll": np.bincount(group_index, weights=nll, minlength=n_groups),
            "brier": np.bincount(group_index, weights=brier, minlength=n_groups),
            "bin_count": bin_count,
            "bin_correct": bin_correct,
            "bin_confidence": bin_confidence,
        }
    rng = np.random.default_rng(seed)
    samples: dict[str, dict[str, np.ndarray]] = {
        name: {
            key: np.empty(repetitions, dtype=np.float64)
            for key in ("accuracy", "nll", "brier", "ece_15_bins")
        }
        for name in logits_by_model
    }
    for repetition in range(repetitions):
        selected = rng.integers(0, n_groups, size=n_groups)
        multiplicity = np.bincount(selected, minlength=n_groups).astype(np.float64)
        n = float(multiplicity @ counts)
        for name, values in prepared.items():
            samples[name]["accuracy"][repetition] = float(multiplicity @ values["correct"]) / n
            samples[name]["nll"][repetition] = float(multiplicity @ values["nll"]) / n
            samples[name]["brier"][repetition] = float(multiplicity @ values["brier"]) / n
            bin_correct_sum = multiplicity @ values["bin_correct"]
            bin_confidence_sum = multiplicity @ values["bin_confidence"]
            samples[name]["ece_15_bins"][repetition] = float(
                np.abs(bin_correct_sum - bin_confidence_sum).sum() / n
            )

    def interval(values: np.ndarray) -> list[float]:
        return [float(value) for value in np.quantile(values, [0.025, 0.975])]

    baseline_name = "legacy_baseline"
    if baseline_name not in samples:
        raise ValueError("paired bootstrap requires the fixed legacy baseline")
    return {
        "method": "paired percentile bootstrap resampling whole speaker groups",
        "repetitions": repetitions,
        "seed": seed,
        "independent_speaker_groups": n_groups,
        "low_power_below_200_speakers": n_groups < 200,
        "metric_95_percent_intervals": {
            name: {metric: interval(values) for metric, values in metrics.items()}
            for name, metrics in samples.items()
        },
        "paired_difference_95_percent_intervals_vs_legacy_baseline": {
            name: {
                metric: interval(values - samples[baseline_name][metric])
                for metric, values in metrics.items()
            }
            for name, metrics in samples.items()
            if name != baseline_name
        },
    }


def _check_cuda_gate() -> dict[str, int | str]:
    from scripts.hotgpu_preflight import validate_gpu_preflight

    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    used, total, utilization = (
        int(item.strip()) for item in query.stdout.splitlines()[0].split(",")
    )
    pids_result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    pids = {int(item.strip()) for item in pids_result.stdout.splitlines() if item.strip()}
    return validate_gpu_preflight(
        memory_used_mib=used,
        utilization_pct=utilization,
        total_mib=total,
        active_pids=pids,
        own_pid=__import__("os").getpid(),
        environment=__import__("os").environ,
    )


def run(
    *,
    manifest_path: Path,
    calibration_report: Path,
    checkpoint: Path,
    baseline_checkpoint: Path,
    sample_dir: Path,
    encoder_dir: Path,
    source_dir: Path,
    output_dir: Path,
    feature_batch_size: int = 32,
) -> dict[str, Any]:
    import torch
    from transformers import WhisperFeatureExtractor, WhisperModel

    from tiny_omni_decision.textfree_audio import TextFreeAudioReadout

    paths = [
        p.resolve()
        for p in (
            manifest_path,
            calibration_report,
            checkpoint,
            baseline_checkpoint,
            sample_dir,
            encoder_dir,
            source_dir,
            output_dir,
        )
    ]
    (
        manifest_path,
        calibration_report,
        checkpoint,
        baseline_checkpoint,
        sample_dir,
        encoder_dir,
        source_dir,
        output_dir,
    ) = paths
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse output path: {output_dir}")
    if feature_batch_size < 1:
        raise ValueError("feature batch size must be positive")
    if sha256_file(checkpoint) != EXPECTED_READOUT_SHA256:
        raise ValueError("frozen readout hash does not match the preregistered checkpoint")
    if sha256_file(baseline_checkpoint) != EXPECTED_BASELINE_READOUT_SHA256:
        raise ValueError("legacy baseline hash does not match the preregistered checkpoint")
    if sha256_file(calibration_report) != EXPECTED_CALIBRATION_REPORT_SHA256:
        raise ValueError("calibration report hash does not match the preregistered report")
    calibration = json.loads(calibration_report.read_text(encoding="utf-8"))
    if calibration.get("calibration_temperature") != EXPECTED_TEMPERATURE:
        raise ValueError("frozen calibration temperature changed")
    if calibration.get("weights_sha256") != EXPECTED_READOUT_SHA256:
        raise ValueError("calibration report references a different readout")
    manifest = DatasetManifest.model_validate(load_structured_file(manifest_path))
    if (
        manifest.dataset_id != "google/speech_commands"
        or manifest.revision != REVISION
        or manifest.split != "test"
        or manifest.usage != "evaluation"
        or manifest.license != "CC-BY-4.0"
        or manifest.notes.get("hub_parquet_sha256", {}).get("test-00000-of-00001.parquet")
        != EXPECTED_PARQUET_SHA256
    ):
        raise ValueError("Speech Commands test manifest is not the pinned evaluation source")
    output_dir.mkdir(parents=True)
    lock = {
        "schema_version": 1,
        "status": "evaluation_started_do_not_resume_or_repeat",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "manifest_sha256": sha256_file(manifest_path),
        "calibration_report_sha256": sha256_file(calibration_report),
        "readout_sha256": sha256_file(checkpoint),
        "baseline_readout_sha256": sha256_file(baseline_checkpoint),
        "test_parquet_sha256_expected": EXPECTED_PARQUET_SHA256,
        "training_or_calibration_on_test": False,
        "checkpoint_selection_on_test": False,
    }
    (output_dir / "run-lock.json").write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    source_dir.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download

    parquet_path = Path(
        hf_hub_download(
            repo_id=manifest.dataset_id,
            repo_type="dataset",
            filename="v0.02/test-00000-of-00001.parquet",
            revision=manifest.revision,
            local_dir=source_dir / "hub",
        )
    )
    parquet_sha = sha256_file(parquet_path)
    if parquet_sha != EXPECTED_PARQUET_SHA256:
        raise ValueError(f"official test Parquet hash mismatch: {parquet_sha}")
    from datasets import Audio, load_dataset

    dataset = load_dataset(
        "parquet", data_files={"test": str(parquet_path)}, split="test", streaming=True
    )
    if "audio" not in dataset.features or "label" not in dataset.features:
        raise ValueError("pinned Speech Commands test Parquet schema changed")
    class_names = list(dataset.features["label"].names)
    if class_names != SPEECH_COMMAND_LABELS:
        raise ValueError("pinned Speech Commands numeric class order changed")
    dataset = dataset.cast_column("audio", Audio(decode=False))
    raw_rows = list(dataset)
    if len(raw_rows) != manifest.notes.get("hub_card_split_rows"):
        raise ValueError("official Speech Commands test row count differs from pinned manifest")
    records = build_test_records(raw_rows, class_names)

    train_rows = _load_rows(sample_dir / "train.jsonl", "train", sample_dir)
    validation_rows = _load_rows(sample_dir / "validation.jsonl", "validation", sample_dir)
    all_train_groups = sorted({row["speaker_group_sha256"] for row in train_rows})
    fit_indices, calibration_indices = _speaker_group_split(
        [row["speaker_group_sha256"] for row in train_rows], seed=17, calibration_fraction=0.2
    )
    fit_groups = {train_rows[i]["speaker_group_sha256"] for i in fit_indices}
    calibration_groups = {train_rows[i]["speaker_group_sha256"] for i in calibration_indices}
    validation_groups = {row["speaker_group_sha256"] for row in validation_rows}
    train_assets = {row["media_sha256"] for row in train_rows}
    validation_assets = {row["media_sha256"] for row in validation_rows}
    if (
        len(all_train_groups) == 0
        or fit_groups & calibration_groups
        or {row["speaker_group_sha256"] for row in train_rows} & validation_groups
        or train_assets & validation_assets
        or calibration_groups & validation_groups
    ):
        raise ValueError("invalid frozen train/calibration speaker partition")
    disjointness = verify_disjoint(records, train_rows, validation_rows, calibration_groups)
    if (
        disjointness["test_train_validation_asset_overlap"]
        or disjointness["test_train_validation_calibration_speaker_overlap"]
    ):
        raise ValueError("official test split is not group/content disjoint")

    weights_path = encoder_dir / "model.safetensors"
    if not weights_path.is_file() or sha256_file(weights_path) != EXPECTED_ENCODER_WEIGHTS_SHA256:
        raise ValueError("local Whisper Tiny weights do not match the pinned encoder")
    if not torch.cuda.is_available():
        raise RuntimeError("official audio feature extraction requires the guarded local CUDA path")
    gate = _check_cuda_gate()
    if gate["mode"] != "hotgpu_reserved":
        raise RuntimeError("official test extraction must launch through HotUGPU reservation")
    device = torch.device("cuda")
    torch.set_num_threads(4)
    encoder = (
        WhisperModel.from_pretrained(encoder_dir, local_files_only=True, dtype=torch.float32)
        .encoder.eval()
        .to(device)
    )
    processor = WhisperFeatureExtractor.from_pretrained(encoder_dir, local_files_only=True)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)

    features: list[torch.Tensor] = []
    feature_started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    for start in range(0, len(records), feature_batch_size):
        batch = records[start : start + feature_batch_size]
        waveforms = [_decode_pcm(row["payload"]) for row in batch]
        encoded = processor(
            waveforms,
            sampling_rate=16000,
            padding="max_length",
            max_length=processor.n_samples,
            truncation=True,
            return_tensors="pt",
        )
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = encoder(encoded.input_features.to(device)).last_hidden_state
            pooled = torch.stack(
                [
                    hidden[i, : min(hidden.shape[1], max(1, math.ceil(len(waveform) / 320)))]
                    .float()
                    .mean(0)
                    for i, waveform in enumerate(waveforms)
                ]
            )
        if pooled.shape != (len(batch), 384) or not torch.isfinite(pooled).all():
            raise ValueError("Whisper test features are malformed or non-finite")
        features.append(pooled.cpu())
    torch.cuda.synchronize(device)
    feature_seconds = time.perf_counter() - feature_started
    x_test = torch.cat(features)
    targets = torch.tensor([LABELS.index(row["target"]) for row in records], dtype=torch.long)

    model = TextFreeAudioReadout(384, LABELS).eval()
    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True
    )
    baseline_model = TextFreeAudioReadout(384, LABELS).eval()
    baseline_model.load_state_dict(
        torch.load(baseline_checkpoint, map_location="cpu", weights_only=True), strict=True
    )
    with torch.inference_mode():
        logits = model(x_test)
        baseline_logits = baseline_model(x_test)
    raw_metrics = _metrics(logits, targets)
    baseline_metrics = _metrics(baseline_logits, targets)
    calibrated_logits = logits / EXPECTED_TEMPERATURE
    calibrated_metrics = _metrics(calibrated_logits, targets)
    bootstrap = grouped_bootstrap(
        records,
        targets,
        {
            "legacy_baseline": baseline_logits,
            "uncalibrated": logits,
            "temperature_scaled": calibrated_logits,
        },
    )
    per_class = {
        label: {
            "count": int((targets == class_index).sum()),
            "legacy_baseline": _metrics(
                baseline_logits[targets == class_index], targets[targets == class_index]
            ),
            "uncalibrated": _metrics(
                logits[targets == class_index], targets[targets == class_index]
            ),
            "temperature_scaled": _metrics(
                calibrated_logits[targets == class_index], targets[targets == class_index]
            ),
        }
        for class_index, label in enumerate(LABELS)
    }
    predictions = output_dir / "official-test-predictions.jsonl"
    with predictions.open("x", encoding="utf-8", newline="\n") as stream:
        for row, target, baseline_row_logits, row_logits, scaled_logits in zip(
            records, targets.tolist(), baseline_logits, logits, calibrated_logits, strict=True
        ):
            stream.write(
                json.dumps(
                    {
                        "id_sha256": row["id_sha256"],
                        "speaker_group_sha256": row["speaker_group_sha256"],
                        "media_sha256": row["media_sha256"],
                        "target_class_id": LABELS[target],
                        "legacy_baseline_prediction": LABELS[int(baseline_row_logits.argmax())],
                        "legacy_baseline_probabilities": torch.softmax(
                            baseline_row_logits.float(), -1
                        ).tolist(),
                        "prediction_class_id": LABELS[int(row_logits.argmax())],
                        "probabilities_uncalibrated": torch.softmax(
                            row_logits.float(), -1
                        ).tolist(),
                        "probabilities_temperature_scaled": torch.softmax(
                            scaled_logits.float(), -1
                        ).tolist(),
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
    report = {
        "schema_version": 1,
        "status": "complete_one_shot_official_speech_commands_test_evaluation",
        "claim_scope": "fixed 10-way English Speech Commands keyword task only; not general speech",
        "source": f"google/speech_commands@{REVISION}",
        "manifest_sha256": sha256_file(manifest_path),
        "test_parquet_sha256": parquet_sha,
        "official_test_raw_rows": len(raw_rows),
        "omitted_nonkeyword_rows": len(raw_rows) - len(records),
        "test_examples": len(records),
        "test_id_order_sha256": sha256_bytes("\n".join(r["id_sha256"] for r in records).encode()),
        "test_media_order_sha256": sha256_bytes(
            "\n".join(r["media_sha256"] for r in records).encode()
        ),
        "test_unique_speakers": len({r["speaker_group_sha256"] for r in records}),
        "class_counts": dict(Counter(r["target"] for r in records)),
        "disjointness": disjointness,
        "encoder_revision": ENCODER_REVISION,
        "encoder_weights_sha256": sha256_file(weights_path),
        "encoder_total_parameters": sum(parameter.numel() for parameter in encoder.parameters()),
        "feature_pooling": (
            "first ceil(samples/320) Whisper encoder tokens; float32 mean; BF16 CUDA autocast"
        ),
        "readout_sha256": sha256_file(checkpoint),
        "legacy_baseline_readout_sha256": sha256_file(baseline_checkpoint),
        "readout_trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "calibration_report_sha256": sha256_file(calibration_report),
        "frozen_temperature": EXPECTED_TEMPERATURE,
        "metrics_legacy_baseline": baseline_metrics,
        "metrics_uncalibrated": raw_metrics,
        "metrics_temperature_scaled": calibrated_metrics,
        "speaker_group_bootstrap": bootstrap,
        "per_class_metrics": per_class,
        "feature_extraction_seconds": feature_seconds,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "hotgpu_gate": gate,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "datasets": importlib.metadata.version("datasets"),
        "training_or_calibration_on_test": False,
        "checkpoint_selection_on_test": False,
        "sealed_audit_loaded": False,
        "predictions_sha256": sha256_file(predictions),
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lock["status"] = "complete"
    lock["finished_at_utc"] = datetime.now(UTC).isoformat()
    (output_dir / "run-lock.json").write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("manifests/candidates/speech-commands-test.yaml")
    )
    parser.add_argument("--calibration-report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--encoder-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature-batch-size", type=int, default=32)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                manifest_path=args.manifest,
                calibration_report=args.calibration_report,
                checkpoint=args.checkpoint,
                baseline_checkpoint=args.baseline_checkpoint,
                sample_dir=args.sample_dir,
                encoder_dir=args.encoder_dir,
                source_dir=args.source_dir,
                output_dir=args.output_dir,
                feature_batch_size=args.feature_batch_size,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
