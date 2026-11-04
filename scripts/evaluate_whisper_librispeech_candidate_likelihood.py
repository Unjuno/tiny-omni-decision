"""One-shot Whisper Tiny likelihood evaluation on fixed LibriSpeech choices.

The runner is inference-only. It reads the existing dev-clean tar sequentially
without extraction, uses the already fixed random/hard options, and never loads
LibriSpeech test or a project sealed audit.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import platform
import random
import tarfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import av
import numpy as np
import torch
import transformers
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from tiny_omni_decision.whisper_option_likelihood import (
    build_decoder_batch,
    candidate_probabilities,
    mean_transcript_log_likelihoods,
    ordered_sample_ids_sha256,
)

DATA_ARCHIVE_SHA256 = "76f87d090650617fca0cac8f88b9416e0ebf80350acb97b343a85fa903728ab3"
EXAMPLES_SHA256 = "1f17d86a15faecc3ba3e13cd406dfd0119ef17850d1a7206956a70f4eb801923"
MODEL_SHA256 = "7ebd0e69e78190ffe1438491fa05cc1f5c1aa3a4c4db3bc1723adbb551ea2395"
BASELINE_REPORT_SHA256 = "087988dcfc2373a8cb38f339dd26e8badd2d7e69bf76ea13ac91edcc9f56c791"
ORDERED_SAMPLE_IDS_SHA256 = "709687d926f46907a3c5d447d8921b51cb8f1188163f25fd777b11dfae34384c"
EXPECTED_EXAMPLES = 1938
CONDITIONS = ("random_distractors", "hard_distractors")
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 17
ECE_BINS = 15

BASELINE_METRICS = {
    "random_distractors": {
        "accuracy": 0.2652,
        "nll": 2.6085,
        "brier": 1.0880,
        "ece_15_bins": 0.4487,
    },
    "hard_distractors": {
        "accuracy": 0.2637,
        "nll": 2.5361,
        "brier": 1.0841,
        "ece_15_bins": 0.4425,
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ordered_ids_sha256(examples: list[dict[str, Any]]) -> str:
    return ordered_sample_ids_sha256([str(row["id"]) for row in examples])


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _metric_summary(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    if not predictions:
        raise ValueError("cannot score an empty prediction set")
    probabilities = np.asarray([item["probabilities"] for item in predictions], dtype=np.float64)
    targets = np.asarray([int(item["target_index"]) for item in predictions])
    if probabilities.shape != (len(targets), 4):
        raise ValueError("four-choice probability vectors are required")
    if not np.isfinite(probabilities).all() or not np.allclose(
        probabilities.sum(axis=1), 1.0, atol=1e-5
    ):
        raise ValueError("candidate probabilities must be finite and normalized")
    correct = probabilities.argmax(axis=1) == targets
    confidence = probabilities.max(axis=1)
    target_probability = probabilities[np.arange(len(targets)), targets]
    one_hot = np.eye(4, dtype=np.float64)[targets]
    ece = 0.0
    bin_counts = []
    for bin_index in range(ECE_BINS):
        lower = bin_index / ECE_BINS
        upper = (bin_index + 1) / ECE_BINS
        mask = (confidence >= lower) & (
            confidence < upper if bin_index < ECE_BINS - 1 else confidence <= upper
        )
        count = int(mask.sum())
        bin_counts.append(count)
        if count:
            ece += count / len(targets) * abs(float(correct[mask].mean() - confidence[mask].mean()))
    return {
        "count": len(targets),
        "accuracy": float(correct.mean()),
        "nll": float(-np.log(np.maximum(target_probability, 1e-12)).mean()),
        "brier": float(np.square(probabilities - one_hot).sum(axis=1).mean()),
        "ece_15_bins": float(ece),
        "ece_bin_counts": bin_counts,
        "mean_confidence": float(confidence.mean()),
    }


def _paired_speaker_bootstrap(
    examples: list[dict[str, Any]],
    predictions: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    speakers = sorted({str(row["speaker_id"]) for row in examples})
    by_speaker: dict[str, dict[str, list[dict[str, Any]]]] = {
        speaker: {condition: [] for condition in CONDITIONS} for speaker in speakers
    }
    for condition in CONDITIONS:
        by_id = {row["sample_id"]: row for row in predictions[condition]}
        if len(by_id) != len(examples):
            raise ValueError(f"duplicate or missing {condition} prediction IDs")
        for example in examples:
            prediction = by_id.get(example["id"])
            if prediction is None:
                raise ValueError(f"missing prediction for {example['id']}/{condition}")
            by_speaker[str(example["speaker_id"])][condition].append(prediction)

    rng = random.Random(BOOTSTRAP_SEED)
    metric_names = ("accuracy", "nll", "brier", "ece_15_bins")
    differences: dict[str, list[float]] = {name: [] for name in metric_names}
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled_speakers = [rng.choice(speakers) for _ in speakers]
        sampled = {
            condition: [
                prediction
                for speaker in sampled_speakers
                for prediction in by_speaker[speaker][condition]
            ]
            for condition in CONDITIONS
        }
        random_metrics = _metric_summary(sampled["random_distractors"])
        hard_metrics = _metric_summary(sampled["hard_distractors"])
        for metric in metric_names:
            differences[metric].append(float(hard_metrics[metric]) - float(random_metrics[metric]))

    interval_results: dict[str, dict[str, float]] = {}
    for metric, values in differences.items():
        values.sort()
        interval_results[metric] = {
            "mean_difference": float(sum(values) / BOOTSTRAP_REPLICATES),
            "percentile_95_ci": [
                float(values[math.floor(0.025 * (BOOTSTRAP_REPLICATES - 1))]),
                float(values[math.floor(0.975 * (BOOTSTRAP_REPLICATES - 1))]),
            ],
        }
    return {
        "unit": "LibriSpeech speaker IDs, retaining all utterances per sampled speaker",
        "speaker_groups": len(speakers),
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "contrast": "hard_distractors minus random_distractors",
        "metrics": interval_results,
    }


def _decode_flac(payload: bytes) -> np.ndarray:
    with av.open(io.BytesIO(payload), mode="r") as container:
        audio_streams = [stream for stream in container.streams if stream.type == "audio"]
        if len(audio_streams) != 1 or audio_streams[0].rate != 16_000:
            raise ValueError("LibriSpeech audio must be mono 16-kHz FLAC")
        frames = list(container.decode(audio=audio_streams[0].index))
        arrays = []
        for frame in frames:
            array = frame.to_ndarray()
            if array.ndim == 2:
                if array.shape[0] != 1:
                    raise ValueError("LibriSpeech audio must be mono")
                array = array[0]
            if np.issubdtype(array.dtype, np.integer):
                scale = float(max(abs(np.iinfo(array.dtype).min), np.iinfo(array.dtype).max))
                array = array.astype(np.float32) / scale
            else:
                array = array.astype(np.float32)
            arrays.append(array)
        if not arrays:
            raise ValueError("FLAC decode returned no audio frames")
        waveform = np.concatenate(arrays)
        if waveform.ndim != 1 or not np.isfinite(waveform).all():
            raise ValueError("decoded waveform must be finite mono float32")
        return waveform


def _model_prefix(processor: WhisperProcessor) -> list[int]:
    tokenizer = processor.tokenizer
    spellings = (
        "<|startoftranscript|>",
        "<|en|>",
        "<|transcribe|>",
        "<|notimestamps|>",
    )
    ids = [int(tokenizer.convert_tokens_to_ids(token)) for token in spellings]
    if len(set(ids)) != 4 or tokenizer.unk_token_id in ids:
        raise ValueError("local Whisper tokenizer does not expose the required English prefix")
    return ids


def _score_unique_candidates(
    *,
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    waveform: np.ndarray,
    transcripts: list[str],
    prefix_token_ids: list[int],
) -> tuple[dict[str, float], int]:
    tokenizer = processor.tokenizer
    candidate_token_ids = [
        tokenizer(text.strip(), add_special_tokens=False, truncation=False)["input_ids"]
        for text in transcripts
    ]
    if any(not token_ids for token_ids in candidate_token_ids):
        raise ValueError("tokenizer returned an empty candidate transcript")
    maximum_positions = int(model.config.max_target_positions)
    if any(
        len(prefix_token_ids) + len(token_ids) > maximum_positions
        for token_ids in candidate_token_ids
    ):
        raise ValueError("candidate transcript exceeds the Whisper decoder context window")
    processed = processor.feature_extractor(waveform, sampling_rate=16_000, return_tensors="pt")
    input_features = processed.input_features.to(device="cpu", dtype=torch.float32)
    with torch.inference_mode():
        encoder_hidden = model.model.encoder(input_features, return_dict=True).last_hidden_state
        decoder_input_ids, decoder_attention_mask = build_decoder_batch(
            candidate_token_ids,
            prefix_token_ids=prefix_token_ids,
            eos_token_id=int(tokenizer.eos_token_id),
            pad_token_id=int(tokenizer.pad_token_id),
            device="cpu",
        )
        expanded_encoder = encoder_hidden.expand(len(transcripts), -1, -1).contiguous()
        decoder_output = model.model.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=expanded_encoder,
            use_cache=False,
            return_dict=True,
        )
        logits = model.proj_out(decoder_output.last_hidden_state)
        scores = mean_transcript_log_likelihoods(
            logits,
            candidate_token_ids,
            prefix_length=len(prefix_token_ids),
            eos_token_id=int(tokenizer.eos_token_id),
        )
    return {
        text: float(score) for text, score in zip(transcripts, scores.cpu().tolist(), strict=True)
    }, max(len(ids) for ids in candidate_token_ids)


def _validate_inputs(
    archive_path: Path,
    examples_path: Path,
    model_dir: Path,
    baseline_report_path: Path,
) -> list[dict[str, Any]]:
    expected_hashes = (
        (archive_path, DATA_ARCHIVE_SHA256, "LibriSpeech archive"),
        (examples_path, EXAMPLES_SHA256, "fixed evaluation examples"),
        (model_dir / "model.safetensors", MODEL_SHA256, "Whisper checkpoint"),
        (baseline_report_path, BASELINE_REPORT_SHA256, "baseline report"),
    )
    for path, expected, label in expected_hashes:
        if not path.is_file():
            raise FileNotFoundError(f"missing {label}: {path}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"{label} SHA-256 mismatch: expected {expected}, got {actual}")
    examples = _load_jsonl(examples_path)
    if len(examples) != EXPECTED_EXAMPLES:
        raise ValueError(f"expected {EXPECTED_EXAMPLES} fixed examples, got {len(examples)}")
    if len({row.get("id") for row in examples}) != len(examples):
        raise ValueError("fixed example IDs must be unique")
    if _ordered_ids_sha256(examples) != ORDERED_SAMPLE_IDS_SHA256:
        raise ValueError("ordered fixed sample IDs do not match the pre-registered hash")
    if any(
        not row.get("audio_path", "").startswith("LibriSpeech/dev-clean/")
        or not isinstance(row.get("audio_sha256"), str)
        or len(row["audio_sha256"]) != 64
        for row in examples
    ):
        raise ValueError("example manifest contains an invalid dev-clean media reference")
    for condition in CONDITIONS:
        if any(
            len(row.get("options_by_condition", {}).get(condition, [])) != 4
            or len(set(row["options_by_condition"][condition])) != 4
            for row in examples
        ):
            raise ValueError(f"invalid fixed four-choice options for {condition}")
    for row in examples:
        target_index = int(row["target_index"])
        if not 0 <= target_index < 4 or any(
            row["options_by_condition"][condition][target_index].strip() != row["text"].strip()
            for condition in CONDITIONS
        ):
            raise ValueError(f"fixed target identity/position mismatch for {row['id']}")
    return examples


def _run(args: argparse.Namespace) -> dict[str, Any]:
    archive_path = args.archive.resolve()
    examples_path = args.examples.resolve()
    model_dir = args.model_dir.resolve()
    baseline_report_path = args.baseline_report.resolve()
    examples = _validate_inputs(archive_path, examples_path, model_dir, baseline_report_path)
    selected_examples = examples[:8] if args.smoke else examples
    expected_count = 8 if args.smoke else EXPECTED_EXAMPLES
    if args.smoke:
        output_dir = args.output_dir.resolve() / "smoke"
    else:
        output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    report_path = output_dir / "report.json"
    if predictions_path.exists() or report_path.exists():
        raise FileExistsError(f"refusing to overwrite completed output in {output_dir}")

    torch.set_num_threads(args.cpu_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    load_started = time.perf_counter()
    processor = WhisperProcessor.from_pretrained(str(model_dir), local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(
        str(model_dir), local_files_only=True, use_safetensors=True
    )
    model.to(device="cpu", dtype=torch.float32).eval()
    prefix_token_ids = _model_prefix(processor)
    load_seconds = time.perf_counter() - load_started
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != 37_760_640:
        raise ValueError(f"unexpected Whisper parameter count: {parameter_count}")

    expected_by_path = {row["audio_path"]: row for row in selected_examples}
    found_paths: set[str] = set()
    condition_predictions: dict[str, list[dict[str, Any]]] = {
        condition: [] for condition in CONDITIONS
    }
    archive_members_scanned = 0
    maximum_candidate_tokens = 0
    scoring_seconds = 0.0
    inference_started = time.perf_counter()
    max_option_order_delta = 0.0
    with predictions_path.open("x", encoding="utf-8", newline="\n") as prediction_stream:
        with tarfile.open(archive_path, mode="r|gz") as archive:
            for member in archive:
                archive_members_scanned += 1
                if not member.isfile() or member.name not in expected_by_path:
                    continue
                example = expected_by_path[member.name]
                handle = archive.extractfile(member)
                if handle is None:
                    raise ValueError(f"cannot read fixed audio member {member.name}")
                audio_bytes = handle.read()
                if hashlib.sha256(audio_bytes).hexdigest() != example["audio_sha256"]:
                    raise ValueError(f"audio SHA-256 mismatch for {example['id']}")
                waveform = _decode_flac(audio_bytes)
                if abs(len(waveform) / 16_000 - float(example["duration_seconds"])) > 0.15:
                    raise ValueError(f"duration mismatch for {example['id']}")
                unique_transcripts = list(
                    dict.fromkeys(
                        text.strip()
                        for condition in CONDITIONS
                        for text in example["options_by_condition"][condition]
                    )
                )
                started = time.perf_counter()
                scores_by_text, token_count = _score_unique_candidates(
                    model=model,
                    processor=processor,
                    waveform=waveform,
                    transcripts=unique_transcripts,
                    prefix_token_ids=prefix_token_ids,
                )
                scoring_seconds += time.perf_counter() - started
                maximum_candidate_tokens = max(maximum_candidate_tokens, token_count)
                for condition in CONDITIONS:
                    options = [text.strip() for text in example["options_by_condition"][condition]]
                    option_scores = torch.tensor(
                        [scores_by_text[text] for text in options], dtype=torch.float32
                    )
                    probabilities_tensor = candidate_probabilities(option_scores)
                    reverse_probabilities = candidate_probabilities(option_scores.flip(0)).flip(0)
                    max_option_order_delta = max(
                        max_option_order_delta,
                        float((probabilities_tensor - reverse_probabilities).abs().max().item()),
                    )
                    probabilities = probabilities_tensor.tolist()
                    prediction = {
                        "sample_id": example["id"],
                        "speaker_id": str(example["speaker_id"]),
                        "chapter_id": str(example["chapter_id"]),
                        "audio_sha256": example["audio_sha256"],
                        "condition": condition,
                        "options": options,
                        "target_index": int(example["target_index"]),
                        "candidate_scores": option_scores.tolist(),
                        "probabilities": probabilities,
                        "prediction_index": int(np.argmax(probabilities)),
                    }
                    condition_predictions[condition].append(prediction)
                    prediction_stream.write(json.dumps(prediction, sort_keys=True) + "\n")
                found_paths.add(member.name)
                if len(found_paths) == expected_count:
                    break
    if len(found_paths) != expected_count:
        missing = sorted(set(expected_by_path) - found_paths)
        raise ValueError(
            f"found {len(found_paths)}/{expected_count} expected audio files; missing={missing[:5]}"
        )

    if args.smoke:
        status = "complete_smoke_only"
        metrics = {
            condition: _metric_summary(condition_predictions[condition]) for condition in CONDITIONS
        }
        bootstrap = None
    else:
        status = "complete_frozen_whisper_candidate_likelihood_diagnostic"
        metrics = {
            condition: _metric_summary(condition_predictions[condition]) for condition in CONDITIONS
        }
        bootstrap = _paired_speaker_bootstrap(selected_examples, condition_predictions)
    inference_seconds = time.perf_counter() - inference_started
    report = {
        "schema_version": 1,
        "status": status,
        "scope": (
            "Fixed-model candidate-likelihood comparison on previously inspected "
            "LibriSpeech dev-clean IDs; not blind or sealed evidence."
        ),
        "started_utc": datetime.fromtimestamp(args.started_at, tz=UTC).isoformat(),
        "finished_utc": datetime.now(UTC).isoformat(),
        "source_commit": args.source_commit,
        "config_sha256": args.config_sha256,
        "experiment": "Whisper Tiny teacher-forced transcript candidate likelihood",
        "inputs": {
            "archive_sha256": DATA_ARCHIVE_SHA256,
            "examples_sha256": EXAMPLES_SHA256,
            "ordered_sample_ids_sha256": ORDERED_SAMPLE_IDS_SHA256,
            "model_sha256": MODEL_SHA256,
            "baseline_report_sha256": BASELINE_REPORT_SHA256,
            "example_count": expected_count,
            "archive_members_scanned": archive_members_scanned,
            "audio_hashes_verified": len(found_paths),
            "official_development_split_read": True,
            "official_test_read": False,
            "sealed_audit_read": False,
        },
        "model": {
            "id": "openai/whisper-tiny",
            "revision": "169d4a4341b33bc18d8881c4b69c2e104e1cc0af",
            "license": "Apache-2.0 per pinned model card",
            "parameters": parameter_count,
            "checkpoint_bytes": (model_dir / "model.safetensors").stat().st_size,
            "device": "cpu",
            "dtype": "float32",
            "trainable_parameters": 0,
            "pretraining_overlap_with_librispeech": "unknown",
        },
        "scoring": {
            "prefix_token_ids": prefix_token_ids,
            "score": "mean teacher-forced log probability of transcript tokens plus EOS",
            "probability_transform": "softmax across the four scores; no fitted temperature",
            "maximum_candidate_token_count": maximum_candidate_tokens,
            "max_probability_delta_after_option_reversal": max_option_order_delta,
        },
        "timing_seconds": {
            "model_load": load_seconds,
            "inference_total": inference_seconds,
            "candidate_scoring": scoring_seconds,
            "feature_extraction_included_in_candidate_scoring": True,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "av": av.__version__,
            "processor_backend": "PyAV FLAC decode + pinned WhisperFeatureExtractor",
            "cpu_threads": args.cpu_threads,
            "gpu_used": False,
            "cloud_cost_usd": 0,
        },
        "baseline": {
            "model": "unchanged Path A Speech Commands Audio+Text readout",
            "report_sha256": BASELINE_REPORT_SHA256,
            "metrics": BASELINE_METRICS,
            "same_sample_ids_and_option_conditions": True,
        },
        "metrics": metrics,
        "hard_minus_random_paired_speaker_bootstrap": bootstrap,
        "predictions_sha256": _sha256(predictions_path),
        "training": False,
        "calibration": False,
        "checkpoint_selection": False,
        "sealed_audit_read": False,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    args.started_at = time.time()
    report = _run(args)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
