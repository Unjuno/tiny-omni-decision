"""Compare random and phoneme/lexical/length-matched LibriSpeech distractors.

This is a frozen-model diagnostic. It does not fit, select, calibrate, or alter
the Speech Commands readout and never loads a test or sealed-audit split.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import random
import subprocess
import time
import wave
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from transformers import AutoTokenizer

from tiny_omni_decision.audio_hard_negatives import (
    DistractorMatch,
    TranscriptRecord,
    find_hard_distractors,
    make_transcript_record,
    multiset_dice_similarity,
    normalize_transcript,
    parse_cmudict,
    phoneme_sequence_similarity,
    vocabulary_jaccard,
)
from tiny_omni_decision.audio_runtime import DecisionQuery, SpeechCommandsMultiQueryRuntime
from tiny_omni_decision.bundle_validation import verify_bundle

OPENSLR_DEV_CLEAN_MD5 = "42e2234ba48799c1f50f24a7926300a1"
OPENSLR_DEV_CLEAN_SHA256 = "76f87d090650617fca0cac8f88b9416e0ebf80350acb97b343a85fa903728ab3"
OPENSLR_DEV_CLEAN_URL = "https://openslr.org/resources/12/dev-clean.tar.gz"
CMUDICT_SHA256 = "81917843c7f44ce2b094ac63873c2c7a4cf802040792c455ba3ca406891c3d22"
CMUDICT_LICENSE_SHA256 = "bd4ce8e44170a5f9f481310ca85c51de3c4f851a65e679b40e603b143bd3542a"
CMUDICT_REVISION = "cmusphinx/cmudict@74790861f652b15e4ac49015a90074ad62a27690"
QUESTION = "Which exact transcript matches the spoken audio?"
CONDITIONS = ("random_distractors", "hard_distractors")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--cmudict", type=Path, required=True)
    parser.add_argument("--cmudict-license", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--baseline-run-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--audio-batch-size", type=int, default=8)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_records(
    dataset_root: Path, pronunciations: dict[str, tuple[str, ...]]
) -> tuple[list[dict[str, Any]], list[TranscriptRecord], dict[str, int]]:
    root = dataset_root / "LibriSpeech" / "dev-clean"
    if not root.is_dir():
        raise FileNotFoundError(f"expected extracted LibriSpeech/dev-clean at {root}")
    transcript_paths = sorted(root.rglob("*.trans.txt"), key=lambda path: path.as_posix())
    raw_rows: list[dict[str, Any]] = []
    records: list[TranscriptRecord] = []
    excluded_oov = 0
    malformed = 0
    seen_ids: set[str] = set()
    seen_texts: set[str] = set()
    duplicate_text = 0
    for transcript_path in transcript_paths:
        for line in transcript_path.read_text(encoding="utf-8").splitlines():
            sample_id, separator, text = line.partition(" ")
            if not separator or not text.strip():
                malformed += 1
                continue
            if sample_id in seen_ids:
                raise ValueError(f"duplicate LibriSpeech utterance id: {sample_id}")
            seen_ids.add(sample_id)
            parts = sample_id.split("-")
            if len(parts) != 3:
                malformed += 1
                continue
            speaker_id, chapter_id, _ = parts
            audio_path = transcript_path.parent / f"{sample_id}.flac"
            if not audio_path.is_file():
                raise FileNotFoundError(audio_path)
            info = sf.info(audio_path)
            if info.samplerate != 16_000 or info.channels != 1:
                raise ValueError(f"unexpected audio format for {sample_id}: {info}")
            normalized = normalize_transcript(text)
            if normalized in seen_texts:
                duplicate_text += 1
            seen_texts.add(normalized)
            record = make_transcript_record(
                sample_id, speaker_id, chapter_id, text, pronunciations
            )
            if record is None:
                excluded_oov += 1
                continue
            raw_rows.append(
                {
                    "id": sample_id,
                    "speaker_id": speaker_id,
                    "chapter_id": chapter_id,
                    "text": record.text,
                    "audio_path": audio_path,
                    "audio_sha256": _sha256(audio_path),
                    "duration_seconds": info.duration,
                    "words": list(record.words),
                }
            )
            records.append(record)
    counts = {
        "transcript_files": len(transcript_paths),
        "raw_rows": len(seen_ids),
        "malformed_rows": malformed,
        "duplicate_normalized_transcripts": duplicate_text,
        "excluded_cmu_oov_rows": excluded_oov,
        "phonemized_rows": len(records),
        "unique_speakers_phonemized": len({row.speaker_id for row in records}),
        "unique_chapters_phonemized": len({row.chapter_id for row in records}),
        "unique_media_sha256": len({row["audio_sha256"] for row in raw_rows}),
    }
    return raw_rows, records, counts


def _stable_seed(seed: int, sample_id: str) -> int:
    data = f"{seed}:{sample_id}".encode()
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "big")


def _filter_text_token_limit(
    rows: list[dict[str, Any]],
    records: list[TranscriptRecord],
    tokenizer: Any,
    *,
    max_tokens: int = 64,
) -> tuple[list[dict[str, Any]], list[TranscriptRecord], dict[str, int]]:
    if len(rows) != len(records):
        raise ValueError("media rows and transcript records are not aligned")
    allowed_ids: set[str] = set()
    for start in range(0, len(records), 256):
        batch = records[start : start + 256]
        texts = [f"Candidate answer: {record.text}." for record in batch]
        encoded = tokenizer(texts, truncation=False, padding=False)
        lengths = [len(input_ids) for input_ids in encoded["input_ids"]]
        allowed_ids.update(
            record.sample_id
            for record, length in zip(batch, lengths, strict=True)
            if length <= max_tokens
        )
    filtered_rows = [row for row in rows if row["id"] in allowed_ids]
    filtered_records = [record for record in records if record.sample_id in allowed_ids]
    return filtered_rows, filtered_records, {
        "phonemized_rows_before_runtime_text_limit": len(records),
        "excluded_transcript_rows_over_runtime_text_limit": len(records) - len(filtered_rows),
        "rows_within_runtime_text_limit": len(filtered_rows),
        "runtime_option_text_max_tokens": max_tokens,
    }


def _make_example_sets(
    rows: list[dict[str, Any]],
    records: list[TranscriptRecord],
    *,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if len(rows) != len(records):
        raise ValueError("media rows and phonemized transcripts are not aligned")
    record_by_id = {record.sample_id: record for record in records}
    if len(record_by_id) != len(records):
        raise ValueError("phonemized transcript IDs must be unique")
    candidates = list(records)
    examples: list[dict[str, Any]] = []
    excluded_no_hard = 0
    excluded_no_random = 0
    for row in rows:
        target = record_by_id[row["id"]]
        hard: list[DistractorMatch] = find_hard_distractors(target, candidates, count=3)
        if len(hard) != 3:
            excluded_no_hard += 1
            continue
        hard_ids = {item.sample_id for item in hard}
        random_pool = [
            candidate
            for candidate in candidates
            if candidate.sample_id != target.sample_id
            and candidate.speaker_id != target.speaker_id
            and candidate.chapter_id != target.chapter_id
            and normalize_transcript(candidate.text) != normalize_transcript(target.text)
            and candidate.sample_id not in hard_ids
        ]
        if len(random_pool) < 3:
            excluded_no_random += 1
            continue
        rng = random.Random(_stable_seed(seed, target.sample_id))
        random_distractors = rng.sample(random_pool, 3)
        target_position = rng.randrange(4)
        condition_options: dict[str, list[str]] = {}
        condition_metadata: dict[str, list[dict[str, Any]]] = {}
        for condition, distractors in (
            ("random_distractors", random_distractors),
            ("hard_distractors", hard),
        ):
            distractor_texts = [item.text for item in distractors]
            if len(set(distractor_texts)) != 3:
                raise ValueError(f"duplicate distractor text for {target.sample_id}/{condition}")
            options = list(distractor_texts)
            options.insert(target_position, target.text)
            if len(set(options)) != 4:
                raise ValueError(f"duplicate option text for {target.sample_id}/{condition}")
            condition_options[condition] = options
            metadata = []
            for item in distractors:
                candidate = record_by_id[item.sample_id]
                target_len = max(1, len(normalize_transcript(target.text)))
                candidate_len = len(normalize_transcript(candidate.text))
                metadata.append(
                    {
                        "sample_id": item.sample_id,
                        "text": item.text,
                        "word_count": len(candidate.words),
                        "phoneme_similarity": phoneme_sequence_similarity(
                            target.phonemes, candidate.phonemes
                        ),
                        "phoneme_bigram_dice": multiset_dice_similarity(
                            target.phonemes, candidate.phonemes
                        ),
                        "vocabulary_jaccard": vocabulary_jaccard(
                            target.words, candidate.words
                        ),
                        "word_count_delta": abs(len(target.words) - len(candidate.words)),
                        "character_length_ratio": min(target_len, candidate_len)
                        / max(target_len, candidate_len),
                    }
                )
            condition_metadata[condition] = metadata
        examples.append(
            {
                **row,
                "target_index": target_position,
                "question": QUESTION,
                "options_by_condition": condition_options,
                "distractors_by_condition": condition_metadata,
            }
        )
    return examples, {
        "eligible_phonemized_targets": len(rows),
        "examples_with_both_conditions": len(examples),
        "excluded_insufficient_hard_distractors": excluded_no_hard,
        "excluded_insufficient_random_distractors": excluded_no_random,
    }


def _metric_summary(predictions: list[dict[str, Any]]) -> dict[str, float | int]:
    if not predictions:
        raise ValueError("cannot score an empty prediction group")
    probabilities = np.asarray([item["probabilities"] for item in predictions], dtype=np.float64)
    targets = np.asarray([int(item["target_index"]) for item in predictions], dtype=np.int64)
    if probabilities.ndim != 2 or probabilities.shape != (len(targets), 4):
        raise ValueError("four-option probability matrix is required")
    if not np.isfinite(probabilities).all() or not np.allclose(
        probabilities.sum(axis=1), 1.0, atol=1e-5
    ):
        raise ValueError("probabilities must be finite and normalized")
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == targets
    target_probability = probabilities[np.arange(len(targets)), targets]
    one_hot = np.eye(probabilities.shape[1], dtype=np.float64)[targets]
    ece = 0.0
    bin_counts = []
    for index in range(15):
        low, high = index / 15, (index + 1) / 15
        mask = (confidence >= low) & (confidence < high if index < 14 else confidence <= high)
        count = int(mask.sum())
        bin_counts.append(count)
        if count:
            ece += count / len(targets) * abs(float(correct[mask].mean() - confidence[mask].mean()))
    return {
        "count": int(len(targets)),
        "accuracy": float(correct.mean()),
        "nll": float(-np.log(np.maximum(target_probability, 1e-12)).mean()),
        "brier": float(np.square(probabilities - one_hot).sum(axis=1).mean()),
        "ece_15_bins": float(ece),
        "ece_bin_counts": bin_counts,
        "query_count": int(len(targets)),
    }


def _bootstrap_differences(
    examples: list[dict[str, Any]],
    predictions_by_condition: dict[str, list[dict[str, Any]]],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    if replicates < 100:
        raise ValueError("paired speaker bootstrap requires at least 100 replicates")
    speakers = sorted({str(item["speaker_id"]) for item in examples})
    by_speaker: dict[str, dict[str, list[dict[str, Any]]]] = {
        speaker: {condition: [] for condition in CONDITIONS} for speaker in speakers
    }
    for condition in CONDITIONS:
        expected = len(examples)
        predictions = predictions_by_condition[condition]
        if len(predictions) != expected:
            raise ValueError(f"wrong prediction count for {condition}")
        by_id = {item["sample_id"]: item for item in predictions}
        for example in examples:
            prediction = by_id.get(example["id"])
            if prediction is None:
                raise ValueError(f"missing {condition} prediction for {example['id']}")
            by_speaker[example["speaker_id"]][condition].append(prediction)
    rng = random.Random(seed)
    metrics = ("accuracy", "nll", "brier", "ece_15_bins")
    differences: dict[str, list[float]] = {metric: [] for metric in metrics}
    for _ in range(replicates):
        sampled = [rng.choice(speakers) for _ in speakers]
        rows = {
            condition: [
                item
                for speaker in sampled
                for item in by_speaker[speaker][condition]
            ]
            for condition in CONDITIONS
        }
        hard = _metric_summary(rows["hard_distractors"])
        random_metrics = _metric_summary(rows["random_distractors"])
        for metric in metrics:
            differences[metric].append(float(hard[metric]) - float(random_metrics[metric]))
    result = {}
    for metric, values in differences.items():
        values.sort()
        result[metric] = {
            "mean_difference": sum(values) / replicates,
            "percentile_95_ci": [
                values[math.floor(0.025 * (replicates - 1))],
                values[math.floor(0.975 * (replicates - 1))],
            ],
        }
    return {
        "unit": "LibriSpeech speaker IDs, keeping all utterances per sampled speaker",
        "speaker_groups": len(speakers),
        "replicates": replicates,
        "seed": seed,
        "low_power": len(speakers) < 200,
        "difference_direction": "hard_distractors minus random_distractors",
        "metrics": result,
    }


def _hardness_summary(examples: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "phoneme_similarity",
        "phoneme_bigram_dice",
        "vocabulary_jaccard",
        "word_count_delta",
        "character_length_ratio",
    )
    result: dict[str, Any] = {}
    for condition in CONDITIONS:
        values = {field: [] for field in fields}
        for example in examples:
            for item in example["distractors_by_condition"][condition]:
                for field in fields:
                    values[field].append(float(item[field]))
        result[condition] = {
            field: sum(items) / len(items) if items else None for field, items in values.items()
        }
    return result


def _write_pcm16_wav(source: Path, destination: Path) -> str:
    samples, samplerate = sf.read(source, dtype="int16", always_2d=False)
    if samplerate != 16_000 or samples.ndim != 1 or samples.size == 0:
        raise ValueError(f"expected nonempty mono 16 kHz source: {source}")
    raw = np.ascontiguousarray(samples, dtype="<i2").tobytes()
    with wave.open(str(destination), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16_000)
        stream.writeframes(raw)
    return hashlib.sha256(raw).hexdigest()


def main() -> None:
    started = datetime.now(UTC)
    wall_started = time.perf_counter()
    args = _args()
    dataset_root = args.dataset_root.resolve()
    cmudict_path = args.cmudict.resolve()
    cmudict_license_path = args.cmudict_license.resolve()
    bundle = args.bundle.resolve()
    baseline_path = args.baseline_run_report.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to reuse evaluation output: {output}")
    if args.audio_batch_size != 8:
        raise ValueError("batch size 8 is fixed to match the frozen-feature reference extraction")
    if (
        not dataset_root.is_dir()
        or not cmudict_path.is_file()
        or not cmudict_license_path.is_file()
    ):
        raise FileNotFoundError("dataset root, CMUdict, and its license must all exist")
    archive = dataset_root / "dev-clean.tar.gz"
    if not archive.is_file():
        raise FileNotFoundError(archive)
    archive_md5 = _md5(archive)
    if archive_md5 != OPENSLR_DEV_CLEAN_MD5:
        raise ValueError(f"OpenSLR dev-clean MD5 mismatch: {archive_md5}")
    archive_sha256 = _sha256(archive)
    if archive_sha256 != OPENSLR_DEV_CLEAN_SHA256:
        raise ValueError(f"OpenSLR dev-clean SHA-256 mismatch: {archive_sha256}")
    cmudict_sha256 = _sha256(cmudict_path)
    cmudict_license_sha256 = _sha256(cmudict_license_path)
    if cmudict_sha256 != CMUDICT_SHA256:
        raise ValueError(f"CMUdict SHA-256 mismatch: {cmudict_sha256}")
    if cmudict_license_sha256 != CMUDICT_LICENSE_SHA256:
        raise ValueError(f"CMUdict license SHA-256 mismatch: {cmudict_license_sha256}")
    bundle_manifest_path = bundle / "bundle-manifest.json"
    manifest = json.loads(bundle_manifest_path.read_text(encoding="utf-8"))
    verify_bundle(bundle, manifest)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if baseline.get("status") != "complete_frozen_audio_multiquery_development_experiment":
        raise ValueError("baseline run is not the expected completed Speech Commands model")
    source_hash = baseline.get("model", {}).get("readout_state_dict_sha256")
    if source_hash != manifest.get("models", {}).get("readout", {}).get("sha256"):
        raise ValueError("bundle readout differs from the immutable Speech Commands baseline")
    if (
        baseline.get("test_split_loaded") is not False
        or baseline.get("sealed_audit_loaded") is not False
    ):
        raise ValueError("baseline metadata indicates test or sealed audit use")
    pronunciations = parse_cmudict(cmudict_path.read_text(encoding="latin-1"))
    rows, records, input_counts = _load_records(dataset_root / "source", pronunciations)
    tokenizer = AutoTokenizer.from_pretrained(
        bundle / "models" / "text-minilm-en", local_files_only=True
    )
    rows, records, token_limit_counts = _filter_text_token_limit(
        rows, records, tokenizer, max_tokens=64
    )
    input_counts.update(token_limit_counts)
    input_counts["unique_speakers_after_runtime_text_limit"] = len(
        {row["speaker_id"] for row in rows}
    )
    input_counts["unique_chapters_after_runtime_text_limit"] = len(
        {row["chapter_id"] for row in rows}
    )
    del tokenizer
    examples, construction_counts = _make_example_sets(rows, records, seed=args.seed)
    if not examples:
        raise ValueError("no LibriSpeech rows support both four-choice conditions")
    speaker_ids = {example["speaker_id"] for example in examples}
    if len(speaker_ids) < 20:
        raise ValueError("need at least 20 independent speakers for this diagnostic")
    order_hash = hashlib.sha256(
        "\n".join(example["id"] for example in examples).encode("utf-8")
    ).hexdigest()
    output.mkdir(parents=True)
    examples_path = output / "evaluation-examples.jsonl"
    source_root = (dataset_root / "source").resolve()
    serialized_examples = []
    for example in examples:
        serialized = dict(example)
        serialized["audio_path"] = Path(example["audio_path"]).resolve().relative_to(
            source_root
        ).as_posix()
        serialized_examples.append(serialized)
    examples_path.write_text(
        "".join(
            json.dumps(example, ensure_ascii=False, sort_keys=True) + "\n"
            for example in serialized_examples
        ),
        encoding="utf-8",
    )

    if args.preflight_only:
        preflight = {
            "schema_version": 1,
            "status": "complete_librispeech_distractor_preflight",
            "source_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
                cwd=Path(__file__).resolve().parents[1],
            ).stdout.strip(),
            "runner_sha256": _sha256(Path(__file__).resolve()),
            "phoneme_helper_sha256": _sha256(
                Path(__file__).resolve().parents[1]
                / "src"
                / "tiny_omni_decision"
                / "audio_hard_negatives.py"
            ),
            "source": {
                "dataset": "LibriSpeech SLR12 dev-clean",
                "archive_md5_verified": archive_md5,
                "archive_sha256": archive_sha256,
                "cmudict_revision": CMUDICT_REVISION,
                "cmudict_sha256": cmudict_sha256,
                "cmudict_license_sha256": cmudict_license_sha256,
                "input_counts": input_counts,
                "construction_counts": construction_counts,
                "speaker_groups": len(speaker_ids),
                "chapter_groups": len({example["chapter_id"] for example in examples}),
                "unique_media": len({example["audio_sha256"] for example in examples}),
                "ordered_sample_id_sha256": order_hash,
                "hardness_summary": _hardness_summary(examples),
                "target_position_counts": {
                    str(index): sum(
                        int(example["target_index"]) == index for example in examples
                    )
                    for index in range(4)
                },
            },
            "evaluation_examples": {
                "path": str(examples_path),
                "sha256": _sha256(examples_path),
                "bytes": examples_path.stat().st_size,
            },
            "training_performed": False,
            "tokenizer_loaded": True,
            "model_weights_loaded": False,
            "sealed_audit_loaded": False,
        }
        (output / "preflight.json").write_text(
            json.dumps(preflight, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(json.dumps(preflight, indent=2, ensure_ascii=False))
        return

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    runtime_started = time.perf_counter()
    runtime = SpeechCommandsMultiQueryRuntime(
        bundle / "models", bundle / "readout" / "best-readout.pt", device=device
    )
    if runtime.parameter_count != int(manifest["total_parameters"]):
        raise ValueError("runtime parameter count differs from bundle manifest")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        load_peak = torch.cuda.max_memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
    else:
        load_peak = None
    load_seconds = time.perf_counter() - runtime_started

    predictions = {condition: [] for condition in CONDITIONS}
    temporary_wav_root = output / "temporary-wav"
    temporary_wav_root.mkdir()
    inference_started = time.perf_counter()
    for batch_start in range(0, len(examples), args.audio_batch_size):
        batch = examples[batch_start : batch_start + args.audio_batch_size]
        wav_paths = []
        for slot, example in enumerate(batch):
            wav_path = temporary_wav_root / f"slot-{slot}.wav"
            pcm_hash = _write_pcm16_wav(Path(example["audio_path"]), wav_path)
            example["pcm16_sha256"] = pcm_hash
            wav_paths.append(wav_path)
        queries_by_audio = [
            [
                DecisionQuery(QUESTION, tuple(example["options_by_condition"][condition]))
                for condition in CONDITIONS
            ]
            for example in batch
        ]
        result_batch = runtime.decide_batch(wav_paths, queries_by_audio)
        for example, results in zip(batch, result_batch, strict=True):
            if len(results) != len(CONDITIONS):
                raise ValueError("runtime did not return both distractor conditions")
            for condition, result in zip(CONDITIONS, results, strict=True):
                prediction = {
                    "sample_id": example["id"],
                    "speaker_id": example["speaker_id"],
                    "chapter_id": example["chapter_id"],
                    "audio_sha256": example["audio_sha256"],
                    "pcm16_sha256": example["pcm16_sha256"],
                    "condition": condition,
                    "question": QUESTION,
                    "options": example["options_by_condition"][condition],
                    "target_index": int(example["target_index"]),
                    "prediction_index": int(result.predicted_index),
                    "target": example["text"],
                    "prediction": example["options_by_condition"][condition][
                        int(result.predicted_index)
                    ],
                    "logits": result.option_logits.float().tolist(),
                    "probabilities": result.probabilities.float().tolist(),
                    "distractors": example["distractors_by_condition"][condition],
                }
                predictions[condition].append(prediction)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        inference_peak = torch.cuda.max_memory_allocated(device)
    else:
        inference_peak = None
    inference_seconds = time.perf_counter() - inference_started
    # Temporary WAVs are the only files removed, and were created in this new run directory.
    for wav_path in temporary_wav_root.iterdir():
        wav_path.unlink()
    temporary_wav_root.rmdir()

    metrics = {
        condition: _metric_summary(predictions[condition]) for condition in CONDITIONS
    }
    prediction_paths: dict[str, dict[str, Any]] = {}
    for condition in CONDITIONS:
        path = output / f"{condition}-predictions.jsonl"
        path.write_text(
            "".join(
                json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
                for item in predictions[condition]
            ),
            encoding="utf-8",
        )
        prediction_paths[condition] = {
            "path": str(path),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
    bootstrap = _bootstrap_differences(
        examples,
        predictions,
        replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    ).stdout.strip()
    report = {
        "schema_version": 1,
        "status": "complete_frozen_librispeech_distractor_diagnostic",
        "scope": (
            "Post-hoc, no-training comparison of random versus phoneme-, vocabulary-, "
            "and length-matched four-choice transcript distractors using the unchanged "
            "Speech Commands Audio+Text checkpoint. This is an OOD readout diagnostic, "
            "not a broad ASR or product-quality result."
        ),
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(UTC).isoformat(),
        "source_commit": source_commit,
        "runner_sha256": _sha256(Path(__file__).resolve()),
        "phoneme_helper_sha256": _sha256(
            Path(__file__).resolve().parents[1]
            / "src"
            / "tiny_omni_decision"
            / "audio_hard_negatives.py"
        ),
        "baseline": {
            "experiment_id": baseline["experiment_id"],
            "run_report_sha256": _sha256(baseline_path),
            "bundle_manifest_sha256": _sha256(bundle_manifest_path),
            "readout_sha256": source_hash,
            "speech_commands_validation_manifest_sha256": baseline["data_hashes"][
                "validation_manifest"
            ],
            "speech_commands_macro_validation_metrics": baseline["validation_metrics"][
                "macro_question_type"
            ],
            "unchanged": True,
        },
        "source": {
            "dataset": "LibriSpeech SLR12 dev-clean",
            "url": OPENSLR_DEV_CLEAN_URL,
            "license": "CC BY 4.0",
            "official_md5_expected": OPENSLR_DEV_CLEAN_MD5,
            "archive_md5_verified": archive_md5,
            "archive_sha256": archive_sha256,
            "cmudict_revision": CMUDICT_REVISION,
            "cmudict_sha256": cmudict_sha256,
            "cmudict_license_sha256": cmudict_license_sha256,
            "cmudict_attribution": "CMU Pronouncing Dictionary; Carnegie Mellon University",
            "transcript_manifest_sha256": _sha256(examples_path),
            "ordered_sample_id_sha256": order_hash,
            "input_counts": input_counts,
            "construction_counts": construction_counts,
            "speaker_groups": len(speaker_ids),
            "chapter_groups": len({example["chapter_id"] for example in examples}),
            "unique_media": len({example["audio_sha256"] for example in examples}),
            "unique_target_transcripts": len(
                {normalize_transcript(example["text"]) for example in examples}
            ),
            "pcm_preprocessing": (
                "soundfile FLAC decode to int16, mono 16 kHz PCM WAV; no resampling"
            ),
        },
        "task": {
            "question": QUESTION,
            "options": 4,
            "maximum_option_tokens": 64,
            "target_positions": {
                str(index): sum(int(example["target_index"]) == index for example in examples)
                for index in range(4)
            },
            "condition_definitions": {
                "random_distractors": (
                    "Three seeded random transcripts from different speakers and chapters; "
                    "random candidates are disjoint from that row's hard candidates."
                ),
                "hard_distractors": (
                    "Top three different-speaker, different-chapter, nonduplicate transcripts "
                    "with word count within one, shared vocabulary, and character-length ratio "
                    "at least 0.65, ranked by CMU phoneme sequence similarity, vocabulary Jaccard, "
                    "then length similarity."
                ),
            },
            "hardness_summary": _hardness_summary(examples),
            "model_selection_or_calibration": False,
            "test_split_loaded": False,
            "sealed_audit_loaded": False,
        },
        "model": {
            "bundle_id": manifest["bundle_id"],
            "runtime_parameters": runtime.parameter_count,
            "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "audio_encoder_batch_size": args.audio_batch_size,
            "audio_encoder_revision": manifest["models"]["audio"]["revision"],
            "text_encoder_revision": manifest["models"]["text"]["revision"],
            "readout_sha256": source_hash,
            "software": {
                "python": __import__("sys").version.split()[0],
                "torch": torch.__version__,
                "transformers": importlib.metadata.version("transformers"),
                "soundfile": importlib.metadata.version("soundfile"),
                "numpy": np.__version__,
            },
        },
        "metrics": metrics,
        "paired_speaker_bootstrap": bootstrap,
        "runtime": {
            "load_seconds": load_seconds,
            "inference_seconds": inference_seconds,
            "total_seconds": time.perf_counter() - wall_started,
            "runner_sha256": _sha256(Path(__file__).resolve()),
            "phoneme_helper_sha256": _sha256(
                Path(__file__).resolve().parents[1]
                / "src"
                / "tiny_omni_decision"
                / "audio_hard_negatives.py"
            ),
            "seconds_per_audio_observation": inference_seconds / len(examples),
            "cuda_load_peak_allocated_bytes": load_peak,
            "cuda_inference_peak_allocated_bytes": inference_peak,
            "cost_usd": 0,
        },
        "artifacts": {
            "examples": {
                "path": str(examples_path),
                "sha256": _sha256(examples_path),
                "bytes": examples_path.stat().st_size,
            },
            "predictions": prediction_paths,
        },
        "limitations": [
            (
                "This utterance-transcript task and question format were not used to train "
                "the fixed Speech Commands scorer; low performance is an OOD transfer "
                "finding, not an unbiased estimate of general speech recognition quality."
            ),
            (
                "The task uses LibriSpeech read audiobook speech and its provided transcripts; "
                "it does not establish conversational speech understanding or acoustic "
                "robustness outside dev-clean."
            ),
            (
                "The same dev-clean transcripts supply distractor text, but target clips, "
                "source audio, speakers and chapters are retained as separate evaluation "
                "rows; speaker-cluster bootstrap accounts for within-speaker dependence only."
            ),
            (
                "LibriSpeech dev-clean is development data and public; this diagnostic is "
                "not blind or sealed and was not used for model selection."
            ),
            (
                "The legacy Speech Commands validation metrics and LibriSpeech metrics "
                "concern different task definitions and must not be treated as a direct "
                "quality delta."
            ),
        ],
        "training_performed": False,
        "model_weights_changed": False,
        "baseline_checkpoint_modified": False,
    }
    report_path = output / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
