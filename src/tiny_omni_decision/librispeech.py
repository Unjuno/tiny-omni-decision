from __future__ import annotations

import hashlib
import random
import re
import unicodedata
from collections import Counter
from typing import Any

from .schema import DecisionExample, MediaRef

SOURCE = "openslr/LibriSpeech"


def normalize_transcript(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def build_librispeech_decisions(
    records: list[dict[str, Any]], *, revision: str, seed: int = 17
) -> tuple[list[DecisionExample], list[DecisionExample], dict[str, Any]]:
    """Build four-choice spoken-transcript decisions with speaker-disjoint splits."""
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("revision must pin the OpenSLR checksum-manifest snapshot")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    normalized_by_split: dict[str, set[str]] = {"train": set(), "validation": set()}
    speakers_by_split: dict[str, set[str]] = {"train": set(), "validation": set()}
    ids_by_split: dict[str, set[str]] = {"train": set(), "validation": set()}
    all_ids: set[str] = set()
    pools: dict[str, dict[str, str]] = {"train": {}, "validation": {}}
    normalized_by_id: dict[str, str] = {}
    for row in records:
        split = row.get("split")
        if split not in {"train", "validation"}:
            raise ValueError("only the official train and development partitions are allowed")
        speaker = row.get("speaker_id")
        utterance_id = row.get("utterance_id")
        transcript = row.get("transcript")
        media_path = row.get("media_path")
        media_sha256 = row.get("media_sha256")
        if not all(
            isinstance(value, str) and value.strip()
            for value in (speaker, utterance_id, transcript, media_path, media_sha256)
        ):
            raise ValueError("each audio record needs source, transcript, and media metadata")
        if not re.fullmatch(r"[0-9a-f]{64}", media_sha256):
            raise ValueError("each audio asset must have a pinned SHA-256")
        if utterance_id in all_ids:
            raise ValueError("LibriSpeech utterance IDs must be unique across all splits")
        all_ids.add(utterance_id)
        ids_by_split[split].add(utterance_id)
        speakers_by_split[split].add(speaker)
        normalized = normalize_transcript(transcript)
        if not normalized:
            raise ValueError("transcripts must contain non-whitespace content")
        normalized_by_split[split].add(normalized)
        normalized_by_id[utterance_id] = normalized
        pools[split].setdefault(normalized, transcript)
    if not ids_by_split["train"] or not ids_by_split["validation"]:
        raise ValueError("both official LibriSpeech train and dev records are required")
    speaker_overlap = speakers_by_split["train"] & speakers_by_split["validation"]
    if speaker_overlap:
        raise ValueError(f"LibriSpeech speakers cross train/validation: {len(speaker_overlap)}")
    transcript_overlap = normalized_by_split["train"] & normalized_by_split["validation"]
    if transcript_overlap:
        raise ValueError(
            f"normalized transcript content crosses train/validation: {len(transcript_overlap)}"
        )

    candidates: dict[str, list[str]] = {}
    for split, pool in pools.items():
        candidates[split] = sorted(pool)
        if len(candidates[split]) < 4:
            raise ValueError(f"LibriSpeech {split} split needs at least four distinct transcripts")

    examples: dict[str, list[DecisionExample]] = {"train": [], "validation": []}
    chapter_ids: dict[str, set[str]] = {"train": set(), "validation": set()}
    for row in records:
        split = row["split"]
        normalized = normalized_by_id[row["utterance_id"]]
        candidate_transcripts = [value for value in candidates[split] if value != normalized]
        row_seed = int(
            hashlib.sha256(f"{seed}\0{row['utterance_id']}".encode()).hexdigest(), 16
        )
        distractors = random.Random(row_seed).sample(candidate_transcripts, 3)
        options = [str(row["transcript"]), *(pools[split][value] for value in distractors)]
        random.Random(row_seed ^ 0xA5A5A5A5).shuffle(options)
        target = str(row["transcript"])
        if target not in options or len(set(options)) != 4:
            raise AssertionError(
                "four-choice transcript candidates must be unique and include truth"
            )
        source_record_id = str(row["utterance_id"])
        speaker_id = str(row["speaker_id"])
        chapter_ids[split].add(f"{speaker_id}:{row['chapter_id']}")
        examples[split].append(
            DecisionExample(
                id=f"{SOURCE}:{source_record_id}",
                modality="audio",
                state="A spoken audiobook excerpt is provided as audio.",
                question="Which sentence matches the spoken audio?",
                options=options,
                target=target,
                media=[
                    MediaRef(
                        kind="audio",
                        path=str(row["media_path"]),
                        sha256=str(row["media_sha256"]),
                        license="CC-BY-4.0",
                    )
                ],
                source=SOURCE,
                source_revision=revision,
                source_record_id=source_record_id,
                split=split,
                task_group_id=f"speaker:{speaker_id}",
                source_target=target,
                provenance={
                    "license": "CC-BY-4.0",
                    "commercial_use": True,
                    "derivative_model_training_allowed": True,
                    "redistribution_allowed": True,
                    "media_redistribution_allowed": True,
                    "attribution": (
                        "Vassil Panayotov, Guoguo Chen, Daniel Povey, and Sanjeev Khudanpur; "
                        "LibriSpeech ASR corpus (ICASSP 2015)."
                    ),
                    "source_component": "OpenSLR SLR12",
                    "trust_status": "trusted",
                },
            )
        )
    for split in examples:
        examples[split].sort(key=lambda item: item.source_record_id)

    accounting: dict[str, Any] = {
        "seed": seed,
        "train_examples": len(examples["train"]),
        "validation_examples": len(examples["validation"]),
        "train_speakers": len(speakers_by_split["train"]),
        "validation_speakers": len(speakers_by_split["validation"]),
        "train_unique_chapters": len(chapter_ids["train"]),
        "validation_unique_chapters": len(chapter_ids["validation"]),
        "train_unique_audio_assets": len(ids_by_split["train"]),
        "validation_unique_audio_assets": len(ids_by_split["validation"]),
        "train_unique_transcripts": len(normalized_by_split["train"]),
        "validation_unique_transcripts": len(normalized_by_split["validation"]),
        "train_validation_speaker_overlap": 0,
        "train_validation_transcript_overlap": 0,
        "option_count": 4,
        "task_type": "spoken transcript selection",
        "source_partitions": {"train": "train-clean-100", "validation": "dev-clean"},
        "source_records_by_split": dict(Counter(str(row["split"]) for row in records)),
    }
    return examples["train"], examples["validation"], accounting


def build_librispeech_development_decisions(
    records: list[dict[str, Any]], *, revision: str, seed: int = 17
) -> tuple[list[DecisionExample], dict[str, Any]]:
    """Build a standalone, official-development-only LibriSpeech decision set.

    This is intended for a new development partition such as dev-other. It does
    not mix development examples with training rows to manufacture a split.
    Distractors are sampled only from distinct normalized transcripts within
    this development partition.
    """
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("revision must pin the OpenSLR checksum-manifest snapshot")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    if not records:
        raise ValueError("LibriSpeech development records must not be empty")

    seen_ids: set[str] = set()
    speakers: set[str] = set()
    transcripts: dict[str, str] = {}
    normalized_by_id: dict[str, str] = {}
    for row in records:
        if row.get("split") != "validation":
            raise ValueError("standalone LibriSpeech records must use the validation split")
        speaker = row.get("speaker_id")
        utterance_id = row.get("utterance_id")
        transcript = row.get("transcript")
        media_path = row.get("media_path")
        media_sha256 = row.get("media_sha256")
        if not all(
            isinstance(value, str) and value.strip()
            for value in (speaker, utterance_id, transcript, media_path, media_sha256)
        ):
            raise ValueError("each audio record needs source, transcript, and media metadata")
        if not re.fullmatch(r"[0-9a-f]{64}", media_sha256):
            raise ValueError("each audio asset must have a pinned SHA-256")
        if utterance_id in seen_ids:
            raise ValueError("LibriSpeech utterance IDs must be unique")
        seen_ids.add(utterance_id)
        speakers.add(speaker)
        normalized = normalize_transcript(transcript)
        if not normalized:
            raise ValueError("transcripts must contain non-whitespace content")
        normalized_by_id[utterance_id] = normalized
        transcripts.setdefault(normalized, transcript)

    candidates = sorted(transcripts)
    if len(candidates) < 4:
        raise ValueError("LibriSpeech development split needs four distinct transcripts")
    examples: list[DecisionExample] = []
    chapters: set[str] = set()
    for row in records:
        normalized = normalized_by_id[row["utterance_id"]]
        distractors = [value for value in candidates if value != normalized]
        row_seed = int(
            hashlib.sha256(f"{seed}\0{row['utterance_id']}".encode()).hexdigest(), 16
        )
        sampled = random.Random(row_seed).sample(distractors, 3)
        options = [str(row["transcript"]), *(transcripts[value] for value in sampled)]
        random.Random(row_seed ^ 0xA5A5A5A5).shuffle(options)
        speaker_id = str(row["speaker_id"])
        chapter_id = str(row["chapter_id"])
        chapters.add(f"{speaker_id}:{chapter_id}")
        examples.append(
            DecisionExample(
                id=f"{SOURCE}:{row['utterance_id']}",
                modality="audio",
                state="A spoken audiobook excerpt is provided as audio.",
                question="Which sentence matches the spoken audio?",
                options=options,
                target=str(row["transcript"]),
                media=[
                    MediaRef(
                        kind="audio",
                        path=str(row["media_path"]),
                        sha256=str(row["media_sha256"]),
                        license="CC-BY-4.0",
                    )
                ],
                source=SOURCE,
                source_revision=revision,
                source_record_id=str(row["utterance_id"]),
                split="validation",
                task_group_id=f"speaker:{speaker_id}",
                source_target=str(row["transcript"]),
                provenance={
                    "license": "CC-BY-4.0",
                    "commercial_use": True,
                    "derivative_model_training_allowed": True,
                    "redistribution_allowed": True,
                    "media_redistribution_allowed": True,
                    "attribution": (
                        "Vassil Panayotov, Daniel Povey, and Sanjeev Khudanpur; "
                        "LibriSpeech ASR corpus (ICASSP 2015)."
                    ),
                    "source_component": "OpenSLR SLR12",
                    "trust_status": "trusted",
                },
            )
        )
    accounting = {
        "examples": len(examples),
        "unique_audio_assets": len(seen_ids),
        "unique_speakers": len(speakers),
        "unique_chapters": len(chapters),
        "unique_normalized_transcripts": len(candidates),
        "normalized_transcript_overlap": 0,
        "seed": seed,
        "split": "official development only",
    }
    return examples, accounting
