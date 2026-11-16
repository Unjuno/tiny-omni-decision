"""Deterministic LibriSpeech speaker splits and transcript-choice examples."""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from tiny_omni_decision.audio_hard_negatives import (
    TranscriptRecord,
    find_hard_distractors,
    normalize_transcript,
)


def split_by_speaker(
    rows: Sequence[dict[str, Any]], *, seed: int = 17, validation_fraction: float = 0.2
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Split by speaker, then reject any residual asset or normalized-text overlap."""
    if not rows or not 0 < validation_fraction < 1:
        raise ValueError("rows must be nonempty and validation_fraction must be in (0, 1)")
    if any(not row.get("speaker_id") or not row.get("id") for row in rows):
        raise ValueError("every row needs a speaker_id and id")
    if len({str(row["id"]) for row in rows}) != len(rows):
        raise ValueError("duplicate LibriSpeech sample IDs")
    speakers = sorted({str(row["speaker_id"]) for row in rows})
    if len(speakers) < 2:
        raise ValueError("at least two speaker groups are required")
    shuffled = speakers.copy()
    random.Random(seed).shuffle(shuffled)
    validation_count = min(len(speakers) - 1, max(1, round(len(speakers) * validation_fraction)))
    validation_speakers = set(shuffled[:validation_count])
    validation = [dict(row) for row in rows if str(row["speaker_id"]) in validation_speakers]
    validation_content = {normalize_transcript(str(row["text"])) for row in validation}
    train = [
        dict(row)
        for row in rows
        if str(row["speaker_id"]) not in validation_speakers
        and normalize_transcript(str(row["text"])) not in validation_content
    ]
    if not train or not validation:
        raise ValueError("speaker split produced an empty partition")
    train_speakers = {str(row["speaker_id"]) for row in train}
    actual_validation_speakers = {str(row["speaker_id"]) for row in validation}
    train_assets = {str(row["audio_sha256"]) for row in train}
    validation_assets = {str(row["audio_sha256"]) for row in validation}
    train_content = {normalize_transcript(str(row["text"])) for row in train}
    if train_speakers & actual_validation_speakers:
        raise ValueError("train and validation speaker groups overlap")
    if train_assets & validation_assets:
        raise ValueError("train and validation audio identity overlap")
    if train_content & {normalize_transcript(str(row["text"])) for row in validation}:
        raise ValueError("train and validation normalized transcript overlap")
    return (
        train,
        validation,
        {
            "seed": seed,
            "validation_fraction": validation_fraction,
            "assigned_validation_speakers": sorted(validation_speakers),
            "train_speakers": len(train_speakers),
            "validation_speakers": len(actual_validation_speakers),
            "train_rows": len(train),
            "validation_rows": len(validation),
            "train_validation_speaker_overlap": 0,
            "train_validation_audio_sha256_overlap": 0,
            "train_validation_normalized_text_overlap": 0,
            "train_rows_removed_for_cross_split_text_duplicate": sum(
                str(row["speaker_id"]) not in validation_speakers
                and normalize_transcript(str(row["text"])) in validation_content
                for row in rows
            ),
        },
    )


def build_choice_examples(
    rows: Sequence[dict[str, Any]],
    records: Sequence[TranscriptRecord],
    *,
    seed: int = 17,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Create paired random/hard four-choice items using only one split's texts."""
    if len(rows) != len(records):
        raise ValueError("audio rows and phonemized transcript records must align")
    by_id = {record.sample_id: record for record in records}
    if len(by_id) != len(records) or {str(row["id"]) for row in rows} != set(by_id):
        raise ValueError("row and transcript IDs must be unique and identical")
    records_by_word_count: dict[int, list[TranscriptRecord]] = defaultdict(list)
    for record in records:
        records_by_word_count[len(record.words)].append(record)
    examples = []
    no_hard = 0
    no_random = 0
    for row in rows:
        target = by_id[str(row["id"])]
        # Cheap exact eligibility filters keep hard matching tractable at corpus scale.
        target_norm = normalize_transcript(target.text)
        target_chars = max(1, len(target_norm))
        target_words = set(target.words)
        pool = []
        for word_count in range(max(1, len(target.words) - 1), len(target.words) + 2):
            pool.extend(records_by_word_count.get(word_count, ()))
        eligible = [
            candidate
            for candidate in pool
            if candidate.sample_id != target.sample_id
            and candidate.speaker_id != target.speaker_id
            and candidate.chapter_id != target.chapter_id
            and normalize_transcript(candidate.text) != target_norm
            and min(target_chars, len(normalize_transcript(candidate.text)))
            / max(target_chars, len(normalize_transcript(candidate.text)))
            >= 0.65
            and bool(target_words.intersection(candidate.words))
        ]
        hard = find_hard_distractors(target, eligible, count=3)
        if len(hard) != 3:
            no_hard += 1
            continue
        stable_seed = int.from_bytes(
            hashlib.sha256(f"{seed}:{target.sample_id}".encode()).digest()[:8], "big"
        )
        rng = random.Random(stable_seed)
        hard_ids = {item.sample_id for item in hard}
        random_distractors = []
        random_ids: set[str] = set()
        # Rejection sampling preserves the legacy random-negative definition:
        # only source-group separation and duplicate-text exclusion are imposed.
        for _ in range(max(256, len(records) * 2)):
            candidate = rng.choice(records)
            if (
                candidate.sample_id == target.sample_id
                or candidate.sample_id in hard_ids
                or candidate.sample_id in random_ids
                or candidate.speaker_id == target.speaker_id
                or candidate.chapter_id == target.chapter_id
                or normalize_transcript(candidate.text) == target_norm
            ):
                continue
            random_distractors.append(candidate)
            random_ids.add(candidate.sample_id)
            if len(random_distractors) == 3:
                break
        if len(random_distractors) != 3:
            no_random += 1
            continue
        target_position = rng.randrange(4)
        options = {}
        distractor_ids = {}
        for condition, distractors in (
            ("random_distractors", random_distractors),
            ("hard_distractors", hard),
        ):
            candidate_texts = [item.text for item in distractors]
            option_texts = candidate_texts.copy()
            option_texts.insert(target_position, target.text)
            if len(set(option_texts)) != 4:
                raise ValueError(f"duplicate choice text for {target.sample_id}/{condition}")
            options[condition] = option_texts
            distractor_ids[condition] = [item.sample_id for item in distractors]
        examples.append(
            {
                "sample_id": target.sample_id,
                "speaker_id": target.speaker_id,
                "chapter_id": target.chapter_id,
                "audio_path": str(row["audio_path"]),
                "audio_sha256": str(row["audio_sha256"]),
                "target_index": target_position,
                "target_text": target.text,
                "options_by_condition": options,
                "distractor_ids_by_condition": distractor_ids,
                "hard_distractors": [
                    {
                        "sample_id": item.sample_id,
                        "phoneme_similarity": item.phoneme_similarity,
                        "vocabulary_jaccard": item.vocabulary_jaccard,
                        "word_count_delta": item.word_count_delta,
                        "character_length_ratio": item.character_length_ratio,
                    }
                    for item in hard
                ],
            }
        )
    return examples, {
        "eligible_targets": len(rows),
        "examples_with_paired_conditions": len(examples),
        "excluded_insufficient_hard_negatives": no_hard,
        "excluded_insufficient_random_negatives": no_random,
    }
