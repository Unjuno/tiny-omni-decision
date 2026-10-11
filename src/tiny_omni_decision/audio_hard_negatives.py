"""Deterministic transcript and distractor helpers for LibriSpeech diagnostics."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from difflib import SequenceMatcher


_WORD_RE = re.compile(r"[A-Z]+(?:'[A-Z]+)?")
_ALT_PRON_RE = re.compile(r"\(\d+\)$")


@dataclass(frozen=True)
class TranscriptRecord:
    sample_id: str
    speaker_id: str
    chapter_id: str
    text: str
    words: tuple[str, ...]
    phonemes: tuple[str, ...]


@dataclass(frozen=True)
class DistractorMatch:
    sample_id: str
    text: str
    phoneme_similarity: float
    vocabulary_jaccard: float
    word_count_delta: int
    character_length_ratio: float


def normalize_transcript(text: str) -> str:
    """Return the uppercase word sequence used for identity and phonetic lookup."""
    return " ".join(_WORD_RE.findall(text.upper()))


def parse_cmudict(text: str) -> dict[str, tuple[str, ...]]:
    """Read a pinned CMUdict file, retaining the first pronunciation per word."""
    pronunciations: dict[str, tuple[str, ...]] = {}
    for line in text.splitlines():
        if not line or line.startswith(";;;"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        raw_word, raw_phones = parts
        word = _ALT_PRON_RE.sub("", raw_word).upper()
        pronunciations.setdefault(
            word,
            tuple(re.sub(r"\d", "", phone) for phone in raw_phones.split()),
        )
    return pronunciations


def transcript_phonemes(
    text: str, pronunciations: dict[str, tuple[str, ...]]
) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    """Return normalized words and their CMU phonemes, or None if any word is OOV."""
    words = tuple(_WORD_RE.findall(text.upper()))
    if not words:
        return None
    if any(word not in pronunciations for word in words):
        return None
    phones: list[str] = []
    for word in words:
        if phones:
            phones.append("|")
        phones.extend(pronunciations[word])
    return words, tuple(phones)


def phoneme_sequence_similarity(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    """Order-aware normalized sequence similarity over CMU phoneme tokens."""
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right, autojunk=False).ratio()


def vocabulary_jaccard(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 0.0


def find_hard_distractors(
    target: TranscriptRecord,
    candidates: Iterable[TranscriptRecord],
    *,
    count: int = 3,
) -> list[DistractorMatch]:
    """Rank different-speaker/chapter transcripts close in phones, words, and length.

    A hard candidate must be from another speaker and chapter, differ by at most
    one word, have at least one shared vocabulary item, and have character length
    within a 0.65 ratio. Among those candidates, the top matches are ordered by
    phoneme-sequence similarity, vocabulary overlap, then length similarity.
    """
    if count < 1:
        raise ValueError("count must be positive")
    target_norm = normalize_transcript(target.text)
    target_char_len = max(1, len(target_norm))
    matches: list[DistractorMatch] = []
    for candidate in candidates:
        if candidate.sample_id == target.sample_id:
            continue
        if candidate.speaker_id == target.speaker_id:
            continue
        if candidate.chapter_id == target.chapter_id:
            continue
        candidate_norm = normalize_transcript(candidate.text)
        if not candidate_norm or candidate_norm == target_norm:
            continue
        word_delta = abs(len(target.words) - len(candidate.words))
        if word_delta > 1:
            continue
        char_ratio = min(target_char_len, len(candidate_norm)) / max(
            target_char_len, len(candidate_norm)
        )
        if char_ratio < 0.65:
            continue
        jaccard = vocabulary_jaccard(target.words, candidate.words)
        if jaccard == 0.0:
            continue
        matches.append(
            DistractorMatch(
                sample_id=candidate.sample_id,
                text=candidate.text,
                phoneme_similarity=phoneme_sequence_similarity(
                    target.phonemes, candidate.phonemes
                ),
                vocabulary_jaccard=jaccard,
                word_count_delta=word_delta,
                character_length_ratio=char_ratio,
            )
        )
    matches.sort(
        key=lambda match: (
            -match.phoneme_similarity,
            -match.vocabulary_jaccard,
            match.word_count_delta,
            -match.character_length_ratio,
            match.sample_id,
        )
    )
    return matches[:count]


def make_transcript_record(
    sample_id: str,
    speaker_id: str,
    chapter_id: str,
    text: str,
    pronunciations: dict[str, tuple[str, ...]],
) -> TranscriptRecord | None:
    encoded = transcript_phonemes(text, pronunciations)
    if encoded is None:
        return None
    words, phones = encoded
    return TranscriptRecord(
        sample_id=sample_id,
        speaker_id=speaker_id,
        chapter_id=chapter_id,
        text=" ".join(words),
        words=words,
        phonemes=phones,
    )


def multiset_dice_similarity(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    """Cheap phoneme-bigram similarity used to audit selected-vs-random hardness."""
    def bigrams(sequence: tuple[str, ...]) -> Counter[tuple[str, str]]:
        return Counter(zip(sequence, sequence[1:], strict=False))

    left_counts = bigrams(left)
    right_counts = bigrams(right)
    denominator = sum(left_counts.values()) + sum(right_counts.values())
    if not denominator:
        return 0.0
    overlap = sum((left_counts & right_counts).values())
    return 2.0 * overlap / denominator
