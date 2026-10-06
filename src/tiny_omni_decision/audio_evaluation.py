from __future__ import annotations

import hashlib
import re
from difflib import SequenceMatcher
from typing import Any

_WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")
_SOUNDEX_GROUPS = {
    **{char: "1" for char in "bfpv"},
    **{char: "2" for char in "cgjkqsxz"},
    **{char: "3" for char in "dt"},
    "l": "4",
    **{char: "5" for char in "mn"},
    "r": "6",
}


def _soundex(word: str) -> str:
    letters = [char for char in word.lower() if char.isalpha()]
    if not letters:
        return ""
    first = letters[0]
    previous = _SOUNDEX_GROUPS.get(first, "")
    digits: list[str] = []
    for char in letters[1:]:
        code = _SOUNDEX_GROUPS.get(char, "")
        if code and code != previous:
            digits.append(code)
        previous = code
    return (first.upper() + "".join(digits) + "000")[:4]


def transcript_similarity(left: str, right: str) -> dict[str, float]:
    """Return reproducible lexical, length, and Soundex phonetic proxies."""
    left_words = _WORD.findall(left.lower())
    right_words = _WORD.findall(right.lower())
    if not left_words or not right_words:
        raise ValueError("transcripts must contain at least one English word")

    left_codes = [_soundex(word) for word in left_words]
    right_codes = [_soundex(word) for word in right_words]
    phonetic = SequenceMatcher(None, left_codes, right_codes, autojunk=False).ratio()
    left_set, right_set = set(left_words), set(right_words)
    lexical = len(left_set & right_set) / len(left_set | right_set)
    left_length = len(" ".join(left_words))
    right_length = len(" ".join(right_words))
    length = min(left_length, right_length) / max(left_length, right_length)
    return {"phonetic_soundex": phonetic, "lexical_jaccard": lexical, "length_ratio": length}


def hard_negative_score(left: str, right: str) -> tuple[float, dict[str, float]]:
    components = transcript_similarity(left, right)
    score = (
        0.50 * components["phonetic_soundex"]
        + 0.30 * components["lexical_jaccard"]
        + 0.20 * components["length_ratio"]
    )
    return score, components


def build_hard_negative_example(
    example: dict[str, Any], transcript_bank: list[dict[str, Any]], *, negative_count: int = 3
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if example.get("modality") != "audio" or example.get("source") != "openslr/LibriSpeech":
        raise ValueError("hard negatives are defined only for LibriSpeech audio examples")
    target = str(example["target"])
    target_id = str(example["source_record_id"])
    ranked: list[tuple[float, str, dict[str, Any], dict[str, float]]] = []
    seen_texts = {target.casefold()}
    for candidate in transcript_bank:
        text = str(candidate.get("target", "")).strip()
        candidate_id = str(candidate.get("source_record_id", ""))
        if not text or candidate_id == target_id or text.casefold() in seen_texts:
            continue
        seen_texts.add(text.casefold())
        score, components = hard_negative_score(target, text)
        ranked.append((score, candidate_id, candidate, components))
    if len(ranked) < negative_count:
        raise ValueError("transcript bank has too few distinct distractor candidates")
    ranked.sort(key=lambda item: (-item[0], item[1]))
    selected = ranked[:negative_count]
    options = [target, *(str(item[2]["target"]).strip() for item in selected)]
    if len({item.casefold() for item in options}) != len(options):
        raise ValueError("selected hard-negative options are not unique")
    seed = hashlib.sha256(f"librispeech-hard-v1:{example['id']}".encode()).digest()
    permutation = sorted(
        range(len(options)), key=lambda index: hashlib.sha256(seed + bytes([index])).digest()
    )
    shuffled = [options[index] for index in permutation]
    output = dict(example)
    output["id"] = f"{example['id']}:hard-neg-v1"
    output["options"] = shuffled
    output["target"] = target
    selection = [
        {
            "source_record_id": item[1],
            "transcript": str(item[2]["target"]).strip(),
            "weighted_similarity": item[0],
            **item[3],
        }
        for item in selected
    ]
    return output, selection
