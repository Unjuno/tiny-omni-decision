"""Audit a hash-pinned BANKING77 train CSV without opening its public test split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

EXPECTED_SHA256 = "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b"
DATASET_CARD_SHA256 = "f3786a5327fd6533c7c0ff3ff654e20ee3d1261d2fa41667381f59a2e42364e1"
LICENSE_SHA256 = "7e7170e3cebf88a9f60c7b8421418323c09304da1af4d5e90f4da1dc1c8a2661"
EXPECTED_TRAIN_ROWS = 10_003
EXPECTED_INTENTS = 77

_POSSIBLE_PII_PATTERNS = {
    "email": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
    "iban": re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]){11,30}\b", re.IGNORECASE),
    "long_digit_run": re.compile(r"(?<!\w)\d{8,}(?!\w)"),
    "sort_code_like": re.compile(r"\b\d{2}[- ]\d{2}[- ]\d{2}\b"),
    "phone_like": re.compile(r"(?<!\w)\+?\d[\d ()./-]{6,}\d(?!\w)"),
}
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def normalize_text(text: str) -> str:
    """Normalize Unicode, case and punctuation for conservative duplicate checks."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join("".join(ch if ch.isalnum() else " " for ch in normalized).split())


def audit_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    """Return aggregate quality signals; never include utterance text in the result."""
    if not rows:
        raise ValueError("train CSV is empty")

    duplicates: dict[str, list[tuple[int, str]]] = defaultdict(list)
    labels: Counter[str] = Counter()
    lengths: list[int] = []
    possible_pii_rows: dict[int, list[str]] = {}
    invalid_rows: list[int] = []

    for row_number, row in enumerate(rows):
        text = row.get("text", "")
        category = row.get("category", "")
        if not text.strip() or not category.strip():
            invalid_rows.append(row_number)
            continue
        labels[category] += 1
        lengths.append(len(text))
        duplicates[normalize_text(text)].append((row_number, category))
        matches = [name for name, pattern in _POSSIBLE_PII_PATTERNS.items() if pattern.search(text)]
        if matches:
            possible_pii_rows[row_number] = matches

    repeated = [members for members in duplicates.values() if len(members) > 1]
    conflicts = [members for members in repeated if len({label for _, label in members}) > 1]
    near_pairs = near_duplicate_candidates(rows)
    class_counts = dict(sorted(labels.items()))
    if invalid_rows:
        raise ValueError(f"invalid or empty text/category in {len(invalid_rows)} train rows")
    if len(class_counts) != EXPECTED_INTENTS:
        raise ValueError(f"expected {EXPECTED_INTENTS} intents, found {len(class_counts)}")

    return {
        "row_count": len(rows),
        "intent_count": len(class_counts),
        "intent_counts": class_counts,
        "intent_count_min": min(class_counts.values()),
        "intent_count_median": statistics.median(class_counts.values()),
        "intent_count_max": max(class_counts.values()),
        "text_char_length_median": statistics.median(lengths),
        "normalized_duplicate_groups": len(repeated),
        "normalized_duplicate_excess_rows": sum(len(members) - 1 for members in repeated),
        "normalized_duplicate_conflict_groups": len(conflicts),
        "normalized_duplicate_conflict_row_ids": [
            [member_row for member_row, _ in members] for members in conflicts
        ],
        "near_duplicate_screen": {
            "method": (
                "token-set Jaccard >= 0.80; candidate pairs share a word bigram "
                "occurring in <=80 rows"
            ),
            "candidate_pairs": len(near_pairs),
            "cross_intent_candidate_pairs": sum(
                pair["left_category"] != pair["right_category"] for pair in near_pairs
            ),
            "candidate_pairs_by_row_id_and_label": near_pairs,
            "limitations": [
                "candidate generation is heuristic and not exhaustive",
                "high lexical overlap does not by itself prove duplicate meaning or wrong labels",
                "pairs are flags for human review and must not be silently filtered",
            ],
        },
        "possible_pii_pattern_row_count": len(possible_pii_rows),
        "possible_pii_pattern_counts": dict(
            sorted(Counter(name for names in possible_pii_rows.values() for name in names).items())
        ),
        "possible_pii_pattern_row_ids": {
            str(row_number): names for row_number, names in sorted(possible_pii_rows.items())
        },
        "limitations": [
            "pattern checks are heuristic and do not establish absence of personal information",
            "no stable user, session, annotator, or source-event identifier is available in CSV",
            "near-duplicate/paraphrase clusters and label correctness require separate review",
        ],
    }


def near_duplicate_candidates(
    rows: list[dict[str, str]], *, threshold: float = 0.80, max_bigram_frequency: int = 80
) -> list[dict[str, Any]]:
    """Find review candidates via rare word bigrams and token-set Jaccard similarity."""
    token_sets: list[set[str]] = []
    bigram_sets: list[set[tuple[str, str]]] = []
    bigram_frequency: Counter[tuple[str, str]] = Counter()
    for row in rows:
        tokens = _TOKEN_PATTERN.findall(normalize_text(row.get("text", "")))
        token_set = set(tokens)
        bigrams = set(zip(tokens, tokens[1:], strict=False))
        token_sets.append(token_set)
        bigram_sets.append(bigrams)
        bigram_frequency.update(bigrams)

    postings: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row_id, bigrams in enumerate(bigram_sets):
        for bigram in bigrams:
            if bigram_frequency[bigram] <= max_bigram_frequency:
                postings[bigram].append(row_id)

    candidate_pairs: set[tuple[int, int]] = set()
    for row_ids in postings.values():
        candidate_pairs.update(itertools.combinations(row_ids, 2))

    results: list[dict[str, Any]] = []
    for left_id, right_id in sorted(candidate_pairs):
        left_tokens, right_tokens = token_sets[left_id], token_sets[right_id]
        union = left_tokens | right_tokens
        if not union:
            continue
        similarity = len(left_tokens & right_tokens) / len(union)
        if similarity >= threshold:
            results.append(
                {
                    "left_row_id": left_id,
                    "left_category": rows[left_id]["category"],
                    "right_row_id": right_id,
                    "right_category": rows[right_id]["category"],
                    "token_set_jaccard": round(similarity, 6),
                }
            )
    return results


def audit_file(path: Path, expected_sha256: str = EXPECTED_SHA256) -> dict[str, Any]:
    payload = path.read_bytes()
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256.lower() != expected_sha256.lower():
        raise ValueError(f"SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}")

    text = payload.decode("utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(text, newline="")))
    if not rows or set(rows[0]) != {"text", "category"}:
        raise ValueError("expected exactly the BANKING77 train fields: text, category")
    if len(rows) != EXPECTED_TRAIN_ROWS:
        raise ValueError(f"expected {EXPECTED_TRAIN_ROWS} train rows, found {len(rows)}")

    return {
        "source": "PolyAI-LDN/task-specific-datasets/banking_data/train.csv",
        "source_revision": "9d081458ff52e53cf7e848f414e6e9344e4e6696",
        "license_revision": "57ec275d8078af65b7731c2a98be812d844a6d6b",
        "dataset_card_revision": "1846984189f2bc0ac51c6fc3a600fa9e1714e0c6",
        "license": "CC-BY-4.0 (publisher dataset card and repository license)",
        "dataset_card_sha256": DATASET_CARD_SHA256,
        "license_sha256": LICENSE_SHA256,
        "input_bytes": len(payload),
        "input_sha256": actual_sha256,
        "audit": audit_rows(rows),
        "access_boundary": {
            "train_file_opened": True,
            "publisher_test_file_opened": False,
            "model_or_predictions_loaded": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train_csv", type=Path)
    parser.add_argument("--expected-sha256", default=EXPECTED_SHA256)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit_file(args.train_csv, args.expected_sha256)
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
