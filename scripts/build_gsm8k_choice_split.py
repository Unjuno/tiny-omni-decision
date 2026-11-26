"""Build one train-only GSM8K-derived four-choice Decision development split.

The official test split is deliberately not read. Candidate distractors are
drawn only from distinct final-answer values in the generated training split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from tiny_omni_decision.dataset import sha256_file
from tiny_omni_decision.schema import DecisionExample, LicenseProvenance

SOURCE = "openai/grade-school-math"
SOURCE_REVISION = "3101c7d5072418e28b9008a6636bde82a006892c"
EXPECTED_TRAIN_SHA256 = "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465"
FINAL_ANSWER = re.compile(r"####\s*\$?([-+]?[\d,]+(?:\.\d+)?)\s*$")


def _normalized_question(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.findall(r"\w+", value, flags=re.UNICODE))


def _answer(row: dict[str, Any]) -> Decimal:
    match = FINAL_ANSWER.search(row["answer"].strip())
    if match is None:
        raise ValueError("GSM8K answer is missing the official final-answer marker")
    try:
        return Decimal(match.group(1).replace(",", ""))
    except InvalidOperation as exc:
        raise ValueError("invalid GSM8K final numeric answer") from exc


def _format_number(value: Decimal) -> str:
    if value == value.to_integral_value():
        return str(value.quantize(Decimal(1)))
    return format(value.normalize(), "f")


def _word_trigrams(text: str) -> set[tuple[str, ...]]:
    tokens = re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold())
    if len(tokens) < 3:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[index : index + 3]) for index in range(len(tokens) - 2)}


def _simhash(shingles: set[tuple[str, ...]]) -> int:
    totals = [0] * 64
    for shingle in shingles:
        payload = " ".join(shingle).encode("utf-8")
        bits = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")
        for bit in range(64):
            totals[bit] += 1 if bits & (1 << bit) else -1
    return sum(1 << bit for bit, total in enumerate(totals) if total >= 0)


def _group_indices(rows: list[dict[str, Any]]) -> list[list[int]]:
    """Group exact normalized duplicates and high-overlap near-duplicates."""
    parents = list(range(len(rows)))

    def find(value: int) -> int:
        while parents[value] != value:
            parents[value] = parents[parents[value]]
            value = parents[value]
        return value

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            parents[max(a, b)] = min(a, b)

    exact: dict[str, int] = {}
    shingle_sets = []
    hashes = []
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        normalized = _normalized_question(row["question"])
        if normalized in exact:
            union(index, exact[normalized])
        else:
            exact[normalized] = index
        shingles = _word_trigrams(row["question"])
        shingle_sets.append(shingles)
        code = _simhash(shingles)
        hashes.append(code)
        for band in range(4):
            buckets[(band, (code >> (band * 16)) & 0xFFFF)].append(index)

    pairs: set[tuple[int, int]] = set()
    for members in buckets.values():
        for offset, left in enumerate(members):
            for right in members[offset + 1 :]:
                pairs.add((left, right))
    for left, right in pairs:
        if (hashes[left] ^ hashes[right]).bit_count() > 6:
            continue
        a, b = shingle_sets[left], shingle_sets[right]
        if len(a & b) / max(1, len(a | b)) >= 0.75:
            union(left, right)

    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        groups[find(index)].append(index)
    return sorted(groups.values(), key=lambda members: min(members))


def _group_id(rows: list[dict[str, Any]], indices: list[int]) -> str:
    members = sorted(
        hashlib.sha256(_normalized_question(rows[i]["question"]).encode()).hexdigest()
        for i in indices
    )
    return hashlib.sha256("\n".join(members).encode()).hexdigest()


def _split_groups(
    groups: list[list[int]], rows: list[dict[str, Any]], seed: int, fraction: float
) -> set[int]:
    if not 0 < fraction < 1:
        raise ValueError("validation fraction must be strictly between zero and one")
    ranked = sorted(
        groups,
        key=lambda members: hashlib.sha256(
            f"seed={seed}:{_group_id(rows, members)}".encode()
        ).hexdigest(),
    )
    count = round(len(ranked) * fraction)
    validation = {index for members in ranked[:count] for index in members}
    if not validation or len(validation) == len(rows):
        raise ValueError("split produced an empty train or validation partition")
    return validation


def _distractors(gold: Decimal, bank: list[Decimal], key: str) -> list[Decimal]:
    candidates = sorted({value for value in bank if value != gold})
    if len(candidates) < 3:
        raise ValueError("training answer bank has fewer than three distinct negatives")
    candidates.sort(key=lambda value: (abs(value - gold) / max(Decimal(1), abs(gold)), value))
    near = candidates[: min(32, len(candidates))]
    selected = sorted(
        near,
        key=lambda value: hashlib.sha256(f"{key}:{value}".encode()).hexdigest(),
    )[:3]
    return selected


def _target_positions(
    groups: list[list[int]], rows: list[dict[str, Any]], seed: int
) -> dict[int, int]:
    ranked = sorted(
        groups,
        key=lambda members: hashlib.sha256(
            f"target-position-v1:{seed}:{_group_id(rows, members)}".encode()
        ).hexdigest(),
    )
    return {
        index: rank % 4
        for rank, members in enumerate(ranked)
        for index in members
    }


def _position_counts(examples: list[DecisionExample]) -> list[int]:
    counts = [0, 0, 0, 0]
    for example in examples:
        counts[example.options.index(example.target)] += 1
    return counts


def _jsonl_line(example: DecisionExample) -> str:
    # ensure_ascii escapes U+2028/U+2029 so they cannot split one JSON record.
    return json.dumps(example.model_dump(mode="json"), ensure_ascii=True, separators=(",", ":"))


def build_examples(
    rows: list[dict[str, Any]], *, seed: int = 17, validation_fraction: float = 0.10
) -> tuple[list[DecisionExample], list[DecisionExample], dict[str, Any]]:
    if not rows:
        raise ValueError("training source is empty")
    if any(set(row) != {"question", "answer"} for row in rows):
        raise ValueError("unexpected GSM8K record schema")
    if any(not isinstance(row["question"], str) or not row["question"].strip() for row in rows):
        raise ValueError("empty question in GSM8K train")
    normalized = [_normalized_question(row["question"]) for row in rows]
    record_ids = [hashlib.sha256(value.encode()).hexdigest() for value in normalized]
    answers = [_answer(row) for row in rows]
    duplicate_answers: dict[str, Decimal] = {}
    for question_id, answer in zip(record_ids, answers, strict=True):
        if question_id in duplicate_answers and duplicate_answers[question_id] != answer:
            raise ValueError("normalized duplicate question has conflicting numeric answers")
        duplicate_answers[question_id] = answer
    groups = _group_indices(rows)
    validation_indices = _split_groups(groups, rows, seed, validation_fraction)
    target_positions = _target_positions(groups, rows, seed)
    group_identity = {index: _group_id(rows, members) for members in groups for index in members}
    train_indices = [index for index in range(len(rows)) if index not in validation_indices]
    answer_bank = sorted({answers[index] for index in train_indices})
    if len(answer_bank) < 4:
        raise ValueError("training split does not provide enough distinct answer values")

    # Source provenance and data semantics are kept in the serialized examples.
    provenance = LicenseProvenance(
        license="MIT",
        commercial_use=True,
        derivative_model_training_allowed=True,
        redistribution_allowed=True,
        attribution="OpenAI, Grade School Math (GSM8K), 2021",
        source_component="GSM8K train.jsonl only; pinned source revision",
        trust_status="trusted",
    )

    def materialize(indices: list[int], split: str) -> list[DecisionExample]:
        examples = []
        for index in indices:
            gold = answers[index]
            rid = record_ids[index]
            distractors = _distractors(
                gold, answer_bank, f"distractor-selection-v1:seed={seed}:{rid}"
            )
            distractors.sort(
                key=lambda value: hashlib.sha256(
                    f"option-order-v1:seed={seed}:{rid}:{value}".encode()
                ).hexdigest()
            )
            position = target_positions[index]
            option_values = [*distractors]
            option_values.insert(position, gold)
            options = [_format_number(value) for value in option_values]
            target = _format_number(gold)
            examples.append(
                DecisionExample(
                    id=f"gsm8k-row-{index:05d}-{rid[:16]}",
                    modality="text",
                    state="",
                    question=rows[index]["question"],
                    options=options,
                    target=target,
                    source=SOURCE,
                    source_revision=SOURCE_REVISION,
                    source_record_id=group_identity[index],
                    split=split,
                    source_target=None,
                    provenance=provenance,
                )
            )
        return examples

    train = materialize(train_indices, "train")
    validation = materialize(sorted(validation_indices), "validation")
    train_ids = {example.source_record_id for example in train}
    validation_ids = {example.source_record_id for example in validation}
    if train_ids & validation_ids:
        raise ValueError("source record identity overlap across the split")
    # Verify the generated task is exactly one-positive, four-option MCQ.
    for example in train + validation:
        if len(example.options) != 4 or len(set(example.options)) != 4:
            raise ValueError("generated choice set is not four unique options")
        if example.target not in example.options:
            raise ValueError("target missing from generated options")
    train_position_counts = _position_counts(train)
    validation_position_counts = _position_counts(validation)
    for name, examples, counts in (
        ("train", train, train_position_counts),
        ("validation", validation, validation_position_counts),
    ):
        if len(examples) >= 100 and any(
            count < 0.15 * len(examples) or count > 0.35 * len(examples)
            for count in counts
        ):
            raise ValueError(f"target option positions are imbalanced in {name}")
    split_info = {
        "seed": seed,
        "validation_fraction": validation_fraction,
        "rows": len(rows),
        "groups": len(groups),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "train_groups": len(groups) - round(len(groups) * validation_fraction),
        "validation_groups": round(len(groups) * validation_fraction),
        "training_answer_bank_unique_values": len(answer_bank),
        "near_duplicate_group_count": sum(len(group) > 1 for group in groups),
        "train_target_option_position_counts": train_position_counts,
        "validation_target_option_position_counts": validation_position_counts,
        "grouping": (
            "exact normalized question plus word-trigram SimHash candidate pairs; "
            "Hamming <=6 and Jaccard >=0.75"
        ),
        "distractor_policy": (
            "three closest-relative numeric answers selected deterministically from "
            "a training-only bank of up to 32 nearest values; distractor selection, "
            "distractor ordering, and balanced target-position assignment use "
            "domain-separated hashes"
        ),
        "official_test_read": False,
    }
    return train, validation, split_info


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    return parser.parse_args()


def main() -> None:
    args = _args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {args.output_dir}")
    actual_hash = sha256_file(args.train_jsonl)
    if actual_hash != EXPECTED_TRAIN_SHA256:
        raise ValueError("GSM8K train payload hash mismatch")
    rows = [
        json.loads(line)
        for line in args.train_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    train, validation, split_info = build_examples(
        rows, seed=args.seed, validation_fraction=args.validation_fraction
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, examples in (("train", train), ("validation", validation)):
        path = args.output_dir / f"{name}.jsonl"
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            for example in examples:
                handle.write(_jsonl_line(example) + "\n")
    manifest = {
        "schema_version": 1,
        "source": SOURCE,
        "source_revision": SOURCE_REVISION,
        "train_source_sha256": actual_hash,
        "output_hashes": {
            name: sha256_file(args.output_dir / f"{name}.jsonl") for name in ("train", "validation")
        },
        "split": split_info,
        "claim_limit": (
            "English GSM8K-derived arithmetic four-choice development task; "
            "no source-writer separation and no product/general-text claim"
        ),
    }
    (args.output_dir / "split-manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
