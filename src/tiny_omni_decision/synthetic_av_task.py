"""Deterministic manifest construction for a synthetic AV order task."""

from __future__ import annotations

import hashlib
import random
from collections import Counter, defaultdict
from typing import Any

SYMBOL_COUNT = 4
SEQUENCE_LENGTH = 8
PAIRINGS = (("A", "A"), ("B", "B"), ("A", "B"), ("B", "A"))
QUERY_POLARITIES = ("same_order", "different_order")


def _sequence_id(symbols: tuple[int, ...]) -> str:
    payload = ",".join(map(str, symbols)).encode("ascii")
    return "seq-" + hashlib.sha256(payload).hexdigest()[:16]


def _make_sequence(rng: random.Random) -> tuple[int, ...]:
    symbols = [index for index in range(SYMBOL_COUNT) for _ in range(2)]
    rng.shuffle(symbols)
    return tuple(symbols)


def _make_pair(rng: random.Random) -> tuple[tuple[int, ...], tuple[int, ...]]:
    first = _make_sequence(rng)
    second = list(first)
    for _ in range(32):
        rng.shuffle(second)
        if tuple(second) != first:
            return first, tuple(second)
    raise RuntimeError("could not generate a distinct permutation")


def build_manifest(
    train_groups: int, validation_groups: int, seed: int = 17
) -> dict[str, Any]:
    """Build a group-disjoint, query-balanced manifest with exact order labels."""
    if train_groups <= 0 or validation_groups <= 0:
        raise ValueError("train_groups and validation_groups must both be positive")
    if seed < 0:
        raise ValueError("seed must be non-negative")

    rng = random.Random(seed)
    examples: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    used_sequences: set[str] = set()
    for split, count in (("train", train_groups), ("validation", validation_groups)):
        for split_index in range(count):
            group_id = f"{split}-g{split_index:05d}"
            for _ in range(100):
                seq_a, seq_b = _make_pair(rng)
                candidate_ids = {_sequence_id(seq_a), _sequence_id(seq_b)}
                if len(candidate_ids) == 2 and not candidate_ids.intersection(used_sequences):
                    used_sequences.update(candidate_ids)
                    break
            else:
                raise RuntimeError("could not generate unique sequence identities")
            sequences = {"A": seq_a, "B": seq_b}
            groups.append(
                {
                    "group_id": group_id,
                    "split": split,
                    "sequence_ids": {key: _sequence_id(value) for key, value in sequences.items()},
                }
            )
            for audio_key, video_key in PAIRINGS:
                audio_symbols = sequences[audio_key]
                video_symbols = sequences[video_key]
                relation = "same_order" if audio_symbols == video_symbols else "different_order"
                pair_id = f"{group_id}-{audio_key}{video_key}"
                for polarity in QUERY_POLARITIES:
                    target = int(relation == polarity)
                    sample_id = f"{pair_id}-{polarity}"
                    examples.append(
                        {
                            "sample_id": sample_id,
                            "pair_id": pair_id,
                            "pairing": audio_key + video_key,
                            "group_id": group_id,
                            "split": split,
                            "audio_sequence_id": _sequence_id(audio_symbols),
                            "video_sequence_id": _sequence_id(video_symbols),
                            "audio_symbols": list(audio_symbols),
                            "video_symbols": list(video_symbols),
                            "query_polarity": polarity,
                            "target": target,
                            "relation": relation,
                        }
                    )

    manifest = {
        "schema_version": 1,
        "seed": seed,
        "sequence_length": SEQUENCE_LENGTH,
        "symbol_count": SYMBOL_COUNT,
        "groups": groups,
        "examples": examples,
    }
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: dict[str, Any]) -> None:
    """Reject malformed labels, duplicate IDs, and cross-split group/sequence use."""
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported synthetic AV manifest schema")
    groups = manifest.get("groups")
    examples = manifest.get("examples")
    if not isinstance(groups, list) or not isinstance(examples, list):
        raise ValueError("manifest groups and examples must be lists")
    group_split: dict[str, str] = {}
    group_sequences: dict[str, set[str]] = {}
    sequence_split: dict[str, str] = {}
    for group in groups:
        group_id, split = group.get("group_id"), group.get("split")
        if split not in {"train", "validation"} or not isinstance(group_id, str):
            raise ValueError("invalid group identity or split")
        if group_id in group_split:
            raise ValueError(f"duplicate group: {group_id}")
        group_split[group_id] = split
        sequence_ids = group.get("sequence_ids")
        if not isinstance(sequence_ids, dict) or set(sequence_ids) != {"A", "B"}:
            raise ValueError(f"group {group_id} must define sequence A and B")
        for sequence_id in sequence_ids.values():
            prior = sequence_split.setdefault(sequence_id, split)
            if prior != split:
                raise ValueError(f"sequence crosses splits: {sequence_id}")
        if len(set(sequence_ids.values())) != 2:
            raise ValueError(f"group {group_id} must use two distinct sequences")
        group_sequences[group_id] = set(sequence_ids.values())

    sample_ids: set[str] = set()
    rows_by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    audio_marginals: dict[tuple[str, str], Counter[int]] = defaultdict(Counter)
    video_marginals: dict[tuple[str, str], Counter[int]] = defaultdict(Counter)
    for row in examples:
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or sample_id in sample_ids:
            raise ValueError("sample IDs must be unique strings")
        sample_ids.add(sample_id)
        group_id, split = row.get("group_id"), row.get("split")
        if group_split.get(group_id) != split:
            raise ValueError(f"example has missing or mismatched group: {sample_id}")
        audio = row.get("audio_symbols")
        video = row.get("video_symbols")
        if not (
            isinstance(audio, list)
            and isinstance(video, list)
            and len(audio) == SEQUENCE_LENGTH
            and len(video) == SEQUENCE_LENGTH
            and all(isinstance(x, int) and 0 <= x < SYMBOL_COUNT for x in audio + video)
        ):
            raise ValueError(f"invalid symbol sequence: {sample_id}")
        relation = "same_order" if audio == video else "different_order"
        if Counter(audio) != Counter(video) or row.get("relation") != relation:
            raise ValueError(f"invalid relation or unmatched symbol multiset: {sample_id}")
        audio_id = _sequence_id(tuple(audio))
        video_id = _sequence_id(tuple(video))
        if row.get("audio_sequence_id") != audio_id or row.get("video_sequence_id") != video_id:
            raise ValueError(f"sequence identity does not match normalized content: {sample_id}")
        if not {audio_id, video_id}.issubset(group_sequences[group_id]):
            raise ValueError(f"example sequence is outside its group: {sample_id}")
        polarity = row.get("query_polarity")
        if polarity not in QUERY_POLARITIES or row.get("target") != int(relation == polarity):
            raise ValueError(f"invalid query target: {sample_id}")
        rows_by_pair[row["pair_id"]].append(row)
        audio_marginals[(row["audio_sequence_id"], polarity)][row["target"]] += 1
        video_marginals[(row["video_sequence_id"], polarity)][row["target"]] += 1

    rows_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in examples:
        rows_by_group[row["group_id"]].append(row)
    if set(rows_by_group) != set(group_split) or any(
        len(group_rows) != 8 for group_rows in rows_by_group.values()
    ):
        raise ValueError("every group must have exactly eight examples")

    for pair in rows_by_pair.values():
        if len(pair) != 2 or {row["query_polarity"] for row in pair} != set(QUERY_POLARITIES):
            raise ValueError("each input pair must include both query polarities exactly once")
        if {row["target"] for row in pair} != {0, 1}:
            raise ValueError("the two query polarities must have complementary targets")
    for group_id, group_rows in rows_by_group.items():
        expected = {f"{audio}{video}" for audio, video in PAIRINGS}
        if len(group_rows) != 8 or {row["pairing"] for row in group_rows} != expected:
            raise ValueError(
                f"group {group_id} must contain each of four pairings and two polarities"
            )
    marginals = (*audio_marginals.values(), *video_marginals.values())
    if any(counts != Counter({0: 1, 1: 1}) for counts in marginals):
        raise ValueError("audio-only and video-only targets must balance within each query")

