from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import defaultdict
from typing import Any

from .schema import DecisionExample

_PARTITION_RE = re.compile(rb'"partition"\s*:\s*"(train|dev|test)"')
_SOURCE = "alexa/massive"
_LICENSE = {
    "license": "CC-BY-4.0",
    "commercial_use": True,
    "derivative_model_training_allowed": True,
    "redistribution_allowed": True,
    "media_redistribution_allowed": None,
    "attribution": "Amazon Science; MASSIVE dataset, CC BY 4.0.",
    "source_component": "MASSIVE en-US",
    "trust_status": "trusted",
}


def partition_from_raw_line(raw: bytes) -> str:
    """Read only the split marker, allowing reserved test rows to be skipped untouched."""
    match = _PARTITION_RE.search(raw)
    if match is None:
        raise ValueError("MASSIVE record has no recognized partition marker")
    return match.group(1).decode("ascii")


def normalize_utterance(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def build_massive_splits(
    records: list[dict[str, Any]], *, revision: str, seed: int = 17
) -> tuple[list[DecisionExample], list[DecisionExample], dict[str, Any]]:
    """Create full-taxonomy text decisions with normalized utterances group-split."""
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("revision must be a pinned 40-character commit")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_ids: set[str] = set()
    labels: set[str] = set()
    for row in records:
        partition = row.get("partition")
        if partition not in {"train", "dev"}:
            raise ValueError("only train and dev records may be passed to the normalizer")
        if row.get("locale") != "en-US":
            raise ValueError("only the en-US locale may be used by this adapter")
        utterance = row.get("utt")
        intent = row.get("intent")
        record_id = row.get("id")
        if not all(isinstance(value, str) and value.strip() for value in (utterance, intent)):
            raise ValueError("each MASSIVE row needs a nonempty utterance and intent")
        if record_id is None or str(record_id) in seen_ids:
            raise ValueError("MASSIVE source record IDs must be present and unique")
        seen_ids.add(str(record_id))
        normalized = normalize_utterance(utterance)
        if not normalized:
            raise ValueError("normalized utterances must be nonempty")
        copied = dict(row)
        copied["_normalized_utterance"] = normalized
        groups[normalized].append(copied)
        labels.add(intent)
    if not labels or not any(row.get("partition") == "train" for row in records):
        raise ValueError("MASSIVE train records are required to define the answer taxonomy")
    if len(labels) > 62:
        raise ValueError("the decision runtime supports at most 62 answer choices")

    validation_groups = {
        key for key, rows in groups.items() if any(row["partition"] == "dev" for row in rows)
    }
    cross_partition_groups = {
        key
        for key, rows in groups.items()
        if {row["partition"] for row in rows} == {"train", "dev"}
    }
    train_intents = {
        row["intent"] for row in records if row["partition"] == "train"
    }
    validation_intents = {
        row["intent"] for key in validation_groups for row in groups[key]
    }
    promoted_groups: list[str] = []
    for intent in sorted(train_intents - validation_intents):
        candidates = [
            key
            for key, rows in groups.items()
            if key not in validation_groups
            and any(row["partition"] == "train" and row["intent"] == intent for row in rows)
        ]
        if not candidates:
            raise ValueError(f"cannot ensure validation coverage for intent {intent!r}")
        chosen = min(
            candidates,
            key=lambda value: hashlib.sha256(f"{seed}\0{value}".encode()).hexdigest(),
        )
        validation_groups.add(chosen)
        promoted_groups.append(chosen)

    options = sorted(train_intents)
    if set(options) != labels:
        raise ValueError("dev contains an intent absent from the official train partition")

    def convert(normalized: str, row: dict[str, Any], split: str) -> DecisionExample:
        intent = row["intent"]
        if intent not in options:
            raise ValueError(f"intent {intent!r} is absent from the training taxonomy")
        source_record_id = f"en-US:{row['id']}"
        group_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        return DecisionExample(
            id=f"massive:{source_record_id}",
            modality="text",
            state=row["utt"],
            question="Which intent best matches the utterance?",
            options=options,
            target=intent,
            source=_SOURCE,
            source_revision=revision,
            source_record_id=source_record_id,
            split=split,
            task_group_id=f"utterance:{group_hash}",
            source_target=intent,
            provenance=_LICENSE,
        )

    training: list[DecisionExample] = []
    validation: list[DecisionExample] = []
    for normalized, rows in groups.items():
        split = "validation" if normalized in validation_groups else "train"
        target = validation if split == "validation" else training
        target.extend(convert(normalized, row, split) for row in rows)
    training.sort(key=lambda item: item.source_record_id)
    validation.sort(key=lambda item: item.source_record_id)
    train_groups = {item.task_group_id for item in training}
    validation_group_ids = {item.task_group_id for item in validation}
    if train_groups & validation_group_ids:
        raise AssertionError("normalized utterance group leaked across MASSIVE splits")
    if {item.target for item in validation} < set(options):
        raise AssertionError("validation must contain every train-defined intent")

    accounting = {
        "seed": seed,
        "option_count": len(options),
        "options_sha256": hashlib.sha256("\n".join(options).encode()).hexdigest(),
        "train_examples": len(training),
        "validation_examples": len(validation),
        "train_unique_utterances": len(train_groups),
        "validation_unique_utterances": len(validation_group_ids),
        "train_validation_normalized_overlap": 0,
        "official_train_dev_utterance_groups_overlapping": len(cross_partition_groups),
        "official_train_rows_grouped_with_dev": sum(
            1
            for key in cross_partition_groups
            for row in groups[key]
            if row["partition"] == "train"
        ),
        "official_train_groups_moved_to_validation": len(promoted_groups),
        "official_train_rows_moved_to_validation": sum(
            1
            for key in promoted_groups
            for row in groups[key]
            if row["partition"] == "train"
        ),
        "validation_intents": len({item.target for item in validation}),
        "train_multi_intent_utterance_groups": sum(
            len({row["intent"] for row in rows}) > 1
            for key, rows in groups.items()
            if key not in validation_groups
        ),
        "validation_multi_intent_utterance_groups": sum(
            len({row["intent"] for row in groups[key]}) > 1 for key in validation_groups
        ),
    }
    return training, validation, accounting
