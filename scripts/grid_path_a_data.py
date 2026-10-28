"""Fail-closed row and frozen-feature contracts for corrected GRID Path A."""

import hashlib
from dataclasses import dataclass

import numpy as np

from scripts.build_grid_direct_speaker_pilot import build_candidate_quartets
from scripts.lombard_grid_corpus import SLOTS

OPTIONS = {slot: list(table.values()) for slot, table in SLOTS.items()}


def _encode_targets(rows: list[dict]) -> np.ndarray:
    values = []
    for row in rows:
        targets = row.get("targets")
        if not isinstance(targets, dict) or set(targets) != set(OPTIONS):
            raise ValueError("all six native slot targets required")
        try:
            values.append([options.index(targets[slot]) for slot, options in OPTIONS.items()])
        except ValueError as error:
            raise ValueError("unknown native target; no vocabulary reduction") from error
    if not values:
        raise ValueError("nonempty native target rows required")
    return np.asarray(values, dtype=np.int64)


@dataclass(frozen=True)
class DirectSplit:
    train_ids: list[str]
    validation_ids: list[str]
    train_targets: np.ndarray
    validation_targets: np.ndarray


def _row_identity(rows: list[dict], name: str) -> tuple[list[str], set[str], set[tuple[str, ...]]]:
    if not rows:
        raise ValueError(f"nonempty {name} rows required")
    ids: list[str] = []
    speakers: set[str] = set()
    targets: set[tuple[str, ...]] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"invalid {name} row")
        speaker, code = row.get("speaker"), row.get("code")
        if (not isinstance(speaker, str) or not speaker.startswith("s")
                or not isinstance(code, str) or len(code) != 6 or not code.isalnum()):
            raise ValueError(f"invalid {name} speaker/code identity")
        ids.append(f"{speaker}_{code}")
        speakers.add(speaker)
        value = row.get("targets")
        if not isinstance(value, dict) or set(value) != set(OPTIONS):
            raise ValueError(f"invalid {name} native targets")
        targets.add(tuple(value[slot] for slot in OPTIONS))
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate {name} asset identity")
    encoded = _encode_targets(rows)
    if encoded.shape != (len(rows), 6):
        raise ValueError(f"invalid {name} target matrix")
    return ids, speakers, targets


def validate_direct_split(train: list[dict], validation: list[dict]) -> DirectSplit:
    """Validate the full speaker-disjoint direct-source split before fitting."""
    train_ids, train_speakers, train_targets = _row_identity(train, "train")
    validation_ids, validation_speakers, validation_targets = _row_identity(
        validation, "validation"
    )
    if set(train_ids) & set(validation_ids):
        raise ValueError("cross-split asset identity overlap")
    if train_speakers & validation_speakers:
        raise ValueError("cross-split speaker overlap")
    if train_targets & validation_targets:
        raise ValueError("cross-split normalized target overlap")
    return DirectSplit(
        train_ids=train_ids,
        validation_ids=validation_ids,
        train_targets=_encode_targets(train),
        validation_targets=_encode_targets(validation),
    )


def validate_feature_bank(
    feature_ids: list[str], expected_ids: list[str], bank: np.ndarray,
    *, width: int | tuple[int, ...], name: str,
) -> None:
    """Require feature identity/order, FP32 shape, and finite values to match."""
    if feature_ids != expected_ids:
        raise ValueError(f"{name} feature ID order differs from the frozen split")
    dims = (width,) if isinstance(width, int) else tuple(width)
    if (not isinstance(bank, np.ndarray) or bank.shape != (len(expected_ids), *dims)
            or bank.dtype != np.float32):
        raise ValueError(f"{name} feature bank shape or dtype differs")
    if not np.isfinite(bank).all():
        raise ValueError(f"{name} feature bank contains nonfinite values")


def select_balanced_quartets(
    rows: list[dict],
    *,
    split: str,
    speakers: list[str],
    excluded_codes: set[tuple[str, str]] | None = None,
    per_speaker: int = 8,
) -> list[dict]:
    """Select fixed, disjoint utterance quartets from one speaker-only split."""
    if split not in {"train", "validation"} or not speakers or per_speaker < 1:
        raise ValueError("valid split, speaker list and positive per-speaker count required")
    if len(set(speakers)) != len(speakers):
        raise ValueError("duplicate speaker in selection plan")
    excluded_codes = excluded_codes or set()
    grouped: dict[str, dict[str, dict]] = {speaker: {} for speaker in speakers}
    for row in rows:
        speaker, code = row.get("speaker"), row.get("code")
        if speaker not in grouped:
            continue
        if not isinstance(code, str) or (speaker, code) in excluded_codes:
            continue
        if code in grouped[speaker]:
            raise ValueError("duplicate code in split rows")
        grouped[speaker][code] = row

    selected: list[dict] = []
    for speaker in speakers:
        by_code = grouped[speaker]
        candidates, _ = build_candidate_quartets(speaker, set(by_code))
        ranked = sorted(
            candidates,
            key=lambda item: (
                hashlib.sha256(
                    f"grid-path-a-v1|{split}|{speaker}|{item['quartet_id']}".encode()
                ).hexdigest(),
                item["quartet_id"],
            ),
        )
        used: set[str] = set()
        count = 0
        for item in ranked:
            codes = item["codes"]
            if used.intersection(codes):
                continue
            selected.append(
                {
                    **item,
                    "split": split,
                    "targets": {code: by_code[code]["targets"] for code in codes},
                }
            )
            used.update(codes)
            count += 1
            if count == per_speaker:
                break
        if count != per_speaker:
            raise ValueError(f"speaker {speaker} has only {count}/{per_speaker} disjoint quartets")
    return selected
