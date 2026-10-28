"""Fail-closed row and frozen-feature contracts for corrected GRID Path A."""

from dataclasses import dataclass

import numpy as np

from scripts.lombard_readout_core import OPTIONS, targets_from_rows


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
    encoded = targets_from_rows(rows)
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
        train_targets=targets_from_rows(train),
        validation_targets=targets_from_rows(validation),
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
