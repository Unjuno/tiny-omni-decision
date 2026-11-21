"""Deterministic participant-disjoint split for the R2-M audio/vibration pilot."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable

SEED = 17
SPLIT_SIZES = {"train": 8, "calibration": 2, "validation": 2, "sealed_audit": 2}
NO_ACTIVITY_PARTICIPANT = 999
SPLIT_ALGORITHM = (
    "sort participant IDs by SHA256(r2m-zenodo-audio-vibration-match-v1|seed=17|"
    "participant=<decimal_id>), tie-break numeric ID; allocate 8/2/2/2 to "
    "train/calibration/validation/sealed_audit; exclude participant 999 no-activity sentinel"
)
_DIGITS = re.compile(r"^[0-9]+$")


def freeze_audio_vibration_participant_split(
    participant_ids: Iterable[int | str],
) -> dict[str, object]:
    """Freeze a fixed 8/2/2/2 split from exactly 14 distinct participant IDs.

    This function accepts identifiers only. It does not inspect labels, media,
    trial signals, or the sealed audit. The publisher's synthetic no-activity
    participant sentinel is removed before enforcing the 14-person inventory.
    """
    try:
        raw_ids = list(participant_ids)
    except TypeError as exc:
        raise ValueError("participant IDs must be an iterable of integer IDs") from exc
    normalized: list[int] = []
    for value in raw_ids:
        if isinstance(value, bool):
            raise ValueError("boolean participant IDs are invalid")
        if isinstance(value, int):
            participant_id = value
        elif isinstance(value, str) and _DIGITS.fullmatch(value.strip()):
            participant_id = int(value.strip())
        else:
            raise ValueError("participant IDs must be nonnegative decimal integers")
        if participant_id < 0:
            raise ValueError("participant IDs must be nonnegative")
        if participant_id != NO_ACTIVITY_PARTICIPANT:
            normalized.append(participant_id)

    if len(set(normalized)) != len(normalized):
        raise ValueError("duplicate participant IDs after decimal normalization")
    if len(normalized) != sum(SPLIT_SIZES.values()):
        raise ValueError("exactly 14 real participants are required; no split fallback")

    ordered = sorted(
        normalized,
        key=lambda participant_id: (
            hashlib.sha256(
                f"r2m-zenodo-audio-vibration-match-v1|seed={SEED}|participant="
                f"{participant_id}".encode("ascii")
            ).hexdigest(),
            participant_id,
        ),
    )
    splits: dict[str, list[int]] = {}
    cursor = 0
    for name, count in SPLIT_SIZES.items():
        splits[name] = sorted(ordered[cursor:cursor + count])
        cursor += count
    assignments = [
        {"participant_id": participant_id,
         "group_hash": hashlib.sha256(
             f"r2m-zenodo-audio-vibration-match-v1|seed={SEED}|participant="
             f"{participant_id}".encode("ascii")
         ).hexdigest(),
         "split": split_name}
        for split_name, group_ids in splits.items()
        for participant_id in group_ids
    ]
    payload: dict[str, object] = {
        "schema": "r2m_audio_vibration_participant_split_v1",
        "seed": SEED,
        "algorithm": SPLIT_ALGORITHM,
        "source_doi": "10.5281/zenodo.7937591",
        "group_unit": "participant; every participant/trial/device/crop/pair remains together",
        "participant_count": len(normalized),
        "splits": splits,
        "assignments": sorted(assignments, key=lambda row: row["participant_id"]),
        "sealed_audit_accessed": False,
        "signal_or_label_contents_read": False,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True).encode("utf-8")
    payload["manifest_sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload
