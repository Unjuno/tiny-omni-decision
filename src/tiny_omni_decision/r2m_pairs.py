"""Deterministic participant-safe audio/vibration matching pairs."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping

ALLOWED_SPLITS = frozenset({"train", "calibration", "validation"})
DEFAULT_SEED = 17
DEFAULT_MAX_DURATION_DELTA = 0.10


def sample_id(row: Mapping[str, object]) -> str:
    """Return the stable source identity for one activity/participant/trial."""
    activity = row.get("activity")
    participant = row.get("participant_id")
    trial = row.get("trial_id")
    if not isinstance(activity, str) or not activity or "/" in activity:
        raise ValueError("activity must be a nonempty archive-safe name")
    if type(participant) is not int or participant < 0:
        raise ValueError("participant_id must be a nonnegative integer")
    if type(trial) is not int or trial < 0:
        raise ValueError("trial_id must be a nonnegative integer")
    return f"{activity}|participant={participant}|trial={trial}"


def build_audio_vibration_pairs(
    records: Iterable[Mapping[str, object]],
    *,
    seed: int = DEFAULT_SEED,
    max_duration_delta: float = DEFAULT_MAX_DURATION_DELTA,
) -> dict[str, object]:
    """Build one deterministic same-activity negative for each eligible anchor.

    Each row describes one binary Decision item: its audio comes from the
    anchor trial, its positive vibration is from that same trial, and its
    negative vibration comes from a different trial with a duration-matched
    recording of the same activity. A cross-participant negative is preferred.
    Sealed-audit and quarantined rows are rejected rather than silently ignored.
    """
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (not isinstance(max_duration_delta, (int, float))
            or isinstance(max_duration_delta, bool)
            or not math.isfinite(max_duration_delta)
            or max_duration_delta < 0):
        raise ValueError("max_duration_delta must be finite and nonnegative")

    normalized: list[dict[str, object]] = []
    ids: set[str] = set()
    participant_splits: dict[int, str] = {}
    for source in records:
        if not isinstance(source, Mapping):
            raise ValueError("each record must be a mapping")
        split = source.get("split")
        if split not in ALLOWED_SPLITS:
            raise ValueError(f"record split {split!r} is outside the fitting scope")
        identifier = sample_id(source)
        participant = int(source["participant_id"])
        previous_split = participant_splits.setdefault(participant, str(split))
        if previous_split != split:
            raise ValueError(f"participant {participant} appears in multiple splits")
        if identifier in ids:
            raise ValueError(f"duplicate source trial: {identifier}")
        ids.add(identifier)
        samples = source.get("audio_samples")
        rate = source.get("audio_sample_rate_hz", 16000)
        if type(samples) is not int or samples <= 0:
            raise ValueError(f"{identifier} has invalid audio_samples")
        if (not isinstance(rate, (int, float)) or isinstance(rate, bool)
                or not math.isfinite(rate) or rate <= 0):
            raise ValueError(f"{identifier} has invalid audio sample rate")
        row = dict(source)
        row["sample_id"] = identifier
        row["duration_seconds"] = samples / float(rate)
        normalized.append(row)

    buckets: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in normalized:
        buckets[(str(row["split"]), str(row["activity"]))].append(row)

    pair_rows: list[dict[str, object]] = []
    for bucket_key in sorted(buckets):
        bucket = sorted(buckets[bucket_key], key=lambda row: str(row["sample_id"]))
        usage: Counter[str] = Counter()
        for anchor in bucket:
            anchor_id = str(anchor["sample_id"])
            anchor_duration = float(anchor["duration_seconds"])
            candidates = []
            for candidate in bucket:
                candidate_id = str(candidate["sample_id"])
                if candidate_id == anchor_id:
                    continue
                if candidate["participant_id"] == anchor["participant_id"] and (
                    candidate["trial_id"] == anchor["trial_id"]
                ):
                    continue
                delta = abs(float(candidate["duration_seconds"]) - anchor_duration)
                relative_delta = delta / max(anchor_duration, 1e-12)
                if relative_delta > max_duration_delta:
                    continue
                is_cross_participant = (
                    candidate["participant_id"] != anchor["participant_id"]
                )
                tie = hashlib.sha256(
                    f"r2m-av-pair-v1|seed={seed}|{anchor_id}|{candidate_id}".encode()
                ).hexdigest()
                candidates.append((
                    0 if is_cross_participant else 1,
                    usage[candidate_id],
                    relative_delta,
                    tie,
                    candidate,
                ))
            if not candidates:
                continue
            _, _, relative_delta, _, negative = min(candidates, key=lambda item: item[:4])
            negative_id = str(negative["sample_id"])
            usage[negative_id] += 1
            pair_id = hashlib.sha256(
                f"r2m-av-pair-v1|seed={seed}|{bucket_key}|{anchor_id}".encode()
            ).hexdigest()
            pair_rows.append({
                "pair_id": pair_id,
                "split": bucket_key[0],
                "activity": bucket_key[1],
                "anchor_sample_id": anchor_id,
                "positive_sensor_sample_id": anchor_id,
                "negative_sensor_sample_id": negative_id,
                "negative_participant_relation": (
                    "cross_participant" if negative["participant_id"] !=
                    anchor["participant_id"] else "within_participant"
                ),
                "negative_duration_delta_fraction": round(relative_delta, 9),
                "positive_target_option": "same_trial",
            })

    pair_rows.sort(key=lambda row: (str(row["split"]), str(row["pair_id"])))
    counts = Counter(str(row["split"]) for row in pair_rows)
    relations = Counter(
        f"{row['split']}:{row['negative_participant_relation']}" for row in pair_rows
    )
    payload: dict[str, object] = {
        "schema": "r2m_audio_vibration_pair_index_v1",
        "seed": seed,
        "max_duration_delta_fraction": float(max_duration_delta),
        "negative_policy": (
            "same activity, different trial, duration matched; cross-participant "
            "preferred; reuse minimized"
        ),
        "split_scope": sorted(ALLOWED_SPLITS),
        "sealed_audit_accessed": False,
        "quarantined_rows_allowed": False,
        "input_trial_count": len(normalized),
        "pair_count_by_split": dict(sorted(counts.items())),
        "negative_relation_count": dict(sorted(relations.items())),
        "unpaired_trial_count_by_split": dict(sorted({
            split: sum(1 for row in normalized if row["split"] == split) - counts[split]
            for split in sorted(ALLOWED_SPLITS)
        }.items())),
        "pairs": pair_rows,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=True).encode("utf-8")
    payload["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload
