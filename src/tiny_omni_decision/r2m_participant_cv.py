"""Deterministic participant-grouped outer CV plans for R2-M experiments."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping, Sequence


def leave_one_participant_out_plan(
    rows: Sequence[Mapping[str, object]], *, seed: int = 17,
) -> list[dict[str, object]]:
    """Plan outer participant holdouts with a label-complete inner calibration group.

    The input must already be restricted to an admitted training pool. For every
    outer fold, all labels must occur in fit, calibration, and held-out rows
    across the complete OOF run. Calibration participants are chosen only from
    the outer training pool and balanced deterministically across folds.
    """
    if not rows:
        raise ValueError("participant CV requires nonempty training rows")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")

    participants: dict[int, set[str]] = {}
    samples_by_participant: dict[int, set[str]] = {}
    labels = set()
    all_sample_ids = set()
    for row in rows:
        participant = int(row["participant_id"])
        label = str(row["activity"])
        sample_id = str(row["sample_id"])
        if not label or not sample_id:
            raise ValueError("each row needs a nonempty activity and sample_id")
        if sample_id in all_sample_ids:
            raise ValueError(f"duplicate sample ID in participant CV: {sample_id}")
        all_sample_ids.add(sample_id)
        participants.setdefault(participant, set()).add(label)
        samples_by_participant.setdefault(participant, set()).add(sample_id)
        labels.add(label)

    if len(participants) < 3 or len(labels) < 2:
        raise ValueError("participant CV requires at least three participants and two labels")

    ordered_participants = sorted(participants)
    calibration_uses = Counter()
    plan: list[dict[str, object]] = []
    validation_ids: set[str] = set()
    for outer_participant in ordered_participants:
        development_participants = [
            participant for participant in ordered_participants
            if participant != outer_participant
        ]
        calibration_candidates = []
        for candidate in development_participants:
            fit_participants = [
                participant for participant in development_participants
                if participant != candidate
            ]
            fit_labels = set().union(*(participants[p] for p in fit_participants))
            if participants[candidate] >= labels and fit_labels >= labels:
                tie_break = hashlib.sha256(
                    f"{seed}:{outer_participant}:{candidate}".encode()
                ).hexdigest()
                calibration_candidates.append(
                    (calibration_uses[candidate], tie_break, candidate, fit_participants)
                )
        if not calibration_candidates:
            raise ValueError(
                "no participant-only calibration group preserves all labels "
                f"for outer participant {outer_participant}"
            )
        _, _, calibration_participant, fit_participants = min(calibration_candidates)
        calibration_uses[calibration_participant] += 1
        train_ids = sorted(set().union(*(samples_by_participant[p] for p in fit_participants)))
        calibration_ids = sorted(samples_by_participant[calibration_participant])
        validation_ids_for_fold = sorted(samples_by_participant[outer_participant])
        if (set(train_ids) & set(calibration_ids)
                or set(train_ids) & set(validation_ids_for_fold)
                or set(calibration_ids) & set(validation_ids_for_fold)):
            raise AssertionError("participant CV plan has sample overlap")
        if validation_ids & set(validation_ids_for_fold):
            raise AssertionError("outer validation sample appears in more than one fold")
        validation_ids.update(validation_ids_for_fold)
        plan.append({
            "outer_participant": outer_participant,
            "train_participants": fit_participants,
            "calibration_participant": calibration_participant,
            "validation_participant": outer_participant,
            "train_sample_ids": train_ids,
            "calibration_sample_ids": calibration_ids,
            "validation_sample_ids": validation_ids_for_fold,
        })

    if validation_ids != all_sample_ids:
        raise AssertionError("outer folds do not cover every admitted training sample once")
    return plan
