"""Deterministic, video-grouped split and option order for AV event selection."""

import hashlib


def select_training_event(candidate_records, video_id, sound_id):
    """Select one frozen TRAIN event by its provenance-only join key."""
    if not isinstance(video_id, str) or not video_id or type(sound_id) is not int or sound_id < 0:
        raise ValueError("a nonempty video ID and nonnegative integer sound ID are required")
    matches = [
        row
        for row in candidate_records
        if row.get("video_id") == video_id and row.get("sound_id") == sound_id
    ]
    if len(matches) != 1:
        raise ValueError("TRAIN event key must identify exactly one eligible event")
    return matches[0]


def model_selection_video_ids(video_ids, selection_count=60):
    ids = list(video_ids)
    if len(ids) != len(set(ids)):
        raise ValueError("video IDs must be unique")
    if selection_count < 1 or selection_count >= len(ids):
        raise ValueError("selection count must leave nonempty fit and selection groups")
    ordered = sorted(
        ids,
        key=lambda video_id: hashlib.sha256(
            f"perception-av-event-v1:model-selection:17:{video_id}".encode()
        ).digest(),
    )
    return set(ordered[:selection_count])


def deterministic_candidate_order(video_id, sound_id, track_ids):
    ids = list(track_ids)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("candidate track IDs must be nonempty and unique")
    if any(type(track_id) is not int or track_id < 0 for track_id in ids):
        raise ValueError("candidate track IDs must be nonnegative integers")
    return sorted(
        ids,
        key=lambda track_id: hashlib.sha256(
            f"perception-av-event-v1:17:{video_id}:{sound_id}:{track_id}".encode()
        ).digest(),
    )
