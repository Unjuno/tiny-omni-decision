"""Select video-disjoint, train-only rows for a fixed visibility transfer probe."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Set


def _stable_rank(*parts: object) -> str:
    return hashlib.sha256(":".join(str(part) for part in parts).encode("utf-8")).hexdigest()


def select_unseen_events(
    annotations: Mapping[str, Mapping[str, object]],
    *,
    excluded_video_ids: Set[str],
    source_split: str,
    seed: int,
    events_per_class: int,
    context_seconds: float = 0.25,
) -> list[dict[str, object]]:
    """Choose one boundary-qualified event per unseen training video.

    Video groups are selected before label stratification. This avoids taking
    multiple events from one source video or choosing the easiest event by its
    label. The function fails closed if the requested class balance is not
    available in the source-training split.
    """
    if source_split != "train":
        raise ValueError("train split only")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(events_per_class) is not int or events_per_class < 1:
        raise ValueError("events_per_class must be a positive integer")
    if not math.isfinite(context_seconds) or context_seconds < 0:
        raise ValueError("context_seconds must be finite and nonnegative")
    if any(not isinstance(video_id, str) or not video_id for video_id in excluded_video_ids):
        raise ValueError("excluded video IDs must be nonempty strings")

    selected_per_video: list[tuple[str, dict[str, object]]] = []
    context_us = context_seconds * 1_000_000
    for video_id in sorted(annotations):
        if not isinstance(video_id, str) or not video_id:
            raise ValueError("video IDs must be nonempty strings")
        if video_id in excluded_video_ids:
            continue
        row = annotations[video_id]
        metadata = row.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError(f"missing metadata for {video_id}")
        if metadata.get("video_id", video_id) != video_id:
            raise ValueError(f"annotation video ID mismatch for {video_id}")
        audio_rate = metadata.get("audio_sample_rate")
        frame_rate = metadata.get("frame_rate")
        audio_samples = metadata.get("audio_samples")
        num_frames = metadata.get("num_frames")
        if (
            not isinstance(audio_rate, (int, float))
            or not math.isfinite(audio_rate)
            or audio_rate <= 0
            or type(audio_samples) is not int
            or audio_samples <= 0
            or not isinstance(frame_rate, (int, float))
            or not math.isfinite(frame_rate)
            or frame_rate <= 0
            or type(num_frames) is not int
            or num_frames <= 0
        ):
            raise ValueError(f"invalid media duration metadata for {video_id}")
        audio_duration = audio_samples / audio_rate
        video_duration = num_frames / frame_rate
        available_us = min(audio_duration, video_duration) * 1_000_000
        sounds = row.get("sound_localisation")
        if not isinstance(sounds, list):
            raise ValueError(f"missing sound annotations for {video_id}")
        event_ids: set[int] = set()
        eligible: list[dict[str, object]] = []
        for sound in sounds:
            if not isinstance(sound, Mapping):
                raise ValueError(f"invalid sound annotation for {video_id}")
            event_id = sound.get("id")
            target = sound.get("is_visible")
            timestamps = sound.get("timestamps")
            if type(event_id) is not int or event_id in event_ids:
                raise ValueError(f"invalid or duplicate event ID for {video_id}")
            event_ids.add(event_id)
            if type(target) is not int or target not in (0, 1):
                raise ValueError(f"binary visibility target required for {video_id}")
            if (
                not isinstance(timestamps, (list, tuple))
                or len(timestamps) != 2
                or any(
                    not isinstance(value, (int, float)) or not math.isfinite(value)
                    for value in timestamps
                )
            ):
                raise ValueError(f"invalid event timestamps for {video_id}")
            start_us, end_us = timestamps
            if start_us < 0 or end_us <= start_us:
                raise ValueError(f"invalid event interval for {video_id}")
            if start_us < context_us or end_us + context_us > available_us:
                continue
            eligible.append(
                {
                    "split": "train_transfer",
                    "video_id": video_id,
                    "event_id": event_id,
                    "target": target,
                    "start_event_s": start_us / 1_000_000,
                    "end_event_s": end_us / 1_000_000,
                    "audio_duration_s": audio_duration,
                    "video_duration_s": video_duration,
                }
            )
        if eligible:
            event = min(
                eligible,
                key=lambda value: _stable_rank(
                    "perception-visibility-transfer-event-v1",
                    seed,
                    video_id,
                    value["event_id"],
                ),
            )
            selected_per_video.append((video_id, event))

    by_target: dict[int, list[dict[str, object]]] = {0: [], 1: []}
    for video_id, event in selected_per_video:
        target = event["target"]
        by_target[target].append(
            event
            | {
                "selection_rank": _stable_rank(
                    "perception-visibility-transfer-cohort-v1",
                    seed,
                    target,
                    video_id,
                    event["event_id"],
                )
            }
        )
    for target, events in by_target.items():
        if len(events) < events_per_class:
            raise ValueError(
                "insufficient unseen groups for class "
                f"{target}: requested {events_per_class}, found {len(events)}"
            )
        events.sort(key=lambda event: event["selection_rank"])
        by_target[target] = [event for event in events[:events_per_class]]
    result = [event for target in (0, 1) for event in by_target[target]]
    if len({event["video_id"] for event in result}) != len(result):
        raise RuntimeError("selected video groups are not unique")
    if any(event["video_id"] in excluded_video_ids for event in result):
        raise RuntimeError("selected video overlaps excluded fit groups")
    return [
        {key: value for key, value in event.items() if key != "selection_rank"}
        for event in result
    ]


def build_inference_manifest(
    selected: list[Mapping[str, object]],
) -> dict[str, list[dict[str, object]]]:
    """Adapt frozen one-event rows to the existing matched-feature extractors."""
    rows: list[dict[str, object]] = []
    seen_videos: set[str] = set()
    for event in selected:
        if event.get("split") != "train_transfer":
            raise ValueError("train-only transfer rows are required")
        video_id = event.get("video_id")
        event_id = event.get("event_id")
        target = event.get("target")
        start = event.get("start_event_s")
        end = event.get("end_event_s")
        if not isinstance(video_id, str) or not video_id:
            raise ValueError("video ID must be a nonempty string")
        if video_id in seen_videos:
            raise ValueError("one event per video is required")
        seen_videos.add(video_id)
        if type(event_id) is not int or type(target) is not int or target not in (0, 1):
            raise ValueError("event ID and binary target are required")
        if (
            not isinstance(start, (int, float))
            or not isinstance(end, (int, float))
            or not math.isfinite(start)
            or not math.isfinite(end)
            or start < 0
            or end <= start
        ):
            raise ValueError("valid event timestamps are required")
        center = (start + end) / 2.0
        window = [center - 1.0, center + 1.0]
        if window[0] < 0:
            raise ValueError("event does not have the full centered 2-second audio/video window")
        frame_times = [window[0] + index * 2.0 / 7 for index in range(8)]
        rows.append(
            {
                "sample_id": f"{video_id}__visibility_{event_id}",
                "split": "train",
                "video_id": video_id,
                "video_event_id": event_id,
                "video_window_s": window,
                "video_frame_times_s": frame_times,
                "target": target,
                "target_index": 0,
                "options": [
                    {
                        "event_id": event_id,
                        "role": "target",
                        "audio_window_s": window,
                    }
                ],
            }
        )
    if not rows:
        raise ValueError("transfer cohort must not be empty")
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate sample IDs")
    return {"train": rows, "validation": []}


def merge_media_receipts(
    video_ids: Set[str],
    prior_files: list[Mapping[str, object]],
    new_files: list[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Combine verified prior/new media records without changing cohort membership."""
    prior = {
        (row.get("video_id"), row.get("member", "").rsplit(".", 1)[-1]): row
        for row in prior_files
    }
    new = {
        (row.get("video_id"), row.get("member", "").rsplit(".", 1)[-1]): row
        for row in new_files
    }
    merged: list[dict[str, object]] = []
    for video_id in sorted(video_ids):
        for extension in ("mp4", "wav"):
            record = new.get((video_id, extension)) or prior.get((video_id, extension))
            if record is None or not record.get("sha256"):
                raise ValueError(f"missing verified {extension} receipt for {video_id}")
            merged.append(dict(record))
    if len(merged) != len(video_ids) * 2:
        raise ValueError("merged media receipt is incomplete")
    return merged
