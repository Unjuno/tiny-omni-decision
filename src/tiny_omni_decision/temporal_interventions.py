"""Deterministic held-out feature interventions for temporal AV readouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FeatureIntervention:
    video: np.ndarray
    audio: np.ndarray
    video_present: np.ndarray
    audio_present: np.ndarray
    eligible_session_count: int = 0
    shuffle_group_count: int = 0


INTERVENTIONS = (
    "baseline",
    "audio_missing",
    "video_missing",
    "audio_reverse_time",
    "video_reverse_time",
    "audio_shuffle_session",
)


def apply_feature_intervention(
    data: dict[str, Any], intervention: str, *, seed: int = 17
) -> FeatureIntervention:
    """Return a non-mutating feature view with a fixed AV intervention.

    Cross-session audio shuffling is restricted to sessions with the same task
    category and exact number of windows, so the intervention changes the
    session-to-audio pairing while retaining each recipient's label/category
    and temporal shape.
    """
    if intervention not in INTERVENTIONS:
        raise ValueError(f"unknown feature intervention: {intervention}")
    video = data["video"]
    audio = data["audio"]
    if (
        video.ndim != 2
        or audio.ndim != 2
        or len(video) != len(audio)
        or len(data["sample_id"]) != len(video)
    ):
        raise ValueError("feature arrays must share the sample-ID axis")
    video_present = np.ones(len(video), dtype=np.bool_)
    audio_present = np.ones(len(audio), dtype=np.bool_)
    eligible_session_count = 0
    shuffle_group_count = 0

    if intervention == "audio_missing":
        audio = np.zeros_like(audio)
        audio_present.fill(False)
    elif intervention == "video_missing":
        video = np.zeros_like(video)
        video_present.fill(False)
    elif intervention == "audio_reverse_time":
        audio = audio.copy()
        for indices in data["session_indices"]:
            audio[indices] = data["audio"][indices[::-1]]
    elif intervention == "video_reverse_time":
        video = video.copy()
        for indices in data["session_indices"]:
            video[indices] = data["video"][indices[::-1]]
    elif intervention == "audio_shuffle_session":
        audio = audio.copy()
        groups: dict[tuple[int, str], list[np.ndarray]] = {}
        for indices in data["session_indices"]:
            task_type = str(data["task_type"][indices[0]])
            groups.setdefault((len(indices), task_type), []).append(indices)
        generator = np.random.default_rng(seed)
        for group_key in sorted(groups):
            sessions = sorted(groups[group_key], key=lambda row: str(data["session_id"][row[0]]))
            if len(sessions) < 2:
                continue
            offset = int(generator.integers(1, len(sessions)))
            for position, recipient in enumerate(sessions):
                donor = sessions[(position + offset) % len(sessions)]
                audio[recipient] = data["audio"][donor]
            eligible_session_count += len(sessions)
            shuffle_group_count += 1

    return FeatureIntervention(
        video=video,
        audio=audio,
        video_present=video_present,
        audio_present=audio_present,
        eligible_session_count=eligible_session_count,
        shuffle_group_count=shuffle_group_count,
    )
